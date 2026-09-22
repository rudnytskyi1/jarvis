"""Per-person application choices, based only on the client's actual inventory."""
import asyncio
import json
import re
import time

from hub.decision_points import names_a_site_step

BROWSERS = {'google chrome': {'chrome', 'google chrome', 'хром'},
            'microsoft edge': {'edge', 'microsoft edge', 'эдж'},
            'mozilla firefox': {'firefox', 'mozilla firefox', 'файрфокс'},
            'brave': {'brave', 'brave browser'}, 'opera': {'opera'}, 'vivaldi': {'vivaldi'}}


def plain(text):
    text = text.casefold().strip(' .!?,')
    text = re.sub(r'^(?:(?:hey|okay|ok)\s+)?(?:rowan|роуан)[\s,.:!]+', '', text)
    return re.sub(r'^(?:can you |could you |please |пожалуйста,? )', '', text).strip(' .!?,')


def app_intent(text):
    value = plain(text)
    match = re.fullmatch(r'(open|close|открой|закрой)\s+(?:the\s+)?([\w -]{1,60}?)(?:,?\s+please)?', value)
    if not match or re.search(r'\b(?:and|then|и|потом|after)\b', match[2]):
        return None
    name = match[2]
    if name not in {'browser', 'web browser', 'браузер', 'spotify', 'steam', 'notepad', 'calculator'} and not any(name in aliases for aliases in BROWSERS.values()):
        return None
    return ('open' if match[1] in {'open', 'открой'} else 'close'), name


def equivalent(a, b):
    a, b = a.casefold(), b.casefold()
    return a == b or any(a in names | {label} and b in names | {label} for label, names in BROWSERS.items())


class ApplicationChoices:
    def __init__(self):
        self.pending = {}
        self.last = {}

    def remember_offer(self, connection, name):
        """Offer explicit preference saving using the existing speaker policy.

        Never in the middle of a longer request. "Open chrome and go to
        youtube" is two steps, and this one sentence - "say Rowan, remember
        this browser" - was what the model read back instead of going to the
        site: it looks like a finished turn. When the same utterance still asks
        for a page that has to be reached, staying silent is the whole fix.
        """
        if names_a_site_step(getattr(connection, '_utterance_text', '')):
            return ''
        owner = connection._known_speaker_name().casefold()
        open_access = not connection._permissions_enabled
        if not owner and not open_access:
            return ''
        self.last[owner] = {'name': name, 'expires': time.monotonic() + 180}
        if owner:
            offer = 'Say Rowan, remember this browser for me'
            return offer + (', or for everyone.' if open_access or connection._speaker_role == 'admin'
                            else ', to save your choice.')
        return 'Say Rowan, remember this browser for everyone, to save your choice.'

    def observe_browser_result(self, connection, args, result):
        """Retain a confirmed explicit desktop-browser choice without saving it."""
        if result.get('ok') is not True or result.get('needs_choice'):
            return result
        browser = args.get('browser')
        selected_browser = isinstance(browser, str) and browser.strip().casefold() not in {
            '', 'browser', 'web browser', 'internet browser', 'браузер'}
        window = args.get('window_ref')
        selected_window = isinstance(window, str) and bool(window.strip())
        if not selected_browser and not selected_window:
            return result
        try:
            output = result.get('output', {})
            snapshot = json.loads(output) if isinstance(output, str) else output
        except (ValueError, TypeError):
            return result
        if not isinstance(snapshot, dict) or snapshot.get('needs_choice') or snapshot.get('ok') is False:
            return result
        name = snapshot.get('browser')
        if (not isinstance(name, str) or not name.strip()
                or not isinstance(snapshot.get('url'), str)
                or not isinstance(snapshot.get('title'), str)
                or not isinstance(snapshot.get('elements'), list)):
            return result
        offer = self.remember_offer(connection, name)
        return {**result, 'remember_offer': offer} if offer else result

    async def run(self, connection, memory, action, name):
        owner = connection._known_speaker_name().casefold()
        outcome = await connection._run_client_action('app_action', {'operation': 'inspect', 'action': action, 'name': name})
        if not outcome.get('ok'):
            return outcome
        try:
            candidates = json.loads(outcome.get('output') or '{}').get('candidates', [])
        except (ValueError, TypeError):
            return {'ok': False, 'error': 'Could not read the application inventory.'}
        if not candidates:
            return {'ok': True, 'reply': f'I found no {"open" if action == "close" else "installed"} application matching {name}.'}
        browser = name.casefold() in {'browser', 'web browser', 'браузер'} or all(any(equivalent(c['name'], b) for b in BROWSERS) for c in candidates)
        if action == 'open' and name.casefold() in {'browser', 'web browser', 'браузер'} and memory:
            saved = await asyncio.to_thread(memory.preference, 'apps.browser', owner)
            preferred = [c for c in candidates if saved and equivalent(c['name'], str(saved['value']))]
            if len(preferred) == 1:
                candidates = preferred
        if len(candidates) > 1:
            self.pending[owner] = {'action': action, 'names': [c['name'] for c in candidates], 'expires': time.monotonic() + 180}
            names = ', '.join(c['name'] for c in candidates)
            return {'ok': True, 'needs_choice': True, 'reply': f'Which should I {action}: {names}? Start your answer with Rowan.'}
        self.pending.pop(owner, None)
        choice = candidates[0]
        result = await connection._run_client_action('app_action', {'operation': 'execute', 'target_id': choice['id']})
        if not result.get('ok'):
            return result
        try:
            details = json.loads(result.get('output') or '{}')
        except (ValueError, TypeError):
            return {'ok': False, 'error': 'Could not verify the application action.'}
        if not details.get('completed'):
            return {'ok': False, 'error': details.get('note') or 'The application did not confirm completion.'}
        reply = f'{"Opened" if action == "open" else "Closed"} {choice["name"]}.'
        offer = self.remember_offer(connection, choice['name']) if browser else ''
        if offer:
            reply += ' ' + offer
        return {'ok': True, 'reply': reply}

    async def followup(self, connection, text):
        owner = connection._known_speaker_name().casefold()
        pending = self.pending.get(owner)
        value = plain(text)
        if pending and pending['expires'] > time.monotonic():
            if value in {'cancel', 'never mind', 'отмена'}:
                self.pending.pop(owner, None)
                return 'Application selection cancelled.'
            chosen = [name for name in pending['names'] if equivalent(name, value)]
            if len(chosen) == 1:
                result = await connection._execute_tool('pc_control', {'command': pending['action'] + '_app', 'value': chosen[0]})
                return result.get('reply') or result.get('error') or 'The action did not complete.'
        previous = self.last.get(owner)
        match = re.fullmatch(r'(?:remember|save) (?:this|that|the) (?:browser|choice)(?: (?:for (me|everyone)|globally))?', value)
        if previous and previous['expires'] > time.monotonic() and match:
            scope = 'global' if match[1] == 'everyone' or value.endswith('globally') else 'personal'
            result = await connection._execute_tool('remember', {'fact': 'Preferred browser: ' + previous['name'],
                'key': 'apps.browser', 'value': previous['name'], 'scope': scope})
            return ('Saved for everyone.' if scope == 'global' else 'Saved for you.') if result.get('ok') else result.get('reply') or result.get('error')
        return None
