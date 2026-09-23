"""Plain-English Telegram panel text; stored values and action IDs stay intact."""
from __future__ import annotations

from datetime import UTC, datetime

from hub.telegram_admin_state import _safe


def label(key):
    return str(key).replace('_', ' ').replace('.', ' · ').capitalize()


def display(value):
    value = _safe(value)
    if value is None or value == '':
        return 'Not set'
    if type(value) is bool:
        return 'Yes' if value else 'No'
    if isinstance(value, dict):
        return '\n'.join(f'{label(key)}: {display(item)}' for key, item in value.items()) or 'No details'
    if isinstance(value, (list, tuple)):
        return '\n'.join('• ' + display(item) for item in value) or 'None'
    return str(value)


def status_text(result):
    if result.get('ok') is False:
        return 'Status\n' + display(result.get('error'))
    places = result.get('workplaces', [])
    online = [row for row in places if row.get('connected')]
    lines = ['Rowan status', '', f'Computers online: {len(online)}',
             'AI model: ' + display(result.get('llm_model')),
             'People profiles: ' + display(result.get('profiles')),
             'Voice access checks: ' + ('Enabled' if result.get('permissions_enabled') else 'Disabled')]
    budget = result.get('api_usage')
    if isinstance(budget, dict):
        amount, limit = budget.get('accounted_usd'), budget.get('limit_usd')
        lines += ['', 'API accounting · ' + str(budget.get('month') or 'this month'),
                  'Local estimates, not provider billing.']
        for provider in budget.get('providers', []):
            lines.append(f"{provider['provider']}: ${provider['settled_estimate_usd']:.2f} estimated")
            if provider.get('reserved_usd'):
                lines.append(f"  Reserved / unconfirmed: ${provider['reserved_usd']:.2f} ({provider.get('unsettled_requests', 0)} requests)")
        settled = budget.get('settled_estimate_usd')
        if isinstance(settled, (int, float)):
            lines.append(f'Total usage estimate: ${settled:.2f}')
        if isinstance(amount, (int, float)) and isinstance(limit, (int, float)):
            lines.append(f'Budget counted incl. reserves: ${amount:.2f} / ${limit:.2f}')
        elif isinstance(amount, (int, float)):
            # The owner removed the monthly ceiling (DECISIONS.md API-01).
            lines.append(f'Budget counted incl. reserves: ${amount:.2f} / no monthly limit')
        if budget.get('unsettled_requests'):
            lines.append('Unconfirmed reserves include requests with missing usage or interrupted responses.')
    alerts = result.get('notifications') or {}
    lines += ['', 'Notification service: ' + ('Running' if alerts.get('running') else 'Stopped'),
              'Notifications pending: ' + display(alerts.get('pending', 0))]
    if online:
        lines += ['', 'Connected computers:']
        lines += ['• ' + display(row.get('name') or row.get('id')) for row in online[:12]]
        if len(online) > 12:
            lines.append('See Computers and cameras for the full list.')
    return '\n'.join(lines)


def profile_text(item):
    return '\n'.join(['Person profile', '', 'Name: ' + display(item.get('name') or item.get('id')),
                      'Role: ' + label(item.get('role', 'user')),
                      'Voice samples: ' + display(item.get('voice_samples', 0)),
                      'Face samples: ' + display(item.get('face_samples', 0)), '',
                      'Resetting or deleting this profile keeps the original recordings and archive.'])


def user_text(user, capabilities):
    lines = ['Telegram user', '', 'Name: ' + display(user.get('label') or user['user_id']),
             'Telegram ID: ' + str(user['user_id']), 'Role: ' + label(user['role']), '', 'Permissions:']
    lines += [('✓ ' if allowed else '— ') + label(key) for key, allowed in capabilities.items()]
    return '\n'.join(lines)


def rule_value(key, value, *, private_to=0, chats=None):
    """One rule setting in words; ``chats`` names the groups the bot has seen.

    ``private_to`` is how many accounts "Private chat" delivers to, so the
    panel can say plainly where a notification goes instead of leaving the
    owner to guess what "owner" means.
    """
    options = {
        'target': {'any': 'Any person', 'unknown': 'Unknown face', 'person': 'Specific person'},
        'media': {'photo': 'Photo', 'video': 'Video'},
        'destination': {'owner': 'Private chat (everyone)', 'group': 'Notifications group chat'},
        # ТЗ F-702: событие и канал доставки — словами, а не кодами.
        'event': {'presence': 'Someone in frame', 'person_entered': 'Person came in',
                  'person_left': 'Person left', 'unknown_appeared': 'Unknown face appeared',
                  'zone_entered': 'Zone changed', 'sound_event': 'Sound event',
                  'object': 'Object of interest'},
        'channel': {'telegram': 'Telegram', 'push': 'Phone push', 'hud': 'HUD caption'},
        # ТЗ F-702: «снимать, пока человек в кадре» — переключатель, а не число.
        'record_until_clear': {True: 'On', False: 'Off'},
    }
    if key in options:
        if key != 'destination':
            return options[key].get(value, display(value))
        return destination_value(value, private_to=private_to, chats=chats)
    if key == 'workplace_id' and not value:
        return 'All computers'
    if key == 'home_id' and not value:
        return 'Every home'
    if key == 'zone' and not value:
        return 'Any zone'
    if key in {'quiet_start', 'quiet_end'} and not value:
        return 'Off'
    return display(value)


def destination_value(value, *, private_to=0, chats=None):
    """Where a notification goes: the group it names, or every private chat."""
    text = str(value or '')
    if text == 'owner':
        return 'Private chat (everyone)' + (f' · {private_to}' if private_to else '')
    named = (chats or {}).get(text)
    if named:
        return str(named)
    if text == 'group':
        return 'Notifications group chat'
    if text.startswith('group:'):
        return 'Group ' + text.split(':', 1)[1]
    return display(value)


def rule_text(item, fields, *, draft=False, chats=None, private_to=0):
    lines = ['Notification draft' if draft else 'Presence notification', '',
             'State: ' + ('Enabled' if item.get('enabled') else 'Disabled')]
    lines += [f'{caption}: {rule_value(key, item.get(key), private_to=private_to, chats=chats)}'
              for key, (caption, _) in fields.items()]
    if draft:
        lines += ['', 'Changes will not take effect until saved.']
    return '\n'.join(lines)


def audit_text(events):
    lines = ['Recent panel activity']
    if not events:
        return lines[0] + '\nNo actions recorded yet.'
    for row in events:
        stamp = row.get('ts')
        when = datetime.fromtimestamp(stamp, UTC).strftime('%b %d, %H:%M UTC') if isinstance(stamp, (int, float)) else ''
        lines += ['', f'{when} · {label(row.get("event", "Action"))}',
                  'By Telegram user ' + str(row.get('actor')), display(row.get('details') or {})]
    return '\n'.join(lines)


def rules_text(result):
    """ТЗ F-419: список правил словами — ни одного JSON-поля на экране."""
    lines = ['Room rules · when the room does something by itself']
    if result.get('ok') is False:
        return lines[0] + '\n' + display(result.get('error'))
    items = list(result.get('items') or [])
    if not items:
        lines += ['', 'No rules yet. Say, for example: when I come home after 22:00, '
                      'turn on the warm light.']
        return '\n'.join(lines)
    lines.append(f'Rules: {len(items)}')
    for item in items:
        mark = '✓' if item.get('enabled') else '—'
        lines.append(f'{mark} {display(item.get("words") or item.get("name") or item.get("id"))}')
    lines += ['', '✓ means the rule is on; — means it is off.']
    return '\n'.join(lines)


def automation_rule_text(item):
    """Одно правило словами: когда, если и что оно делает (ТЗ F-419)."""
    mark = 'on' if item.get('enabled') else 'off'
    return f'Rule ({mark})\n{display(item.get("words") or item.get("id"))}'


def calibration_text(result):
    """The weekly calibration report of ТЗ 5.4: errors per type and provider."""
    days = result.get('days', 7)
    lines = [f'Decision calibration · last {display(days)} day(s)']
    if result.get('ok') is False:
        return lines[0] + '\n' + display(result.get('error'))
    totals = result.get('totals') or {}
    share = totals.get('error_share')
    checked = display(totals.get('observed', 0))
    errors = display(totals.get('errors', 0))
    lines += ['', 'Decisions: ' + display(totals.get('decisions', 0)),
              f'Checked against what happened: {checked}',
              f'Errors: {errors}' + ('' if share is None else f' · {share:.1%} of checked'),
              'A decision counts as an error when the turn contradicted it: the router promised a local command and none matched, or the self-check found the reply was right after all.']
    if share is None:
        lines.append('Nothing has been checked against what happened yet, so there is no error share to report.')
    rows = result.get('rows') or []
    if not rows:
        lines += ['', 'No decisions were recorded in this window.']
        return '\n'.join(lines)
    lines.append('')
    for row in rows[:12]:
        row_share = row.get('error_share')
        lines += [f"{label(row.get('type'))} · {display(row.get('provider'))}",
                  f"  decisions {display(row.get('decisions'))} · checked {display(row.get('observed'))}"
                  f" · errors {display(row.get('errors'))}"
                  + ('' if row_share is None else f' ({row_share:.1%})'),
                  f"  mean confidence {row.get('mean_confidence')} · mean latency {row.get('mean_latency_ms')} ms"]
    if len(rows) > 12:
        lines.append(f'Showing 12 of {len(rows)} type/provider pairs.')
    return '\n'.join(lines)


def switches_text(result):
    """The ESP32 wall switches and their servo angles (ТЗ F-503)."""
    lines = ['Wall switches']
    if result.get('ok') is False:
        return lines[0] + '\n' + display(result.get('error'))
    items = result.get('items') or []
    if not items:
        return (lines[0] + '\nNo switch is registered yet. Add an ESP32 switch to a home, '
                'then calibrate its servo here.')
    for item in items:
        state = 'calibrated' if item.get('calibrated') else 'factory angles'
        lines += ['', f"{item.get('name')} · {item.get('home_id')}",
                  f"  Adapter {display(item.get('adapter'))} · switch {display(item.get('switch'))}"
                  f" · {state}",
                  f"  Closed {display(item.get('closed_angle'))}° · open "
                  f"{display(item.get('open_angle'))}° · hold {display(item.get('dwell_s'))} s"]
    return '\n'.join(lines)


def scenes_text(result):
    """The scenes of this home and what they do (ТЗ F-506)."""
    lines = ['Scenes']
    if result.get('ok') is False:
        return lines[0] + '\n' + display(result.get('error'))
    items = result.get('items') or []
    if not items:
        return lines[0] + '\nNo scenes yet. The five presets appear on the next refresh.'
    if result.get('created'):
        lines.append('Presets created: ' + ', '.join(str(name) for name in result['created']))
    for item in items:
        lines += ['', f"{item.get('name')}{' (preset)' if item.get('preset') else ''}",
                  f"  {display(item.get('steps'))} step(s): {str(item.get('summary') or '')[:180]}"]
        if item.get('aliases'):
            lines.append('  Also called: ' + ', '.join(str(alias) for alias in item['aliases']))
    return '\n'.join(lines)
