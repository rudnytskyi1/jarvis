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


def rule_value(key, value):
    options = {
        'target': {'any': 'Any person', 'unknown': 'Unknown face', 'person': 'Specific person'},
        'media': {'photo': 'Photo', 'video': 'Video'},
        'destination': {'owner': 'Owner private chat', 'group': 'Group chat'},
    }
    if key in options:
        return options[key].get(value, display(value))
    if key == 'workplace_id' and not value:
        return 'All computers'
    if key in {'quiet_start', 'quiet_end'} and not value:
        return 'Off'
    return display(value)


def rule_text(item, fields, *, draft=False):
    lines = ['Notification draft' if draft else 'Presence notification', '',
             'State: ' + ('Enabled' if item.get('enabled') else 'Disabled')]
    lines += [f'{caption}: {rule_value(key, item.get(key))}' for key, (caption, _) in fields.items()]
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
