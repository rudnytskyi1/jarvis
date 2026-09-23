"""Owner settings validate full config, exclude secrets and survive restart."""
import json
from types import SimpleNamespace

import pytest

from common.config import Config
from hub.admin_settings import IMMUTABLE, apply_live, catalogue, restore_overrides, validated_value
from hub.telegram_admin_state import TelegramAdminState

OWNER = 8322835915


def test_catalogue_excludes_credentials_endpoints_paths_and_client_writes():
    cfg = Config()
    cfg.server.llm.api_key = 'synthetic-credential-never-render'
    cfg.server.llm.base_url = 'https://private-endpoint.invalid/v1'
    cfg.client.camera.stream_url = 'rtsp://synthetic:credential@camera.invalid/live'
    rows = catalogue(cfg)
    by_key = {row['key']: row for row in rows}
    serialized = json.dumps(rows, ensure_ascii=False)
    assert 'synthetic-credential' not in serialized and 'private-endpoint.invalid' not in serialized
    assert 'rtsp://' not in serialized
    for key in ('server.llm.api_key', 'server.llm.api_key_env', 'server.llm.base_url',
                'server.telegram.api_key_env', 'server.segment.checkpoint',
           'server.training_archive.path', 'client.server_url', 'client.camera.stream_url'):
        assert key not in by_key
    assert by_key['client.camera.index']['editable'] is False
    assert all(not row['editable'] for row in rows if row['key'].startswith('client.'))
    assert all(by_key[key]['editable'] is False for key in IMMUTABLE)
    assert by_key['server.face.threshold']['requires_restart'] is False
    assert by_key['server.tts.speaker']['requires_restart'] is True


@pytest.mark.parametrize('key', sorted(IMMUTABLE) + [
    'server.llm.api_key', 'server.llm.api_key_env', 'server.llm.base_url',
           'server.training_archive.path', 'client.camera.index', 'client.server_url', 'server.missing',
])
def test_protected_or_unrecognized_settings_are_never_editable(key):
    cfg = Config()
    before = cfg.model_dump()
    with pytest.raises(ValueError):
        validated_value(cfg, key, 'changed')
    assert cfg.model_dump() == before


@pytest.mark.parametrize('key,value,expected', [
    ('server.permissions_enabled', 'нет', False),
    ('server.face.greetings_enabled', 'ON', True),
    ('server.face.threshold', '0.61', .61),
    ('server.telegram.poll_timeout_s', '30', 30),
    ('server.stt.allowed_languages', 'ru, en', ['ru', 'en']),
    ('server.stt.hotwords', '["Rowan", "Антон"]', ['Rowan', 'Антон']),
    ('server.stt.language', 'null', None),
])
def test_typed_input_validation_does_not_mutate_live_config(key, value, expected):
    cfg = Config()
    before = cfg.model_dump()
    assert validated_value(cfg, key, value) == expected
    assert cfg.model_dump() == before


@pytest.mark.parametrize('key,value', [
    ('server.face.threshold', 1.01), ('server.face.threshold', 'nan'),
    ('server.llm.monthly_budget_usd', -1), ('server.llm.monthly_budget_usd', 'inf'),
    ('server.telegram.poll_timeout_s', '0'), ('server.telegram.poll_timeout_s', 'twenty'),
    ('server.permissions_enabled', 'perhaps'), ('server.stt.hotwords', '[not json'),
    ('server.image_generation.model', 'unapproved-model'),
])
def test_invalid_values_are_rejected_by_actual_config_constraints(key, value):
    with pytest.raises((ValueError, TypeError)):
        validated_value(Config(), key, value)


def test_the_monthly_budget_has_no_upper_ceiling_and_zero_means_no_limit():
    """DECISIONS.md API-01: the owner removed the $20 ceiling on 2026-09-22."""
    cfg = Config()
    assert validated_value(cfg, 'server.llm.monthly_budget_usd', 900) == 900
    assert validated_value(cfg, 'server.llm.monthly_budget_usd', 0) == 0


def test_restore_applies_live_and_pending_values_but_ignores_stale_and_protected_rows(tmp_path):
    state = TelegramAdminState(tmp_path / 'admin.sqlite3', OWNER)
    for key, value in [('server.face.threshold', .62), ('server.tts.speaker', 'en_1'),
                       ('server.telegram.control_user_id', 17), ('server.telegram.chat_id', -999),
                       ('client.camera.index', 9), ('server.llm.monthly_budget_usd', -5)]:
        state.set_setting('config:' + key, {'value': value})
    state.set_setting('config:server.stt.language', ['malformed override'])
    cfg = Config()
    cfg.server.telegram.control_user_id, cfg.server.telegram.chat_id = OWNER, -100
    budget = cfg.server.llm.monthly_budget_usd
    restore_overrides(cfg, TelegramAdminState(state.path, OWNER))
    assert cfg.server.face.threshold == .62 and cfg.server.tts.speaker == 'en_1'
    assert cfg.server.telegram.control_user_id == OWNER and cfg.server.telegram.chat_id == -100
    assert cfg.client.camera.index == 0 and cfg.server.llm.monthly_budget_usd == budget
    assert cfg.server.stt.language is None


def test_live_updates_reach_copied_engine_fields_and_both_budget_providers(tmp_path):
    cfg = Config()
    from hub.api_budget import ApiBudget
    budget = ApiBudget(tmp_path / 'usage.sqlite3', 1.0, model='gpt-5.6-luna')
    runtime = dict(voices=SimpleNamespace(threshold=.4), face=SimpleNamespace(threshold=.45),
        stt=SimpleNamespace(default_language=None, allowed_languages=[], hotwords=''),
        llm=SimpleNamespace(_responses=SimpleNamespace(budget=budget)),
        image_generator=SimpleNamespace(budget=budget))
    updates = [('server.speaker.threshold', .63), ('server.face.threshold', .64),
               ('server.stt.language', 'ru'), ('server.stt.allowed_languages', ['ru', 'en']),
               ('server.stt.hotwords', ['Rowan', 'Антон']), ('server.llm.monthly_budget_usd', 12.75)]
    for key, value in updates:
        apply_live(cfg, key, validated_value(cfg, key, value), runtime)
    assert cfg.server.speaker.threshold == runtime['voices'].threshold == .63
    assert cfg.server.face.threshold == runtime['face'].threshold == .64
    assert runtime['stt'].default_language == 'ru'
    assert runtime['stt'].allowed_languages == ['ru', 'en']
    assert runtime['stt'].hotwords == 'Rowan, Антон'
    assert runtime['llm']._responses.budget.limit == 12_750_000
    assert runtime['image_generator'].budget.limit == 12_750_000


def test_live_updates_can_remove_the_monthly_ceiling(tmp_path):
    """DECISIONS.md API-01: 0 in the owner panel means "no ceiling"."""
    cfg = Config()
    from hub.api_budget import ApiBudget
    budget = ApiBudget(tmp_path / 'usage.sqlite3', 12.0, model='gpt-5.6-luna')
    runtime = dict(llm=SimpleNamespace(_responses=SimpleNamespace(budget=budget)),
                   image_generator=SimpleNamespace(budget=budget))
    apply_live(cfg, 'server.llm.monthly_budget_usd',
               validated_value(cfg, 'server.llm.monthly_budget_usd', 0), runtime)
    assert cfg.server.llm.monthly_budget_usd == 0
    assert runtime['llm']._responses.budget.limit is None
    assert budget.status()['limit_usd'] is None
