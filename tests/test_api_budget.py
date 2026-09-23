import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from hub.api_budget import ApiBudget, BudgetExceeded


def test_pending_reservation_survives_restart(tmp_path):
    path = tmp_path / "ledger.db"
    budget = ApiBudget(path, 0.01)
    budget.reserve(10_000, 500)  # $0.00975, uncertain after a timeout/crash
    reopened = ApiBudget(path, 0.01)
    with pytest.raises(BudgetExceeded):
        reopened.reserve(1_000, 0)
    assert reopened.status()["unsettled_requests"] == 1


def test_settlement_refunds_estimate_once_and_counts_output(tmp_path):
    budget = ApiBudget(tmp_path / "ledger.db")
    key = budget.reserve(10_000, 600)
    budget.settle(key, 3000, 300)
    assert budget.status()["accounted_usd"] == 0.0036
    budget.settle(key, 0, 0)  # retries cannot refund the same request twice
    assert budget.status()["accounted_usd"] == 0.0036


def test_concurrent_process_style_clients_cannot_overspend(tmp_path):
    path = tmp_path / "ledger.db"
    ApiBudget(path, 0.009)
    def reserve(_):
        try:
            ApiBudget(path, 0.009).reserve(0, 1000)  # $0.0045 each
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(16))) == 2
    assert ApiBudget(path, 0.009).status()["accounted_usd"] == 0.009


def test_month_rollover_does_not_refund_previous_month(tmp_path, monkeypatch):
    budget = ApiBudget(tmp_path / "ledger.db", 0.0045)
    monkeypatch.setattr(budget, "month", lambda: "2026-09")
    key = budget.reserve(0, 1000)
    monkeypatch.setattr(budget, "month", lambda: "2026-10")
    assert budget.status()["accounted_usd"] == 0
    budget.reserve(0, 1000)
    budget.settle(key, 0, 0)
    assert budget.status()["accounted_usd"] == 0.0045


def test_invalid_usage_preserves_reservation(tmp_path):
    budget = ApiBudget(tmp_path / "ledger.db")
    key = budget.reserve(100, 100)
    with pytest.raises(ValueError):
        budget.settle(key, -1, 1)
    assert budget.status()["unsettled_requests"] == 1


@pytest.mark.parametrize("limit", [-1, -0.01])
def test_a_negative_ceiling_is_refused(limit, tmp_path):
    with pytest.raises(ValueError):
        ApiBudget(tmp_path / "ledger.db", limit)


def test_zero_means_no_ceiling_and_a_big_number_is_allowed(tmp_path):
    """DECISIONS.md API-01: the owner removed the $20 ceiling on 2026-09-22."""
    unlimited = ApiBudget(tmp_path / "unlimited.db", 0)
    assert unlimited.limit is None
    assert unlimited.status()["limit_usd"] is None
    # A month's worth of spending is never refused without a ceiling.
    for _ in range(8):
        unlimited.reserve(5_000_000, 2048)
    assert unlimited.status()["accounted_usd"] > 20
    assert ApiBudget(tmp_path / "big.db", 900).limit == 900_000_000
def test_full_model_reservation_and_settlement_use_full_price(tmp_path):
    budget = ApiBudget(tmp_path / 'ledger.db', .04, model='gpt-5.4')
    key = budget.reserve(10_000, 600)
    assert budget.status()['accounted_usd'] == .034
    with pytest.raises(BudgetExceeded):
        budget.reserve(0, 1000)
    budget.settle(key, 3000, 300)
    assert budget.status()['accounted_usd'] == .012


def test_model_switch_preserves_combined_spend_and_original_reservation_rate(tmp_path):
    path = tmp_path / 'ledger.db'
    mini = ApiBudget(path)
    old_key = mini.reserve(10_000, 600)
    full = ApiBudget(path, model='gpt-5.4')
    new_key = full.reserve(10_000, 600)
    assert full.status()['accounted_usd'] == .0442
    # Even a different client's reconciliation uses the original request model.
    full.settle(old_key, 3000, 300)
    mini.settle(new_key, 3000, 300)
    assert full.status()['accounted_usd'] == .0156


def test_old_ledger_migration_keeps_settled_spend_and_pending_mini_requests(tmp_path):
    path = tmp_path / 'ledger.db'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE requests (id TEXT PRIMARY KEY, month TEXT NOT NULL, amount INTEGER NOT NULL, '
                   'settled INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER, output_tokens INTEGER)')
        db.executemany('INSERT INTO requests (id,month,amount,settled) VALUES (?,?,?,?)',
                       [('old', ApiBudget.month(), 3600, 1), ('pending', ApiBudget.month(), 10200, 0)])
    full = ApiBudget(path, model='gpt-5.4')
    assert full.status()['accounted_usd'] == .0138
    full.settle('pending', 3000, 300)
    assert full.status()['accounted_usd'] == .0072
    assert full.status()['unsettled_requests'] == 0


def test_concurrent_mixed_models_share_the_same_monthly_limit(tmp_path):
    path = tmp_path / 'ledger.db'
    ApiBudget(path, .03)
    def reserve(index):
        model = 'gpt-5.4' if index % 2 else 'gpt-5.4-mini'
        try:
            ApiBudget(path, .03, model=model).reserve(0, 1000)
            return .015 if index % 2 else .0045
        except BudgetExceeded:
            return 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        spent = sum(pool.map(reserve, range(16)))
    assert spent <= .03
    assert ApiBudget(path, .03).status()['accounted_usd'] == pytest.approx(spent)


def test_luna_reserves_cache_write_ceiling_and_settles_actual_usage_after_switch(tmp_path):
    path = tmp_path / 'ledger.db'
    luna = ApiBudget(path, .004, model='gpt-5.6-luna')
    key = luna.reserve(10_000, 600)
    assert luna.status()['accounted_usd'] == .00322
    with pytest.raises(BudgetExceeded):
        luna.reserve(4000, 0)
    other_model = ApiBudget(path, .004, model='gpt-5.4')
    other_model.settle(key, 3000, 300, input_tokens_details={'cached_tokens': 1500, 'cache_write_tokens': 1000})
    assert luna.status()['accounted_usd'] == .00074


@pytest.mark.parametrize('details', [{'cached_tokens': -1}, {'cached_tokens': True},
                                     {'cache_write_tokens': '100'}, {'cached_tokens': 2000, 'cache_write_tokens': 2000}])
def test_invalid_luna_cache_usage_keeps_reservation(tmp_path, details):
    luna = ApiBudget(tmp_path / 'ledger.db', model='gpt-5.6-luna')
    key = luna.reserve(3000, 600)
    before = luna.status()
    with pytest.raises(ValueError):
        luna.settle(key, 3000, 300, input_tokens_details=details)
    assert luna.status() == before


@pytest.mark.parametrize('model,cost', [('gpt-5.4-mini', .002588), ('gpt-5.4', .008625)])
def test_legacy_text_models_apply_only_reported_cache_discount(tmp_path, model, cost):
    budget = ApiBudget(tmp_path / 'ledger.db', model=model)
    key = budget.reserve(3000, 300)
    budget.settle(key, 3000, 300, input_tokens_details={'cached_tokens': 1500})
    assert budget.status()['settled_estimate_usd'] == cost
    assert budget.status()['reserved_usd'] == 0


def test_status_separates_providers_and_reserves_without_changing_budget_guard(tmp_path):
    path = tmp_path / 'ledger.db'
    text = ApiBudget(path, monthly_usd=.1)
    key = text.reserve(3000, 300)
    text.settle(key, 3000, 300)
    image = ApiBudget(path, monthly_usd=.1, model='gemini-3.1-flash-image')
    key = image.reserve(1000, 100)
    image.settle(key, 1000, 100, image_output_tokens=100)
    text.reserve(0, 1000)
    image.reserve(0, 1000)
    status = text.status()
    by_provider = {row['provider']: row for row in status['providers']}
    assert by_provider['OpenAI']['settled_estimate_usd'] == .0036
    assert by_provider['OpenAI']['reserved_usd'] == .0045
    assert by_provider['Google Gemini']['settled_estimate_usd'] == .0065
    assert by_provider['Google Gemini']['reserved_usd'] == .06
    assert status['settled_estimate_usd'] == .0101
    assert status['reserved_usd'] == .0645 and status['accounted_usd'] == .0746
    assert status['unsettled_requests'] == 2 and status['billing_synced'] is False
    with pytest.raises(BudgetExceeded):
        image.reserve(0, 500)
    assert ApiBudget(path, monthly_usd=.1).status() == status
