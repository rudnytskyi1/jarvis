"""Persistent, conservative API accounting. Uncertain requests stay charged.

Amounts are integer microdollars. BEGIN IMMEDIATE serializes reservations across
threads/processes; a crash never silently refunds a possibly billed request.
The ledger covers this application only, not other users of the API project.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

from common.image_models import IMAGE_MODEL_RATES
from common.openai_models import OPENAI_CACHE_RATES, OPENAI_CACHED_INPUT_RATES, OPENAI_TEXT_RATES


class CloudUnavailable(RuntimeError):
    """A cloud request cannot safely be made or completed."""


class BudgetExceeded(CloudUnavailable):
    pass


def microdollars(input_tokens: int, output_tokens: int, model: str = 'gpt-5.4-mini',
                 *, input_tokens_details: dict | None = None,
                 image_output_tokens: int | None = None) -> int:
    if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
        raise ValueError('invalid token usage')
    if model in IMAGE_MODEL_RATES:
        # Without modality usage (including reservations), charge every output
        # token at the highest rate. Thinking tokens count as text when known.
        images = output_tokens if image_output_tokens is None else image_output_tokens
        if type(images) is not int or not 0 <= images <= output_tokens:
            raise ValueError('invalid image token usage')
        incoming, text, image = map(Decimal, IMAGE_MODEL_RATES[model])
        return int((input_tokens * incoming + (output_tokens - images) * text
                    + images * image).to_integral_value(rounding=ROUND_CEILING))
    input_rate, output_rate = OPENAI_TEXT_RATES[model]
    input_cost = Decimal(input_tokens) * Decimal(input_rate)
    if model in OPENAI_CACHED_INPUT_RATES:
        details = {} if input_tokens_details is None else input_tokens_details
        if not isinstance(details, dict):
            raise ValueError('invalid input token details')
        cached = details.get('cached_tokens', 0)
        if type(cached) is not int or not 0 <= cached <= input_tokens:
            raise ValueError('invalid cached token usage')
        write_pricing = OPENAI_CACHE_RATES.get(model)
        writes = details.get('cache_write_tokens', input_tokens - cached) if write_pricing else 0
        if type(writes) is not int or not 0 <= writes <= input_tokens - cached:
            raise ValueError('invalid cache write usage')
        cached_rate = OPENAI_CACHED_INPUT_RATES[model]
        write_rate = write_pricing[1] if write_pricing else input_rate
        input_cost = (Decimal(input_tokens - cached - writes) * Decimal(input_rate)
                      + Decimal(cached) * Decimal(cached_rate) + Decimal(writes) * Decimal(write_rate))
    return int((input_cost
                + Decimal(output_tokens) * Decimal(output_rate)).to_integral_value(rounding=ROUND_CEILING))


class ApiBudget:
    def __init__(self, path: Path, monthly_usd: float = 18.0, *, model: str = 'gpt-5.4-mini'):
        if not 0 < monthly_usd <= 20:
            raise ValueError("API monthly budget must be greater than zero and at most $20")
        if model not in OPENAI_TEXT_RATES and model not in IMAGE_MODEL_RATES:
            raise ValueError('Model pricing has not been reviewed')
        self.model = model
        self.path = Path(path)
        self.limit = int(Decimal(str(monthly_usd)) * 1_000_000)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, month TEXT NOT NULL, amount INTEGER NOT NULL,
                settled INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER, output_tokens INTEGER)""")
            # Existing rows were all mini. Preserve their recorded amounts and
            # assign the old rate to pending reservations across an upgrade.
            if 'model' not in {row[1] for row in db.execute('PRAGMA table_info(requests)')}:
                db.execute("ALTER TABLE requests ADD COLUMN model TEXT NOT NULL DEFAULT 'gpt-5.4-mini'")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def month() -> str:
        return datetime.now(UTC).strftime("%Y-%m")

    def reserve(self, input_tokens: int, max_output_tokens: int) -> str:
        amount = microdollars(input_tokens, max_output_tokens, self.model)
        if input_tokens < 0 or max_output_tokens < 0:
            raise ValueError("negative token estimate")
        request_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            month = self.month()
            used = db.execute("SELECT COALESCE(SUM(amount), 0) FROM requests WHERE month=?", (month,)).fetchone()[0]
            if used + amount > self.limit:
                raise BudgetExceeded("Monthly API allowance reached; local commands remain available.")
            db.execute("INSERT INTO requests(id, month, amount, model) VALUES (?, ?, ?, ?)",
                       (request_id, month, amount, self.model))
        return request_id

    def settle(self, request_id: str, input_tokens: int, output_tokens: int,
               *, input_tokens_details: dict | None = None,
               image_output_tokens: int | None = None) -> None:
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            raise ValueError("invalid usage; reservation must be retained")
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT model FROM requests WHERE id=? AND settled=0', (request_id,)).fetchone()
            if row is None:
                return
            db.execute("UPDATE requests SET amount=?, settled=1, input_tokens=?, output_tokens=? WHERE id=? AND settled=0",
                       (microdollars(input_tokens, output_tokens, row[0], input_tokens_details=input_tokens_details,
                                     image_output_tokens=image_output_tokens),
                        input_tokens, output_tokens, request_id))

    def status(self) -> dict:
        month = self.month()
        with self._connect() as db:
            rows = db.execute(
                'SELECT model, settled, COUNT(*), SUM(amount) FROM requests WHERE month=? GROUP BY model, settled',
                (month,)).fetchall()
        providers = {}
        for model, settled, count, amount in rows:
            provider = 'OpenAI' if model in OPENAI_TEXT_RATES else 'Google Gemini' if model in IMAGE_MODEL_RATES else 'Other'
            entry = providers.setdefault(provider, dict(provider=provider, settled_amount=0, reserved_amount=0,
                                                       settled_requests=0, unsettled_requests=0))
            entry['settled_amount' if settled else 'reserved_amount'] += amount
            entry['settled_requests' if settled else 'unsettled_requests'] += count
        settled = sum(row['settled_amount'] for row in providers.values())
        reserved = sum(row['reserved_amount'] for row in providers.values())
        pending = sum(row['unsettled_requests'] for row in providers.values())
        breakdown = [dict(provider=row['provider'], settled_estimate_usd=row['settled_amount'] / 1_000_000,
                          reserved_usd=row['reserved_amount'] / 1_000_000,
                          settled_requests=row['settled_requests'], unsettled_requests=row['unsettled_requests'])
                     for row in sorted(providers.values(), key=lambda row: row['provider'])]
        # This is local token-based accounting, never a provider billing balance.
        # Preserve the conservative total used by the spending guard.
        return {'month': month, 'accounted_usd': (settled + reserved) / 1_000_000,
                'settled_estimate_usd': settled / 1_000_000, 'reserved_usd': reserved / 1_000_000,
                'limit_usd': self.limit / 1_000_000, 'unsettled_requests': pending,
                'providers': breakdown, 'billing_synced': False}
