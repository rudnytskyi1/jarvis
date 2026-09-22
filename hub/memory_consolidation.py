"""Ночная консолидация памяти (ТЗ F-416).

В 04:00 по часовому поясу ДОМА день превращается в 5–10 фактов: дубли
сливаются, старые факты теряют вес, истёкшие удаляются, а недостающие
векторы догоняются (DECISIONS P3-16, пункт 8: строки, написанные до
появления модели эмбеддингов, доходят до векторной половины поиска именно
здесь).

Правило раздела 1 ТЗ «никаких фейков» здесь означает конкретную вещь:
сводку дня пишет только суммаризатор (локальная модель через structured
output). Нет модели, нет ответа модели или сводка вышла за границы 5–10
фактов — сводка не пишется вовсе, в отчёте остаётся ``summarizer="none"``
и причина в логе. Механическая часть (слияние дублей, вес, TTL, векторы)
идёт всегда: она ничего не выдумывает.

Приватность (F-415): сводка — факт ДОМА (``scope=home``,
``owner_id=home_id``), поэтому гость её не видит
(``hub.memory_search.visible``), а имена людей остаются внутри текста.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hub import memories, memory_search, vectors
from hub.memories import Kind, MemoryFact, MemoryIndex, Scope

log = logging.getLogger("jarvis.server.memory_consolidation")

#: ТЗ F-416: «день в 5–10 фактов».
MIN_DIGEST_FACTS = 5
MAX_DIGEST_FACTS = 10

#: За сколько часов до прохода начинается «день».
DEFAULT_WINDOW_HOURS = 24.0

#: Ключ в ``homes.settings_json``, которым дом отмечает сделанный проход. Он
#: и есть защита от двойной консолидации: хаб, перезапущенный в 05:00, не
#: сведёт день второй раз, а хаб, простоявший всю ночь, — сведёт.
MARKER_KEY = "memory_consolidated_on"


class DigestUnavailable(RuntimeError):
    """Сводка дня не получена: суммаризатор молчит или нарушил контракт."""


class DigestFact(BaseModel):
    """Один факт сводки, как его отдаёт суммаризатор."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=memories.MAX_TEXT_CHARS)
    kind: Kind = Kind.EVENT
    weight: float = Field(default=memories.DEFAULT_WEIGHT, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _clean(self) -> DigestFact:
        text = " ".join(self.text.split())
        if not text:
            raise ValueError("a digest fact cannot be empty")
        self.text = text[:memories.MAX_TEXT_CHARS].rstrip()
        return self


class ConsolidationReport(BaseModel):
    """Что именно сделал один проход по одному дому."""

    model_config = ConfigDict(extra="forbid")

    home_id: str
    ran_at: datetime
    tz: str
    #: Истёкшие факты, удалённые по TTL.
    purged: int = Field(default=0, ge=0)
    #: Факты, слитые в более новые (сколько строк исчезло).
    merged: int = Field(default=0, ge=0)
    #: Факты, которым понизили вес за возраст.
    decayed: int = Field(default=0, ge=0)
    #: Недостающие векторы, посчитанные за этот проход.
    embedded: int = Field(default=0, ge=0)
    #: Факты сводки, записанные в таблицу.
    digest: int = Field(default=0, ge=0)
    #: Исходные факты дня, удалённые после того, как вошли в сводку.
    folded: int = Field(default=0, ge=0)
    #: "llm" или "none" — кто написал сводку (никаких «как будто»).
    summarizer: str = "none"
    #: Почему сводки нет, когда её нет.
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Отчёт в том виде, в каком его пишет планировщик и аудит."""
        return self.model_dump(mode="json")


def normalize_text(text: Any) -> str:
    """Нормальная форма факта: то, что должно совпасть у двух его копий."""
    lowered = str(text or "").casefold()
    kept = "".join(char if (char.isalnum() or char.isspace()) else " " for char in lowered)
    return " ".join(kept.split())


def _run_sync(step: Any) -> Any:
    """Синхронный вход в шаг прохода: свой цикл, если его ещё нет.

    Проход живёт на цикле хаба (соединение с БД принадлежит ему, DECISIONS
    P1-44), поэтому ``async``-версия — основная, а синхронная существует для
    тестов и для вызова из потока планировщика.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(step)
    raise RuntimeError(
        "consolidate_home() cannot run inside a running loop; use consolidate_home_async()"
    )


def _tz(name: Any) -> ZoneInfo:
    """Часовой пояс дома; незнакомый — UTC, а не исключение по пути."""
    try:
        return ZoneInfo(str(name or "UTC"))
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("Unknown home time zone %r; consolidating on UTC", name)
        return ZoneInfo("UTC")


class MemoryConsolidator:
    """Проход F-416 по памяти дома.

    :param conn: соединение с ``data/hub.db`` того же потока, что и цикл
        (DECISIONS P1-44).
    :param config: ``cfg.server.memory`` — расписание, границы сводки, веса.
    :param summarizer: ``(texts, minimum, maximum) -> [DigestFact]``; может
        быть coroutine, проход дождётся результата.
    :param embedder: ``hub.embeddings.TextEmbedder`` для догона векторов.
    """

    def __init__(self, conn: sqlite3.Connection, *, config: Any = None,
                 summarizer: Any = None, embedder: Any = None, audit: Any = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._conn = conn
        self._index = MemoryIndex(conn)
        self.config = config
        self.summarizer = summarizer
        self.embedder = embedder
        self.audit = audit
        self.clock = clock

    # -- расписание ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._setting("consolidation_enabled", True))

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    def target(self) -> tuple[int, int]:
        """Час и минута прохода по ТЗ (по умолчанию 04:00)."""
        return (int(self._setting("consolidation_hour", 4)),
                int(self._setting("consolidation_minute", 0)))

    def _local_now(self, tz_name: Any, now: datetime | None) -> datetime:
        moment = now or self.clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(_tz(tz_name))

    def last_pass_day(self, home_id: str) -> str:
        """Дата последнего прохода дома (``''``, если его ещё не было)."""
        row = self._conn.execute("SELECT settings_json FROM homes WHERE home_id=?",
                                 (str(home_id),)).fetchone()
        return _settings(row[0] if row else "").get(MARKER_KEY, "") if row else ""

    def _mark(self, home_id: str, day: str) -> None:
        """Отметить проход за эту локальную дату дома."""
        row = self._conn.execute("SELECT settings_json FROM homes WHERE home_id=?",
                                 (str(home_id),)).fetchone()
        if row is None:
            return  # дома нет в таблице: консолидировать нечего, но и падать не за что
        settings = _settings(row[0])
        settings[MARKER_KEY] = str(day)
        self._conn.execute("UPDATE homes SET settings_json=? WHERE home_id=?",
                           (json.dumps(settings, ensure_ascii=False), str(home_id)))
        self._conn.commit()

    def is_due(self, home_id: str, tz_name: Any, *, now: datetime | None = None) -> bool:
        """Пора ли консолидировать этот дом прямо сейчас.

        Дом «созрел», когда его собственные часы прошли время прохода, а
        отметки за этот локальный день ещё нет: хаб, поднявшийся днём,
        наверстывает ночь, а хаб, перезапущенный сразу после прохода, не
        сводит день дважды.
        """
        if not self.enabled:
            return False
        hour, minute = self.target()
        local = self._local_now(tz_name, now)
        day = local.date().isoformat()
        if self.last_pass_day(home_id) == day:
            return False
        return (local.hour, local.minute) >= (hour, minute)

    def due_homes(self, homes: Iterable[Any], *, now: datetime | None = None) -> list[Any]:
        """Дома, которым пора, в порядке конфига."""
        return [home for home in homes or []
                if self.is_due(getattr(home, "home_id", ""), getattr(home, "tz", "UTC"),
                               now=now)]

    # -- проход ----------------------------------------------------------

    def consolidate_home(self, home_id: str, *, tz_name: Any = "UTC",
                         now: datetime | None = None) -> ConsolidationReport:
        """Один проход по дому, синхронно (тесты и вызов из потока)."""
        return _run_sync(self.consolidate_home_async(home_id, tz_name=tz_name, now=now))

    async def consolidate_home_async(self, home_id: str, *, tz_name: Any = "UTC",
                                     now: datetime | None = None) -> ConsolidationReport:
        """Один проход по дому: TTL → дубли → вес → векторы → сводка дня.

        Механика обращается к SQLite синхронно (это микросекунды, как и у
        остальных задач планировщика), а модельные части асинхронны: сводка
        ждёт ответа модели, векторы считаются в рабочем потоке.
        """
        moment = now or self.clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        report = ConsolidationReport(home_id=str(home_id), ran_at=moment,
                                     tz=str(tz_name or "UTC"))

        report.purged = self._purge_expired(moment)
        report.merged = self._merge_duplicates(moment)
        report.decayed = self._decay(moment)
        report.embedded = await self._catch_up_embeddings(moment)
        digest, note = await self._digest(home_id, moment)
        report.digest = len([self._index.write(fact) for fact in digest])
        report.summarizer = "llm" if digest else "none"
        report.note = note
        if digest and bool(self._setting("fold_sources", True)):
            report.folded = self._fold_sources(home_id, moment)

        self._mark(home_id, self._local_now(tz_name, moment).date().isoformat())
        self._audit(report)
        log.info(
            "Memory consolidation %s: %d purged, %d merged, %d decayed, %d embedded, "
            "%d digest fact(s) from %s%s",
            home_id, report.purged, report.merged, report.decayed, report.embedded,
            report.digest, report.summarizer, f" ({note})" if note else "",
        )
        return report

    async def run_due(self, homes: Iterable[Any], *,
                      now: datetime | None = None) -> list[ConsolidationReport]:
        """Все дома, которым пора, — по очереди, чтобы не занять цикл надолго."""
        reports: list[ConsolidationReport] = []
        for home in self.due_homes(homes, now=now):
            try:
                reports.append(await self.consolidate_home_async(
                    getattr(home, "home_id", ""), tz_name=getattr(home, "tz", "UTC"), now=now))
            except Exception as exc:  # noqa: BLE001 - один дом не отменяет остальные
                log.warning("Consolidation of %s failed (%s)",
                            getattr(home, "home_id", "?"), exc)
        return reports

    # -- механика --------------------------------------------------------

    def _purge_expired(self, now: datetime) -> int:
        """ТЗ F-416: «удаление истёкших»."""
        try:
            return self._index.purge_expired(now=now)
        except sqlite3.Error as exc:
            log.warning("Expired memories were not purged (%s)", exc)
            return 0

    def _live(self, *, now: datetime) -> list[MemoryFact]:
        try:
            return self._index.active(now=now)
        except sqlite3.Error as exc:
            log.warning("Memories could not be read (%s)", exc)
            return []

    def _decay(self, now: datetime) -> int:
        """ТЗ F-416: «понижение веса старых».

        Факт теряет ``decay_per_day`` за каждые полные сутки возраста, но
        никогда не опускается ниже ``decay_floor``: живой факт с нулевым весом
        выпал бы из поиска молча, а это уже потеря, а не забвение.
        """
        per_day = float(self._setting("decay_per_day", 0.9))
        floor = float(self._setting("decay_floor", 0.05))
        if per_day >= 1.0:
            return 0
        changed = 0
        for fact in self._live(now=now):
            age_days = (now - fact.created_at).total_seconds() / 86400.0
            if age_days < 1.0:
                continue
            wanted = max(floor, min(1.0, per_day ** math.floor(age_days)))
            if wanted >= fact.weight - 1e-9:
                continue
            fact.weight = round(wanted, 4)
            try:
                self._index.write(fact)
            except sqlite3.Error as exc:
                log.warning("Weight of %s stayed as it was (%s)", fact.memory_id, exc)
                continue
            changed += 1
        return changed

    def _merge_duplicates(self, now: datetime) -> int:
        """ТЗ F-416: «слияние дублей».

        Одинаковый нормализованный текст у одного владельца, вида и области —
        это один факт, сколько бы раз его ни сказали. Совпавшие по смыслу
        (косинус выше ``duplicate_similarity``) считаются тем же фактом только
        тогда, когда у обоих есть вектор.
        """
        threshold = float(self._setting("duplicate_similarity", 0.93))
        groups: dict[tuple[str, str, str], list[MemoryFact]] = {}
        for fact in self._live(now=now):
            key = (str(fact.scope), fact.owner_id.casefold(), str(fact.kind))
            groups.setdefault(key, []).append(fact)

        removed = 0
        for facts in groups.values():
            if len(facts) < 2:
                continue
            newest_first = sorted(facts, key=lambda item: (item.created_at, item.memory_id),
                                  reverse=True)
            kept: list[MemoryFact] = []
            for fact in newest_first:
                twin = self._twin(fact, kept, threshold)
                if twin is None:
                    kept.append(fact)
                    continue
                # Дубль уходит, но его вес и его более полный текст остаются:
                # сказанное дважды весит больше, чем сказанное один раз.
                self._absorb(twin, fact)
                try:
                    if self._index.delete(fact.memory_id):
                        removed += 1
                except sqlite3.Error as exc:
                    log.warning("Duplicate %s stayed (%s)", fact.memory_id, exc)
        return removed

    def _absorb(self, kept: MemoryFact, gone: MemoryFact) -> None:
        """Перенести вес и текст дубля в оставленный факт."""
        changed = False
        if kept.weight < gone.weight:
            kept.weight = gone.weight
            changed = True
        if len(gone.text) > len(kept.text):
            kept.text = gone.text
            changed = True
        if not changed:
            return
        try:
            self._index.write(kept)
        except sqlite3.Error as exc:
            log.warning("Merged fact %s was not saved (%s)", kept.memory_id, exc)

    @staticmethod
    def _twin(fact: MemoryFact, kept: Sequence[MemoryFact],
              threshold: float) -> MemoryFact | None:
        """Тот же факт среди уже отобранных, или ``None``."""
        wanted = normalize_text(fact.text)
        for candidate in kept:
            if normalize_text(candidate.text) == wanted:
                return candidate
            if fact.vector is None or candidate.vector is None or fact.dim != candidate.dim:
                continue
            left = vectors.unpack_vector(fact.vector)
            right = vectors.unpack_vector(candidate.vector)
            if memory_search.cosine(left, right) >= threshold:
                return candidate
        return None

    async def _catch_up_embeddings(self, now: datetime) -> int:
        """Догон векторов фактов, записанных без модели (DECISIONS P3-16 #8)."""
        embedder = self.embedder
        batch = int(self._setting("embed_batch", 256))
        if embedder is None or batch <= 0:
            return 0
        pending = [fact for fact in self._live(now=now) if fact.vector is None][:batch]
        if not pending:
            return 0
        try:
            # The embedder adds the model's own ``passage:`` prefix itself, the
            # same way ``remember`` embeds a fact: one vector space for both.
            # It is a CPU model by design (ТЗ 9.4), so it never runs on the loop.
            computed = await asyncio.to_thread(embedder.encode, [fact.text for fact in pending])
        except Exception as exc:  # noqa: BLE001 - модель может отказать, факт остаётся
            log.info("Memory embeddings are still behind (%s)", exc)
            return 0
        written = 0
        for fact, embedding in zip(pending, computed, strict=False):
            try:
                fact.vector = vectors.pack_vector(embedding)
                fact.dim = len(fact.vector) // 4
                self._index.write(fact)
                written += 1
            except Exception as exc:  # noqa: BLE001 - один факт не отменяет проход
                log.warning("Embedding of %s was not saved (%s)", fact.memory_id, exc)
        return written

    async def _digest(self, home_id: str, now: datetime) -> tuple[list[MemoryFact], str]:
        """Сводка дня: 5–10 фактов дома, или ``([], причина)``."""
        if self.summarizer is None:
            return [], "no summarizer is configured"
        window = float(self._setting("consolidation_window_hours", DEFAULT_WINDOW_HOURS))
        since = now - timedelta(hours=window)
        material = self._day_material(home_id, since, now)
        if not material:
            return [], "the day holds nothing to sum up"
        minimum = int(self._setting("consolidation_min_facts", MIN_DIGEST_FACTS))
        maximum = int(self._setting("consolidation_max_facts", MAX_DIGEST_FACTS))
        try:
            produced = self.summarizer(material, minimum, maximum)
            if inspect.isawaitable(produced):
                produced = await produced
        except Exception as exc:  # noqa: BLE001 - сводка не обязана получиться
            return [], f"the summarizer failed ({type(exc).__name__})"
        facts = self._validated(produced, home_id, minimum, maximum)
        if facts is None:
            return [], "the summarizer did not answer with 5-10 valid facts"
        return facts, ""

    def _day_material(self, home_id: str, since: datetime, now: datetime) -> list[str]:
        """Из чего складывается день: факты дома и хаба плюс реплики дня."""
        texts: list[str] = []
        wanted = str(home_id).casefold()
        for fact in self._live(now=now):
            if not since <= fact.created_at <= now:
                continue
            if fact.scope is Scope.HOME and fact.owner_id.casefold() == wanted:
                texts.append(fact.text)
            elif fact.scope is Scope.HUB:
                texts.append(fact.text)
        texts.extend(self._day_dialogs(home_id, since, now))
        return texts[:200]

    def _day_dialogs(self, home_id: str, since: datetime, now: datetime) -> list[str]:
        """Реплики дня из ``dialog_turns`` — таблица наполняется всегда (P3-17)."""
        try:
            rows = self._conn.execute(
                "SELECT text FROM dialog_turns WHERE home_id=? AND ts>=? AND ts<=? "
                "ORDER BY ts",
                (str(home_id), since.timestamp(), now.timestamp()),
            ).fetchall()
        except sqlite3.Error as exc:
            log.debug("Dialog turns of %s are unavailable (%s)", home_id, exc)
            return []
        out: list[str] = []
        for row in rows:
            text = " ".join(str(row[0] or "").split())
            if text:
                out.append(text[:400])
        return out

    @staticmethod
    def _validated(produced: Any, home_id: str, minimum: int,
                   maximum: int) -> list[MemoryFact] | None:
        """Строгий контракт сводки: 5–10 фактов ДОМА, иначе ничего.

        Сводка всегда принадлежит дому: так её не увидит гость (F-415), а
        личные факты человека в неё не переезжают.
        """
        if produced is None:
            return None
        try:
            items = list(produced)
        except TypeError:
            return None
        if not minimum <= len(items) <= maximum:
            return None
        facts: list[MemoryFact] = []
        for item in items:
            try:
                entry = item if isinstance(item, DigestFact) else DigestFact.model_validate(item)
            except Exception as exc:  # noqa: BLE001 - негодный пункт отменяет сводку
                log.info("A digest item was refused (%s)", exc)
                return None
            facts.append(memories.fact_from(scope=Scope.HOME, owner_id=str(home_id),
                                            kind=entry.kind, text=entry.text,
                                            weight=entry.weight))
        return facts

    def _fold_sources(self, home_id: str, now: datetime) -> int:
        """Убрать исходные факты дня: их содержание уже в сводке."""
        window = float(self._setting("consolidation_window_hours", DEFAULT_WINDOW_HOURS))
        since = now - timedelta(hours=window)
        removed = 0
        for fact in self._live(now=now):
            if fact.scope is not Scope.HOME or fact.owner_id.casefold() != str(home_id).casefold():
                continue
            if not since <= fact.created_at <= now or fact.weight >= 1.0:
                # Факт с полным весом — это либо только что записанная сводка,
                # либо то, что человек сказал уже после неё.
                continue
            try:
                if self._index.delete(fact.memory_id):
                    removed += 1
            except sqlite3.Error as exc:
                log.warning("Source fact %s stayed (%s)", fact.memory_id, exc)
        return removed

    def _audit(self, report: ConsolidationReport) -> None:
        if self.audit is None:
            return
        try:
            self.audit.record(action="memory.consolidate", target=report.home_id, result="ok",
                              detail=report.as_dict())
        except Exception as exc:  # noqa: BLE001 - аудит не отменяет сделанного
            log.info("The consolidation report was not written to the audit (%s)", exc)


def _settings(raw: Any) -> dict[str, Any]:
    """``homes.settings_json`` как словарь; мусор читается как пустой."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def llm_summarizer(llm: Any, *, schema_name: str = "day_facts") -> Callable[..., Any]:
    """Суммаризатор дня на локальной модели (structured output, F-416).

    Возвращает coroutine, которую вызывает проход. Контракт жёсткий: модель
    обязана ответить объектом ``{"facts": [{"text": ..., "kind": ...}]}``;
    всё остальное — :class:`DigestUnavailable`, и сводка не пишется.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "kind": {"type": "string", "enum": [str(kind) for kind in Kind]},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["facts"],
        "additionalProperties": False,
    }

    async def summarize(texts: Sequence[str], minimum: int, maximum: int) -> list[DigestFact]:
        from hub.llm import StructuredUnavailable

        joined = "\n".join(f"- {text}" for text in texts)
        messages = [
            {"role": "system",
             "content": ("You compress one day of a shared student room into memory facts. "
                         f"Answer with {minimum} to {maximum} facts in the given JSON schema. "
                         "Write only what the day's material says: no guesses, no advice. "
                         "One fact is one short sentence in the language of the material. "
                         "Prefer what still matters tomorrow: preferences, plans, "
                         "appointments, changes in the room.")},
            {"role": "user", "content": f"Material of the day:\n{joined}"},
        ]
        try:
            answer = await llm.structured_json(messages, schema, name=schema_name)
        except StructuredUnavailable as exc:
            raise DigestUnavailable(str(exc)) from exc
        items = answer.get("facts") if isinstance(answer, Mapping) else None
        if not isinstance(items, list):
            raise DigestUnavailable("the model answered without a list of facts")
        return [DigestFact.model_validate(item) for item in items]

    return summarize


class MemoryConsolidationTask:
    """Задача планировщика хаба (ТЗ F-416, ``hub/scheduler.py``).

    Один интервал — одна проверка: у каких домов местные часы уже прошли время
    прохода и нет отметки за сегодня. Дом, который не созрел, не трогаем;
    созревшие проходятся по очереди, чтобы ночная работа не занимала цикл
    надолго. Отчёт попадает в аудит, ошибка — в ``on_error`` планировщика.
    """

    NAME = "memory.consolidate"

    def __init__(self, consolidator: MemoryConsolidator, *, homes: Iterable[Any] = (),
                 interval_s: float = 900.0) -> None:
        self.consolidator = consolidator
        self.homes = list(homes or [])
        self.interval_s = float(interval_s)
        self.name = self.NAME

    async def run(self) -> dict[str, Any]:
        """Одна проверка расписания; отчёт — по домам, которые прошли.

        Задача асинхронная: сводку дня пишет модель, и ждать её надо на цикле,
        а не в рабочем потоке — соединение с ``data/hub.db`` принадлежит циклу
        (DECISIONS P1-44).
        """
        reports = await self.consolidator.run_due(self.homes)
        if not reports:
            return {"homes": 0}
        return {
            "homes": len(reports),
            "purged": sum(report.purged for report in reports),
            "merged": sum(report.merged for report in reports),
            "decayed": sum(report.decayed for report in reports),
            "embedded": sum(report.embedded for report in reports),
            "digest": sum(report.digest for report in reports),
            "summaries": [report.summarizer for report in reports],
        }


__all__ = [
    "DEFAULT_WINDOW_HOURS",
    "MARKER_KEY",
    "MAX_DIGEST_FACTS",
    "MIN_DIGEST_FACTS",
    "ConsolidationReport",
    "DigestFact",
    "DigestUnavailable",
    "MemoryConsolidationTask",
    "MemoryConsolidator",
    "llm_summarizer",
    "normalize_text",
]
