"""Матрица массового аудита: тысячи проверок договора запросов без сети.

Владелец просил аудит «с несколько тысяч тестов разных сценариев». Живой
стенд (`scripts/live-eval.py` + собранный корпус) стоит денег и времени на
каждый прогон, поэтому рядом с ним идёт эта матрица: те же сценарии, но
проверяются вещи, которые решаются в коде, а не моделью —

* каждый инструмент, которого сценарий ждёт, вообще существует в договоре
  (`hub.tools.TOOL_NAMES`) — опечатка в ожидании ловится сразу;
* сужение набора инструментов по семейству (Jev, U-14) не может спрятать
  инструмент, который сценарию нужен;
* ``pc_control`` кладёт имя приложения туда, куда смотрит клиент;
* сценарии уникальны и не дублируют друг друга.

Корпус берётся из генератора (`scripts/gen-audit-scenarios.py`), поэтому
матрица растёт вместе с ним.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_generator() -> Any:
    path = REPO_ROOT / "scripts" / "gen-audit-scenarios.py"
    spec = importlib.util.spec_from_file_location("gen_audit_scenarios", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bench() -> Any:
    """Сам стенд, чтобы проверить его вердикт без живого прогона."""
    path = REPO_ROOT / "scripts" / "live-eval.py"
    spec = importlib.util.spec_from_file_location("live_eval_bench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generator = _load_generator()
_bench = _load_bench()
SCENARIOS: list[dict[str, Any]] = _generator.build_scenarios()


@pytest.fixture(scope="module", autouse=True)
def _config() -> Any:
    """The hub's own config, so the narrowing rules run with real thresholds."""
    from common.config import load_config
    from hub import app as hub_app

    cfg = load_config(str(REPO_ROOT / "config.openai.yaml"))
    hub_app.configure(cfg)
    return cfg


def _ids(rows: list[Any]) -> list[str]:
    return [f"{row[0]}-{'/'.join(row[1:])}" if isinstance(row, tuple) else str(row) for row in rows]


def _expected_names(scenario: dict[str, Any]) -> list[str]:
    names = list(scenario.get("expect_tools") or [])
    names += list(scenario.get("expect_any") or [])
    names += list(scenario.get("expect_first") or [])
    names += list(scenario.get("forbid_tools") or [])
    return names


CORPUS_CASES = [(item["id"], item["said"], name)
                for item in SCENARIOS
                for name in _expected_names(item)]


def test_the_corpus_is_large_enough() -> None:
    """The owner asked for thousands of scenarios; the corpus must stay big."""
    assert len(SCENARIOS) >= 900, f"корпус усох до {len(SCENARIOS)} сценариев"


def test_scenario_ids_are_unique() -> None:
    ids = [item["id"] for item in SCENARIOS]
    assert len(ids) == len(set(ids))


def test_no_scenario_repeats_a_sentence() -> None:
    said = [item["said"] for item in SCENARIOS]
    duplicated = {text for text in said if said.count(text) > 1}
    assert not duplicated, f"дубли реплик: {sorted(duplicated)[:5]}"


@pytest.mark.parametrize("scenario_id,said,name", CORPUS_CASES, ids=_ids(CORPUS_CASES))
def test_expected_tool_exists_in_the_contract(scenario_id: str, said: str, name: str) -> None:
    """An expectation for a tool that does not exist would fail every run."""
    from hub.tools import TOOL_NAMES

    assert name in TOOL_NAMES, f"{scenario_id}: «{said}» ждёт несуществующий {name!r}"


FAMILY_CASES: list[tuple[str, str, str, str]] = []
for _item in SCENARIOS:
    from hub.tools import TOOL_FAMILIES  # noqa: E402 - table built at import time

    _needed = list(_item.get("expect_tools") or []) + list(_item.get("expect_first") or [])
    for _family, _members in TOOL_FAMILIES.items():
        _inside = [name for name in _needed if name in _members]
        if _inside:
            FAMILY_CASES.append((_item["id"], _family, ",".join(_inside), _item["said"]))


@pytest.mark.parametrize("scenario_id,family,needed,said", FAMILY_CASES, ids=_ids(FAMILY_CASES))
def test_family_narrowing_keeps_the_needed_tool(scenario_id: str, family: str,
                                                needed: str, said: str) -> None:
    """Jev's narrowing may only take tools away (U-14) — never the right one.

    ``None`` from ``_narrow_tools_for`` means "every tool stays", which also
    offers the needed one, so both answers are accepted.
    """
    from hub.app import _narrow_tools_for
    from hub.tools import TOOLS

    understanding = {"act": {"value": True, "confidence": 0.99},
                     "family": {"value": family, "confidence": 0.99}}
    offered = _narrow_tools_for(understanding) or TOOLS
    names = {tool["function"]["name"] for tool in offered}
    for name in needed.split(","):
        assert name in names, (
            f"{scenario_id}: «{said}» — сужение на семейство {family} спрятало {name}")


def test_a_vision_question_keeps_vision_tools_even_when_jev_says_no_action() -> None:
    """AU-03: «кто в комнате», «что на экране» — вопрос, но смотреть всё равно.

    Живой прогон 2026-09-23: на такие реплики Jev отвечал ``act=false``
    (это вопрос) и ``family=vision`` одним чтением, а ветка «просто вопрос»
    отдавала модели только ядро — ``look_at_camera``/``look_at_screen``
    исчезали, и сценарий падал.
    """
    from hub.app import _narrow_tools_for

    cases = [item for item in SCENARIOS
             if item["family"] == "vision" and item.get("expect_first")]
    assert cases, "сценарии зрения пропали из корпуса"
    understanding = {"act": {"value": False, "confidence": 0.99},
                     "family": {"value": "vision", "confidence": 0.99}}
    names = {tool["function"]["name"] for tool in _narrow_tools_for(understanding)}
    for item in cases:
        needed = item["expect_first"][0]
        assert needed in names, f"{item['id']}: «{item['said']}» потерял {needed}"


def test_the_room_prompt_sends_a_named_object_to_find_object() -> None:
    """AU-03: «do you see my phone» — поиск конкретной вещи, а не общий осмотр.

    Живой прогон: на «do you see my phone/laptop» модель звала
    ``look_at_camera`` и никогда ``find_object``, хотя Jev уже отдал весь
    набор зрения. Промпт комнаты и описание инструмента обязаны называть эти
    слова явно — иначе двусмысленность решает модель, а не аудит.
    """
    from hub.tools import TOOLS

    prompt = " ".join(
        (REPO_ROOT / "prompts" / "system.md").read_text(encoding="utf-8").split()
    ).casefold()
    description = next(
        " ".join(tool["function"]["description"].split()).casefold()
        for tool in TOOLS if tool["function"]["name"] == "find_object")
    for phrase in ("where is my", "find my", "do you see my", "can you see my"):
        assert phrase in prompt, f"промпт не отправляет «{phrase}» в find_object"
        assert phrase in description, f"описание find_object не знает «{phrase}»"
    assert "call it first" in description


def test_a_telegram_photo_scenario_is_skipped_by_the_bench() -> None:
    """В стенде нет вложения Telegram, но это не поломка модели (AUDIT-07)."""
    photos = [item for item in SCENARIOS
              if "inspect_photo" in (item.get("expect_tools") or [])]
    assert photos, "сценарии с присланным фото пропали из корпуса"
    for item in photos:
        assert item.get("bench_skip"), f"{item['id']}: оценка фото без вложения"


# --- медиа: картинки, обои, показ (AU-07) ------------------------------------

MEDIA_CASES = [item for item in SCENARIOS if item["family"] == "media"]


def _collapsed(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split()).casefold()


def _description(name: str) -> str:
    from hub.tools import TOOLS

    return " ".join(
        next(tool["function"]["description"] for tool in TOOLS
             if tool["function"]["name"] == name).split()).casefold()


def test_an_announcement_is_spoken_out_loud_and_not_messaged() -> None:
    """«Скажи всем, что ужин готов» — say_in_room, а не Telegram.

    Живой прогон 2026-09-23 (AU-07): на «tell everyone that dinner is ready»
    модель прочитала слово «everyone» как рассылку и ответила «у меня один
    общий чат, хочешь — напишу туда», хотя просьба произнести это вслух в
    комнатах. Промпт комнаты и описание инструмента обязаны говорить одно и то
    же — как AUDIT-08 для сайтов и AUDIT-11 для правил.
    """
    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    assert "tell everyone" in prompt, "промпт не называет «tell everyone»"
    assert "`say_in_room` plays exactly your words on the room speaker" in prompt, \
        "промпт не отправляет объявление в say_in_room"
    assert "is never a telegram send" in prompt, \
        "промпт не отличает слово вслух от сообщения в Telegram"
    description = _description("say_in_room")
    for phrase in ("tell everyone", "announce"):
        assert phrase in description, f"описание say_in_room не знает «{phrase}»"
    spoken = [item for item in MEDIA_CASES
              if "tell everyone" in str(item["said"]).casefold()]
    assert spoken, "просьбы «tell everyone …» пропали из корпуса"
    for item in spoken:
        assert item.get("expect_first") == ["say_in_room"], item["id"]


def test_a_photo_edit_scenario_is_skipped_by_the_bench() -> None:
    """Правка присланного фото: у generate_image нет источника «вложение».

    Живой прогон AU-07: на «edit this photo and make it warmer» модель
    половину раз честно отвечала «вложения нет» — и это верный ход, потому что
    ``generate_image`` принимает source = none / camera / screen / last, а
    присланное фото живёт только в Telegram-ходе
    (``hub/telegram_chat.py::_image_reference``). Стенд фото не шлёт (та же
    причина, что у ``inspect_photo``, AUDIT-09d), поэтому сценарий печатает
    SKIP, а сам путь проверяют ``tests/test_telegram_edit_prompt.py`` и
    ``tests/test_telegram_reply_photo.py``.
    """
    from hub.tools import TOOLS

    edits = [item for item in MEDIA_CASES
             if "edit this photo" in str(item["said"]).casefold()
             or "attached picture" in str(item["said"]).casefold()]
    assert edits, "сценарии правки присланного фото пропали из корпуса"
    for item in edits:
        assert item.get("expect_tools") == ["generate_image"], item["id"]
        assert item.get("bench_skip"), f"{item['id']}: правка без вложения"
    sources = next(tool["function"]["parameters"]["properties"]["source"]["enum"]
                   for tool in TOOLS if tool["function"]["name"] == "generate_image")
    assert sources == ["none", "camera", "screen", "last"], (
        "у generate_image появился источник вложения — стенд обязан его слать")


def test_a_new_wallpaper_is_drawn_and_an_existing_one_is_installed() -> None:
    """Обои из новой картинки — generate_image target=wallpaper; готовой — set_wallpaper.

    Живой прогон AU-07: обе просьбы верны («make a wallpaper of a cat» рисует
    и ставит, «put this picture on my wallpaper» ставит уже готовую). Промпт
    обязан называть оба хода, иначе модель либо ставит картинку, которой ещё
    нет, либо рисует вместо установки.
    """
    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    assert "target=wallpaper" in prompt, "промпт не называет обои через generate_image"
    assert "`set_wallpaper`" in prompt, "промпт не называет установку готовой картинки"
    assert "do not call `look_at_screen`, `look_at_camera` or `show_photo` first" in prompt, (
        "промпт снова разрешает искать картинку для обоев отдельным взглядом")
    assert "the tool's own answer says whether a picture exists" in prompt, (
        "промпт снова позволяет отвечать за set_wallpaper, не вызвав его")
    description = _description("set_wallpaper")
    assert "do not call look_at_screen" in description, (
        "описание set_wallpaper не запрещает искать картинку взглядом")
    drawn = [item for item in MEDIA_CASES
             if "wallpaper" in str(item["said"]).casefold()
             and item.get("expect_first") == ["generate_image"]]
    installed = [item for item in MEDIA_CASES
                 if item.get("expect_first") == ["set_wallpaper"]]
    assert drawn, "просьбы «сделай обои из …» пропали из корпуса"
    assert installed, "просьбы «поставь эту картинку на обои» пропали из корпуса"
    assert _description("generate_image").count("target=wallpaper") >= 1
    assert "existing image" in description


def test_a_named_object_is_checked_by_the_simple_word() -> None:
    """Описание find_object само просит простое слово — проверка не строже его.

    Живой прогон AU-0573: по просьбе «where is my keys» модель искала «key»,
    a корпус требовал буквально «keys» и падал на верном ходе.
    """
    words = [word for item in SCENARIOS
             if item.get("expect_first") == ["find_object"]
             for word in (item.get("expect_args") or {}).get("find_object", [])]
    assert words, "проверка слова для find_object пропала из корпуса"
    plurals = sorted({str(word) for word in words if str(word).endswith("s")})
    assert not plurals, f"проверка требует множественное число: {plurals}"


def test_reading_the_browser_window_accepts_the_page_or_the_screen() -> None:
    """«Прочитай окно браузера» — верно и взглядом, и чтением страницы.

    Живой прогон AU-0513: модель один раз прочитала страницу
    ``browser_control read``, другой раз посмотрела на экран — оба хода
    отвечают на просьбу (AUDIT-08b в DECISIONS.md).
    """
    items = [item for item in SCENARIOS
             if item["family"] == "vision" and "browser window" in item["said"].casefold()]
    assert items, "сценарии «окно браузера» пропали из корпуса"
    for item in items:
        allowed = set(item.get("expect_any") or [])
        assert {"look_at_screen", "browser_control"} <= allowed, (
            f"{item['id']}: «{item['said']}» снова требует один инструмент")
        assert not item.get("expect_first"), f"{item['id']}: «первый обязательный» неверен"


PC_SLOTS = [("open_app", "value", "chrome"), ("open_app", "target", "chrome"),
            ("close_app", "target", "spotify"), ("minimize_app", "value", "discord"),
            ("focus_app", "target", "code"), ("volume_up", "value", "ignored")]


@pytest.mark.parametrize("command,slot,name", PC_SLOTS, ids=_ids(PC_SLOTS))
def test_pc_control_app_name_reaches_the_client(command: str, slot: str, name: str) -> None:
    """The model sends the app name in either slot; the client reads ``value``."""
    from hub.tools import PC_APP_COMMANDS, normalize_pc_control_args

    cleaned = normalize_pc_control_args({"command": command, slot: name})
    if command in PC_APP_COMMANDS:
        assert cleaned.get("value") == name, f"{command}: имя приложения потерялось ({cleaned})"
        assert "target" not in cleaned or cleaned.get("target") == name
    else:
        assert cleaned == {"command": command, slot: name}, "чужая команда не должна меняться"


@pytest.mark.parametrize("value,expected", [
    (None, "left"), ("", "left"), ("right", "right"), ("context", "right"),
    ("double click", "double"), ("middle", "left"), ("RIGHT", "right"),
], ids=lambda value: str(value))
def test_click_button_normalisation(value: Any, expected: str) -> None:
    from hub.tools import normalize_click_button

    assert normalize_click_button(value) == expected


# --- один смысл не может требовать двух разных инструментов -------------------


def test_the_ambiguous_names_are_really_ambiguous() -> None:
    """Имена, которые есть и сайтом, и установленной программой."""
    assert _generator.AMBIGUOUS_SITE_APPS, "список спорных имён опустел"
    sites = {name for name, _ in _generator.SITES}
    apps = {name for name, _ in _generator.APPS}
    assert _generator.AMBIGUOUS_SITE_APPS == sites & apps


# --- люди: запись лица и голоса (AU-04) --------------------------------------

PEOPLE_CASES = [item for item in SCENARIOS if item["family"] == "people"]


def test_a_save_this_person_request_accepts_a_face_or_a_voice() -> None:
    """«Сохрани этого человека как X» верно и лицом, и голосом.

    Живой прогон AU-0811: на «can you save this person as Theodric» модель
    записала голос с верным именем, а корпус требовал именно лицо — ругать
    модель за верный ход не аудит (AUDIT-08b, AUDIT-09e).
    """
    checked = 0
    for item in PEOPLE_CASES:
        if "save this person" not in str(item["said"]).casefold():
            continue
        checked += 1
        allowed = set(item.get("expect_any") or [])
        assert {"enroll_face", "enroll_voice"} <= allowed, (
            f"{item['id']}: «{item['said']}» снова требует один способ записи")
        assert not item.get("expect_first"), f"{item['id']}: «первый обязательный» неверен"
    assert checked, "просьбы «save this person as» пропали из корпуса"


def test_a_rename_request_names_the_new_name() -> None:
    """rename_person требует ОБА имени, поэтому корпус называет новое.

    «change the name of John» без нового имени — честный ход модели это
    вопрос «на какое имя?», а не вызов с пустым ``new_name``.
    """
    renames = [item for item in PEOPLE_CASES
               if "rename_person" in (item.get("expect_tools") or [])]
    assert renames, "просьбы переименования пропали из корпуса"
    for item in renames:
        assert " to " in str(item["said"]).casefold(), (
            f"{item['id']}: «{item['said']}» не называет новое имя")


def test_the_bench_room_knows_the_people_the_corpus_names() -> None:
    """В комнате стенда люди записаны — иначе «сделай Джона админом» не о том.

    Живой стенд отвечал на ``list_people``/``set_role``/``rename_person``
    «speaker recognition is disabled» (реестра не было), и модель отказывалась
    за хаб. Реестр стенда — свой (TEST-DB-01), а имена — те, что называет
    корпус.
    """
    assert _bench.BENCH_PEOPLE, "у комнаты стенда нет людей"
    known = {name.casefold() for name, _ in _bench.BENCH_PEOPLE}
    wanted = {word.casefold()
              for item in PEOPLE_CASES
              for words in (item.get("expect_args") or {}).values()
              for word in words}
    missing = sorted(wanted - known)
    assert not missing, f"корпус называет людей, которых в комнате стенда нет: {missing}"


def test_the_bench_people_registry_is_not_the_owners() -> None:
    """Стенд пишет свои данные, а не ``data/people.json`` владельца."""
    assert _bench.PEOPLE_DIR.parent == _bench.REPORTS
    assert _bench.PEOPLE_DIR != REPO_ROOT / "data"
    assert ".tmp" not in str(_bench.PEOPLE_DIR)


# --- память: remember / forget / list / recall (AU-08) ------------------------

MEMORY_CASES = [item for item in SCENARIOS if item["family"] == "memory"]


def test_the_bench_room_keeps_its_own_memory_and_archive(tmp_path: Path,
                                                         monkeypatch: Any,
                                                         _config: Any) -> None:
    """Стенд читал и писал ``data/memory.jsonl`` ВЛАДЕЛЬЦА.

    ``Memory()`` и ``Conversations(...)`` без каталога — это ``data/`` в корне
    репозитория. Комната стенда поэтому просыпалась с фактами из прошлых
    прогонов («recent facts: "Anton likes tea."» в префиксе хода), модель
    честно отвечала «ты уже говорил», не вызывая ``remember``, а вопросы «что мы
    обсуждали» получали «Conversation storage is unavailable» (массовый аудит
    2026-09-23, AU-08). Каталог стенда — свой (TEST-DB-01), как у реестра людей.
    """
    import sqlite3
    from datetime import datetime, timedelta

    monkeypatch.setattr(_bench, "PEOPLE_DIR", tmp_path / "people")
    bench = _bench.Bench(_config, actions=False, worker=97)
    assert bench.data_dir != REPO_ROOT / "data"
    bench._build_room_fixtures()
    assert bench.memory_fixture.exists() and bench.archive_fixture.exists()

    remembered = bench.memory_fixture.read_text(encoding="utf-8")
    for _, fact in _bench.BENCH_FACTS:
        assert fact in remembered, f"в фикстуре памяти нет «{fact}»"

    with sqlite3.connect(bench.archive_fixture) as db:
        rows = db.execute("SELECT person, ts, question FROM turns").fetchall()
    assert len(rows) == len(_bench.BENCH_CONVERSATIONS)
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    assert all(str(row[1]).startswith(yesterday) for row in rows), (
        "история стенда не вчерашняя: «что мы обсуждали вчера» искало бы её зря")


def test_a_room_that_already_knows_the_fact_cannot_answer_for_remember() -> None:
    """Фикстура памяти не содержит предложений, которые корпус просит сохранить.

    Иначе ход «remember that I like tea» снова упирался бы в «ты уже говорил»
    и измерял бы фикстуру, а не модель.
    """
    saved = " ".join(fact for _, fact in _bench.BENCH_FACTS).casefold()
    wanted = {str(word).casefold()
              for item in MEMORY_CASES
              for words in (item.get("expect_args") or {}).values()
              for word in words}
    assert wanted, "проверка слов памяти пропала из корпуса"
    clash = sorted(word for word in wanted if word in saved)
    assert not clash, f"комната стенда уже знает то, о чём сценарий просит: {clash}"


def test_a_recall_scenario_only_checks_words_the_person_said() -> None:
    """Проверка слова берётся из САМОЙ реплики, а не из соседнего предмета.

    «do you remember what I said about the exam» требовало слово «dorm» —
    ожидание соседней формулировки, которого реплика не называла: верный ход
    модели («query: exam») падал (массовый аудит 2026-09-23, AU-08).
    """
    recalled = [item for item in MEMORY_CASES
                if "recall_conversation" in (item.get("expect_any") or [])
                and (item.get("expect_args") or {}).get("recall_conversation")]
    assert recalled, "сценарии с проверкой слова в recall_conversation пропали"
    for item in recalled:
        for word in item["expect_args"]["recall_conversation"]:
            assert str(word).casefold() in str(item["said"]).casefold(), (
                f"{item['id']}: «{item['said']}» требует слово {word!r}, которого не называет")


def test_a_recall_about_a_day_checks_no_subject_word() -> None:
    """«Вчера» — это время, а не тема: слова в аргументах не проверяем.

    Модель вправе передать окно датами (``since``/``until``) — она так и
    делает, — и требовать при этом слово «yesterday» значило бы ругать её за
    верный разбор просьбы.
    """
    yesterday = [item for item in MEMORY_CASES
                 if "yesterday" in str(item["said"]).casefold()
                 and "recall_conversation" in (item.get("expect_any") or [])]
    assert yesterday, "сценарии «что мы обсуждали вчера» пропали из корпуса"
    for item in yesterday:
        assert not (item.get("expect_args") or {}).get("recall_conversation"), (
            f"{item['id']}: «{item['said']}» снова проверяет тему вместо времени")


def test_a_forget_scenario_states_what_the_room_already_knows() -> None:
    """Удалять можно только то, что комната слышала.

    Живой прогон AU-08: на «forget that I like tea» пустая комната стенда
    отвечала «ничего такого у меня нет» — верная фраза о комнате без памяти,
    из-за которой сценарий падал на верном ходе модели. Сценарий теперь сам
    называет факт, который комната знает ДО хода (``assumes_fact``), а стенд
    кладёт его в свою память через сам хаб.
    """
    forgetting = [item for item in MEMORY_CASES
                  if item.get("expect_tools") == ["forget_fact"]]
    assert forgetting, "сценарии «забудь …» пропали из корпуса"
    for item in forgetting:
        assumed = item.get("assumes_fact") or {}
        fact = str(assumed.get("fact") or "")
        assert fact, f"{item['id']}: «{item['said']}» не говорит, что комната уже знает"
        assert assumed.get("about"), f"{item['id']}: предусловие без владельца факта"
        for word in (item.get("expect_args") or {}).get("forget_fact", []):
            assert str(word).casefold() in fact.casefold(), (
                f"{item['id']}: проверяет {word!r}, которого нет в предусловии {fact!r}")


def test_a_remembered_time_may_be_written_in_digits() -> None:
    """Час можно записать словами или цифрами — это один и тот же факт.

    Модель сохраняла «Anton wakes up at 7:00» и падала на ожидании слова
    «seven», хотя факт записан верно (живой прогон AU-08).
    """
    wake = [item for item in MEMORY_CASES
            if "wake up at seven" in str(item["said"]).casefold()]
    assert wake, "сценарии про подъём пропали из корпуса"
    for item in wake:
        words = (item.get("expect_args") or {}).get("remember", [])
        assert "seven" not in [str(word).casefold() for word in words], (
            f"{item['id']}: снова требует написание часа вместо самого факта")


def test_the_bench_stores_the_fact_a_forget_scenario_assumes(tmp_path: Path,
                                                             monkeypatch: Any,
                                                             _config: Any) -> None:
    """Стенд кладёт предусловие в память комнаты, а не в файл владельца."""
    from hub.storage import Memory

    monkeypatch.setattr(_bench, "PEOPLE_DIR", tmp_path / "people")
    bench = _bench.Bench(_config, actions=False, worker=96)
    bench._build_room_fixtures()
    _bench.ROOM_MEMORY.bind(Memory(bench.data_dir))
    scenario = next(item for item in MEMORY_CASES
                    if (item.get("assumes_fact") or {}).get("fact"))
    bench._apply_assumed_facts(scenario)
    facts = _bench.ROOM_MEMORY.facts("Anton")
    assert scenario["assumes_fact"]["fact"] in facts, facts
    assert bench.data_dir != REPO_ROOT / "data"


def test_the_prompt_and_the_tools_separate_facts_from_conversations() -> None:
    """«Что ты знаешь обо мне» — это факты, «что я говорил» — история.

    Живой прогон AU-08: на «do you remember what I said about the exam» модель
    отвечала сохранённым фактом про экзамен (``list_memory``) вместо поиска в
    истории разговоров. Промпт и описания инструментов обязаны говорить одно и
    то же — как AUDIT-08 для сайтов и AUDIT-11 для правил.
    """
    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    assert '"what do you know about me"' in prompt or "what do you know about me" in prompt
    assert "saved facts are not a conversation" in prompt, \
        "промпт не различает сохранённый факт и разговор"
    for phrase in ("do you remember what i said about", "find the conversation where we discussed",
                   "what did we talk about"):
        assert phrase in _description("recall_conversation"), \
            f"описание recall_conversation не знает «{phrase}»"
    assert "list_memory" in _description("recall_conversation"), \
        "recall_conversation не отличает себя от list_memory"
    assert "recall_conversation" in _description("list_memory"), \
        "list_memory не отправляет вопрос «что я говорил» в историю"


def test_recall_conversation_declares_the_dates_it_really_honours() -> None:
    """Инструмент читает ``since``/``until``/``limit``/``person`` — и объявляет их.

    ``Connection._execute_tool_now`` разбирает эти поля, а схема объявляла один
    ``query``: живой ход отбрасывал даты модели как необъявленный аргумент
    (``hub/tool_args.py``), и «что мы обсуждали вчера» искало по всему архиву.
    """
    from hub.tools import TOOLS

    schema = next(tool["function"]["parameters"] for tool in TOOLS
                  if tool["function"]["name"] == "recall_conversation")
    declared = set(schema["properties"])
    assert {"query", "person", "since", "until", "limit"} <= declared, declared
    assert schema["required"] == ["query"]


def test_a_tool_refusal_is_not_read_as_a_dead_key() -> None:
    """Ответ инструмента «на сервере нет распознавания лиц» — не отказ ключа."""
    scenario = {"id": "T", "said": "remember this face as my roommate",
                "expect_tools": ["enroll_face"],
                "expect_args": {"enroll_face": ["roommate"]}}
    run = _run(["enroll_face"],
               [{"tool": "enroll_face", "args": {"name": "my roommate"}}],
               reply="Face recognition is unavailable on the server right now, "
                     "so I can't save his face.")
    ok, problems = _bench.judge(scenario, run)
    assert ok, problems


def test_a_dead_model_still_fails_the_scenario() -> None:
    """А вот «модель не ответила» обязано остаться провалом."""
    scenario = {"id": "T", "said": "make John an admin",
                "expect_tools": ["set_role"]}
    run = _run([], [], reply="API accounting is unavailable; I couldn't finish "
                             "this request.")
    ok, problems = _bench.judge(scenario, run)
    assert not ok
    assert any("no model answered" in problem for problem in problems), problems


def test_the_prompt_sends_a_named_person_to_the_enrolment_tool() -> None:
    """«Это Макс, запомни его лицо» — вызов, а не отказ за инструмент.

    Живой прогон AU-0797/AU-0830: модель отвечала «только сам человек может
    записать себя» и не звала инструмент, хотя F-210 разрешает владельцу
    записать гостя по имени. Инструкция модели и договор инструментов обязаны
    говорить одно и то же.
    """
    from hub.tools import TOOLS

    prompt = " ".join(
        (REPO_ROOT / "prompts" / "system.md").read_text(encoding="utf-8").split()
    ).casefold()
    for phrase in ("never answer that only the person themselves can be enrolled",
                   "the tool is the enrollment"):
        assert phrase in prompt, f"промпт не отправляет названного человека в запись: {phrase!r}"
    for tool, phrase in (("enroll_face", "memorize his face"),
                         ("enroll_voice", "this is max"),
                         ("set_role", "call the tool and relay its answer")):
        description = " ".join(
            next(item["function"]["description"] for item in TOOLS
                 if item["function"]["name"] == tool).split()).casefold()
        assert phrase in description, f"описание {tool} не знает «{phrase}»"


def test_open_a_shared_name_accepts_a_page_or_a_program() -> None:
    """«Открой spotify» одинаково верно и страницей, и приложением.

    Массовый аудит 2026-09-23: браузерные сценарии требовали
    ``browser_control``, а ПК-сценарии на ту же просьбу — ``pc_control``, и
    верный ход модели падал в одном из двух семейств (DECISIONS.md AUDIT-08).
    """
    seen = 0
    for item in SCENARIOS:
        # Только одиночные просьбы: в PAIRS («выключи звук и открой spotify»)
        # список инструментов значит «обязаны быть вызваны все», и там
        # pc_control нужен для половины про громкость.
        if item["family"] not in ("browser", "pc") \
                or not _generator._page_or_app_name(item["said"]):
            continue
        seen += 1
        allowed = set(item.get("expect_any") or [])
        assert {"browser_control", "pc_control"} <= allowed, (
            f"{item['id']}: «{item['said']}» снова требует один инструмент: {allowed}")
        assert not item.get("expect_first"), (
            f"{item['id']}: у спорной просьбы не может быть «первого обязательного»")
    assert seen >= 2, "спорные просьбы пропали из корпуса"


def test_a_named_page_still_demands_the_browser() -> None:
    """Как только назван домен или сказано «в браузере», верный инструмент один."""
    checked = 0
    for item in SCENARIOS:
        said = str(item["said"]).casefold()
        if not any(name in said for name in _generator.AMBIGUOUS_SITE_APPS):
            continue
        if "in the browser" not in said and ".com" not in said:
            continue
        checked += 1
        assert item.get("expect_tools") == ["browser_control"] or \
            "browser_control" in (item.get("expect_any") or []), item["id"]
    assert checked > 0, "сценарии с явной страницей не найдены"


def test_a_weather_look_up_may_use_the_home_skill() -> None:
    """Погоду в этом доме умеет собственный скилл (``skills/weather``).

    «Найди погоду в омахе» — один вызов ``run_skill``, и он точнее поиска в
    браузере; требовать только браузер значит ругать модель за лучший ход.
    """
    checked = 0
    for item in SCENARIOS:
        said = str(item["said"]).casefold()
        if not said.startswith(("look up weather", "find weather")):
            continue
        checked += 1
        assert "run_skill" in (item.get("expect_any") or []), item["id"]
    assert checked > 0, "просьбы про погоду пропали из корпуса"


def test_inside_the_page_scenarios_admit_the_bench_cannot_type() -> None:
    """fill берёт ref из read: без --actions страницы у стенда нет.

    Сценарий не выбрасывается из корпуса — он честно помечен
    ``needs_actions``, и ``scripts/live-eval.py`` печатает по нему SKIP, а не
    провал модели.
    """
    typing = [item for item in SCENARIOS
              if item.get("expect_args")
              and ("search box" in str(item["said"]) or "address bar" in str(item["said"]))]
    assert typing, "сценарии набора текста пропали из корпуса"
    for item in typing:
        assert item.get("needs_actions"), (
            f"{item['id']}: «{item['said']}» снова считается без реальной страницы")


# --- вердикт стенда ----------------------------------------------------------


def _run(tools: list[str], calls: list[dict[str, Any]], reply: str = "Done.") -> dict[str, Any]:
    return {"id": "T", "said": "open spotify", "reply": reply, "error": "",
            "tools": tools, "calls": calls, "offered": None, "seconds": 1.0}


def test_the_word_from_the_request_may_come_through_either_allowed_tool() -> None:
    """Спорная просьба: слово из просьбы ищется в том вызове, что был сделан."""
    scenario = {"id": "T", "said": "open spotify",
                "expect_any": ["browser_control", "pc_control"],
                "expect_args": {"browser_control": ["spotify"]}}
    run = _run(["pc_control"], [{"tool": "pc_control",
                                 "args": {"command": "open_app", "value": "spotify"}}])
    ok, problems = _bench.judge(scenario, run)
    assert ok, problems


def test_a_page_opened_by_the_shell_is_not_a_pass() -> None:
    """Откат на run_command не считается верным ходом (AUDIT-02)."""
    scenario = {"id": "T", "said": "open youtube",
                "expect_tools": ["browser_control"], "expect_first": ["browser_control"]}
    run = _run(["run_command", "browser_control"],
               [{"tool": "run_command", "args": {"command": 'Start-Process "https://youtube.com"'}},
                {"tool": "browser_control", "args": {"command": "navigate",
                                                     "url": "https://www.youtube.com"}}])
    ok, problems = _bench.judge(scenario, run)
    assert not ok
    assert any("first tool" in problem for problem in problems), problems


# --- уведомления и правила (AU-05) --------------------------------------------

NOTIFY_CASES = [item for item in SCENARIOS if item["family"] == "notify"]


def test_a_rule_request_is_a_call_and_not_a_question() -> None:
    """«Сообщи, когда кто-то войдёт» — правило, а не вопрос «куда сказать?».

    Живой прогон AU-0929/AU-0931: на «notify me when the door opens» модель
    спрашивала, что должно случиться, вместо ``create_rule``, хотя у правила
    уже есть подтверждение голосом (F-113). Промпт комнаты и описание
    инструмента обязаны говорить одно и то же.
    """
    from hub.tools import TOOLS

    prompt = " ".join(
        (REPO_ROOT / "prompts" / "system.md").read_text(encoding="utf-8").split()
    ).casefold()
    description = " ".join(
        next(item["function"]["description"] for item in TOOLS
             if item["function"]["name"] == "create_rule").split()).casefold()
    for phrase in ("propose the rule and let the spoken yes confirm it",
                   "do not answer with a question about how they want to be told"):
        assert phrase in prompt, f"промпт не отправляет правило в create_rule: {phrase!r}"
    for phrase in ("a named place - the door, the window, the desk - is one of the frame zones",
                   "never answer it with a question about what the notification should look like",
                   "still propose the rule with a notify or say action"):
        assert phrase in description, f"описание create_rule не знает «{phrase}»"


def test_a_private_message_is_not_posted_to_the_one_group_chat() -> None:
    """«Скажи Джону, что я иду домой» — не адрес: у дома один общий чат.

    Хаб отказывает такому вызову сам (``hub/telegram_intent.py``), поэтому
    корпус не может требовать ``telegram_send``: он проверял бы вызов,
    который хаб обязан отклонить.
    """
    from hub.telegram_intent import telegram_send_requested

    private = [item for item in NOTIFY_CASES
               if "tell john" in str(item["said"]).casefold()
               or "tell max" in str(item["said"]).casefold()]
    assert private, "сценарии «скажи <имя>» пропали из корпуса"
    for item in private:
        assert item.get("forbid_tools") == ["telegram_send"], item["id"]
        assert item.get("note"), f"{item['id']}: отказ без пояснения"
        assert not telegram_send_requested(str(item["said"])), (
            f"{item['id']}: хаб снова читает личное имя как адрес Telegram")
    for phrase in ("send a message to the telegram group that dinner is ready",
                   "message the group: I am on my way"):
        assert telegram_send_requested(phrase), phrase


def test_a_dark_room_is_an_honest_refusal() -> None:
    """Темноту в этом доме нечем измерить: правило о ней — выдумка.

    В F-419 четыре триггера, и ни один не про освещённость; датчика тоже нет
    (``config.openai.yaml``: ``devices: []``). Первый прогон аудита «прошёл»
    на выдуманном ``device_state device_id=room_light`` — это фейк в корпусе,
    а не достижение модели.
    """
    dark = [item for item in NOTIFY_CASES if "gets dark" in str(item["said"]).casefold()]
    assert dark, "сценарии «станет темно» пропали из корпуса"
    for item in dark:
        assert "create_rule" in (item.get("forbid_tools") or []), item["id"]
        assert "датчик" in str(item.get("note") or ""), (
            f"{item['id']}: отказ без пояснения, что датчика нет")


def test_the_bench_records_the_turn_for_the_transcript_policies() -> None:
    """Стенд обязан записать ход: политики читают слова человека оттуда.

    ``telegram_send`` отправляет только когда ЭТОТ ход просил Telegram
    (``_recording_turn``), а ``generate_image`` берёт оттуда буквальную
    формулировку. Стенд звал модель напрямую, поэтому каждое явное «отправь в
    группу» получало отказ «requires an explicit user request in this turn» —
    измерялся хаб с пустым транскриптом (массовый аудит 2026-09-23, AU-05).
    """
    from types import SimpleNamespace

    from hub import app as hub_app

    seen: dict[str, Any] = {}

    class _Llm:
        async def generate(self, request: Any, execute: Any, tools: Any = None) -> Any:
            seen["turn"] = hub_app._recording_turn.get()
            return SimpleNamespace(text="ok")

    bench = object.__new__(_bench.Bench)
    bench.connection = SimpleNamespace(
        session=SimpleNamespace(system_prompt="sys"),
        _turn_prefix=lambda at, text: f"[at ... ] {text}",
    )
    bench._llm = _Llm()
    bench.understanding = False
    bench.calls = []
    bench.messages = []
    bench.devices = []
    bench.skills = []
    bench._reset_room = lambda: None  # noqa: ARG005 - файлы стенда тут не нужны
    said = "send a message to the telegram group that dinner is ready"

    import asyncio

    asyncio.run(bench.run({"id": "T", "said": said}))

    assert seen["turn"], "стенд не записал ход: политики видят пустой транскрипт"
    assert seen["turn"]["transcript"] == said
    assert hub_app._recording_turn.get() is None, "запись хода не снята после прогона"


def _config_with_devices(tmp_path: Path, devices: str) -> Any:
    """Живой конфиг стенда с прописанными приборами (AU-06).

    Комната называет приборы сама — в ``hello``, который клиент собирает из
    ``client.devices`` (``client.main.build_hello``). Поэтому и проверка берёт
    настоящий конфиг и меняет в нём ровно этот список.
    """
    from common.config import load_config

    source = (REPO_ROOT / "config.openai.yaml").read_text(encoding="utf-8")
    assert "\n  devices: []" in source, "в конфиге стенда больше нет пустого client.devices"
    patched = source.replace("\n  devices: []", "\n  devices:\n" + devices, 1)
    path = tmp_path / "config.devices.yaml"
    path.write_text(patched, encoding="utf-8")
    return load_config(str(path))


def test_the_bench_room_takes_its_devices_from_the_room_itself(tmp_path: Path) -> None:
    """AU-06: стенд не выдумывает приборы, а спрашивает конфиг комнаты.

    ``[home: ...]`` и список устройств системного промпта живой ход берёт не из
    хаба, а из ``hello`` комнаты. Стенд собирает тот же кадр тем же кодом
    клиента (``client.main.build_hello``), поэтому обе стороны говорят об одной
    комнате: у этой комнаты приборов нет, у комнаты из фикстуры они есть.
    """
    live = _config_with_devices(tmp_path, "    []")
    assert _bench.room_devices(live) == [], "стенд придумал приборы живой комнате"

    furnished = _config_with_devices(
        tmp_path,
        "    - name: Desk lamp\n      type: magichome\n      area: room\n")
    devices = _bench.room_devices(furnished)
    assert [device["name"] for device in devices] == ["Desk lamp"]
    assert devices[0]["type"] == "magichome"

    from hub.session import Session

    prompt = Session(client_id="livingroom", devices=devices,
                     history_turns=4).system_prompt
    assert "Desk lamp" in prompt, "список устройств комнаты не дошёл до модели"

    from hub.speaker_context import home_state_from, render_home

    block = render_home(home_state_from(home_id="livingroom", devices=devices,
                                        skills=["weather"]))
    assert "Desk lamp" in block and "skills: weather" in block, (
        "живой блок [home: ...] не назвал приборы и скиллы комнаты")


def test_the_device_scenarios_follow_the_devices_of_the_room() -> None:
    """Прибор, которого комната не называла, — честный отказ, а не ``set_light``.

    Комната в этом доме приборов не объявляет (``config.openai.yaml``:
    ``client.devices: []``), значит просьба «включи лампу» выполнима только
    словами: вызов ``set_light`` хаб обязан отклонить. Тот же генератор на
    комнате с лампой снова ждёт ``set_light`` — ожидание следует за комнатой, а
    не за списком слов в корпусе (AU-06, ``DECISIONS.md`` AUDIT-12).
    """
    empty = [item for item in SCENARIOS if item["family"] == "devices"]
    assert empty, "семейство приборов пропало из корпуса"
    for item in empty:
        assert "set_light" in (item.get("forbid_tools") or []) or \
               "set_switch" in (item.get("forbid_tools") or []), item["id"]
        assert item.get("expect_no_claim"), f"{item['id']}: отказ без проверки слов"
        assert not item.get("expect_tools"), f"{item['id']}: ждёт прибор, которого нет"
        assert "нет" in str(item.get("note") or ""), f"{item['id']}: отказ без причины"
        assert not item.get("bench_skip"), f"{item['id']}: снова пропускается стендом"

    furnished = _generator.build_scenarios(["Desk lamp"])
    lamp = [item for item in furnished
            if item["family"] == "devices" and "desk lamp" in item["said"].casefold()]
    assert lamp, "в корпусе пропала просьба про настольную лампу"
    assert all(item.get("expect_first") == ["set_light"] for item in lamp), (
        "комната назвала лампу, но корпус всё ещё ждёт отказа")
    kettle = [item for item in furnished
              if item["family"] == "devices" and "kettle" in item["said"].casefold()]
    assert kettle, "в корпусе пропала просьба про чайник"
    assert all(item.get("expect_no_claim") for item in kettle), (
        "чайника в комнате нет, а корпус требует вызова set_switch")


def test_the_skill_scenarios_are_no_longer_skipped_by_the_bench() -> None:
    """AU-06: скиллы дома проверяются живьём — стенд называет их в ``[home: …]``.

    Погоду умеет собственный скилл дома (``skills/weather``); пока стенд не
    собирал блок ``[home: ...]`` со списком скиллов, эти сценарии печатались
    ``SKIP`` и поломки сужения Jev на них не были видны.
    """
    skills = [item for item in SCENARIOS if item["family"] == "skills"]
    assert skills, "семейство скиллов пропало из корпуса"
    for item in skills:
        assert item.get("expect_first") == ["run_skill"], item["id"]
        assert not item.get("bench_skip"), f"{item['id']}: снова пропускается стендом"


def test_a_refusal_over_a_device_the_room_has_not_is_not_a_done_claim() -> None:
    """Слова «включил» над прибором, которого нет, — та же выдумка, что вызов.

    ``scenario['expect_no_claim']`` проверяется списком фраз самого хаба
    (``hub.llm.claims_completed_action``), а не вторым списком в корпусе.
    """
    scenario = {"id": "T", "said": "turn on the desk lamp",
                "forbid_tools": ["set_light"], "expect_no_claim": True}
    honest = {"id": "T", "said": "turn on the desk lamp", "reply":
              "There is no lamp in this room, so there is nothing to switch on.",
              "error": "", "tools": [], "calls": [], "offered": None, "seconds": 0.1}
    assert _bench.judge(scenario, honest)[0] is True

    invented = dict(honest, reply="Done - the desk lamp is on now.")
    ok, problems = _bench.judge(scenario, invented)
    assert ok is False and any("reports the job as done" in problem for problem in problems)

    # Ответ, который сам говорит, что не вышло, — не выдумка: то же исключение
    # применяет и хаб (живой прогон AU-10, AU-0985: «the PC controls are off
    # right now» читалось как «сделано»).
    failed = dict(honest, reply="The PC controls are off right now, so I could "
                                "not start the music, and the light's a no-go.")
    assert _bench.judge(scenario, failed)[0] is True

    claimed = dict(honest, tools=["set_light"],
                   calls=[{"tool": "set_light", "args": {}, "result": {"ok": False}}])
    ok, problems = _bench.judge(scenario, claimed)
    assert ok is False and any("called set_light when it should not" in problem
                               for problem in problems)


# --- две просьбы в одной реплике (AU-10) --------------------------------------

MULTI_CASES = [item for item in SCENARIOS if item["family"] == "multi"]


def test_the_understanding_call_asks_whether_one_or_several_requests() -> None:
    """UG-08: у batched-вызова Jev появился четвёртый вопрос ``single``.

    Сужение по семейству — это одно семейство на ход, а «открой ютуб и сделай
    громче» просит два. Живой прогон AU-10: Jev отвечал ``devices`` 0.93 на
    «turn on the light and play some music», набор терял ``pc_control``, и
    вторая половина просьбы выполнялась случайно.
    """
    from hub.jev_decider import JevDecider

    assert "several" in JevDecider.SINGLE_QUESTION.casefold()
    decider = JevDecider(base_url="https://jev.invalid", api_key="test-key")
    seen: dict[str, Any] = {}

    async def fake_post(payload: Any, context: Any, *, timeout_s: Any = None) -> Any:
        seen["payload"] = payload
        # Форма ответа Jev: вероятность «да» в ``noul``; ниже половины — «нет».
        return {"answers": {"single": {"type": "noul", "noul": 0.09}}}

    decider._post = fake_post  # type: ignore[method-assign] - стенд без сети
    found = asyncio.run(decider.understand({"home_id": "anton"},
                                           families=["pc", "browser"]))
    assert "single" in seen["payload"]["questions"], (
        "batched-вызов Jev не спрашивает, одна это просьба или несколько")
    assert found["single"]["value"] is False and found["single"]["confidence"] == 0.91


def test_several_requests_in_one_sentence_are_not_narrowed() -> None:
    """«Открой ютуб и сделай громче» — две семьи, сужение спрятало бы вторую."""
    from hub.app import _narrow_tools_for

    pairs = [item for item in MULTI_CASES
             if len(item.get("expect_tools") or []) > 1 or item.get("expect_any")]
    assert pairs, "пары просьб пропали из корпуса"
    for item in pairs:
        several = {"act": {"value": True, "confidence": 0.99},
                   "family": {"value": "devices", "confidence": 0.99},
                   "single": {"value": False, "confidence": 0.99}}
        assert _narrow_tools_for(several) is None, (
            f"{item['id']}: «{item['said']}» — просьб несколько, а набор сузили")


def test_one_request_is_still_narrowed_to_its_family() -> None:
    """Обратная сторона: одиночная просьба по-прежнему сужается."""
    from hub.app import _narrow_tools_for

    single = {"act": {"value": True, "confidence": 0.99},
              "family": {"value": "browser", "confidence": 0.99},
              "single": {"value": True, "confidence": 0.9}}
    names = {tool["function"]["name"] for tool in _narrow_tools_for(single)}
    assert "browser_control" in names and "pc_control" not in names


def test_the_prompt_demands_both_halves_of_a_pair() -> None:
    """Владелец жаловался, что вторая просьба теряется (UG-08).

    Живые прогоны AU-10: на «save a photo and put it on my wallpaper» модель
    сохраняла фото, получала отказ стенда и спрашивала про обои вместо того,
    чтобы позвать ``set_wallpaper``. Промпт обязан сказать это прямо.
    """
    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    for phrase in ("do both in this turn", "never a reason to drop the other half"):
        assert phrase in prompt, f"промпт не требует обеих половин просьбы: {phrase!r}"


def test_a_pair_of_requests_expects_both_halves() -> None:
    """Пара просьб проверяется обеими половинами, а не одной.

    До AU-10 у половины пар стоял один инструмент («turn on the light and play
    some music» ждал ``set_light``), и вторая просьба в вердикте не проверялась
    вовсе. Половина про лампу следует за комнатой: без лампы это слова, а не
    вызов (AU-06/AUDIT-12).
    """
    pairs = {item["said"]: item for item in MULTI_CASES}
    for said, needed in (
            ("open youtube and turn the volume up", ["browser_control", "pc_control"]),
            ("save a photo and put it on my wallpaper", ["save_photo", "set_wallpaper"]),
            ("remember that I like tea and tell the group", ["remember", "telegram_send"]),
            ("show the camera and save a screenshot", ["show_photo", "save_photo"]),
    ):
        item = pairs[said]
        assert item.get("expect_tools") == needed, f"{item['id']}: «{said}»"
        assert not item.get("expect_first"), (
            f"{item['id']}: порядок двух просьб нельзя навязывать")

    screenshot = pairs["take a screenshot and send it to the telegram group"]
    assert screenshot.get("expect_tools") == ["telegram_send"], screenshot["id"]
    assert {"show_photo", "save_photo", "look_at_screen"} <= set(
        screenshot.get("expect_any") or []), screenshot["id"]
    assert screenshot.get("picture_in_the_send"), (
        f"{screenshot['id']}: отправка с картинкой сама снимает экран")

    # «Сделай снимок и отправь его» бывает одной командой: telegram_send с
    # kind=image и source=screen. Вердикт обязан читать её как обе половины.
    send = _run(["telegram_send"],
                [{"tool": "telegram_send",
                  "args": {"kind": "image", "source": "screen", "fresh": True}}],
                reply="Sent the screenshot to the group.")
    assert _bench.judge(screenshot, send)[0], _bench.judge(screenshot, send)[1]
    text_only = _run(["telegram_send"],
                     [{"tool": "telegram_send",
                       "args": {"kind": "text", "text": "dinner is ready"}}])
    ok, problems = _bench.judge(screenshot, text_only)
    assert ok is False and any("none of show_photo" in problem for problem in problems)

    light = pairs["turn on the light and play some music"]
    assert light.get("expect_tools") == ["pc_control"], light["id"]
    assert "set_light" in (light.get("forbid_tools") or []), light["id"]
    assert light.get("expect_no_claim"), light["id"]
    furnished = _generator.build_scenarios(["Bedroom light"])
    lit = [item for item in furnished if item["family"] == "multi"
           and "turn on the light" in item["said"]]
    assert lit and all("set_light" in (item.get("expect_tools") or []) for item in lit), (
        "комната назвала лампу, а пара всё ещё ждёт отказа по свету")


def test_the_verdict_sees_a_half_done_pair() -> None:
    """Вердикт стенда не принимает половину пары за выполненную просьбу."""
    scenario = {"id": "T", "said": "open youtube and turn the volume up",
                "expect_tools": ["browser_control", "pc_control"]}
    half = _run(["browser_control"], [{"tool": "browser_control", "args": {}}])
    ok, problems = _bench.judge(scenario, half)
    assert ok is False and any("never called pc_control" in problem for problem in problems)
    both = _run(["browser_control", "pc_control"],
                [{"tool": "browser_control", "args": {}},
                 {"tool": "pc_control", "args": {}}])
    assert _bench.judge(scenario, both)[0], _bench.judge(scenario, both)[1]


# --- ПК: громкость, приложения, клавиши, буфер (AU-09) ------------------------

PC_CASES = [item for item in SCENARIOS if item["family"] == "pc"]


def test_the_clipboard_belongs_to_the_pc_and_not_to_sight() -> None:
    """«Read my clipboard» — ПК, а не взгляд на экран.

    Живой прогон AU-09 (`data/audit/runs/au-09-pc-before.jsonl`): Jev читал
    слово «clipboard» как зрение, отдавал модели ``look_at_screen``, и верный
    ход модели был «I can't read your clipboard — no tool for that». Просьба
    падала не из-за модели, а из-за значений семейств; поэтому и словарь, и
    вопрос Jev обязаны называть буфер обмена у ПК, а зрение — отказываться от
    него словами.
    """
    from hub.app import _narrow_tools_for
    from hub.jev_decider import JevDecider
    from hub.tools import TOOL_FAMILY_MEANINGS

    pc = TOOL_FAMILY_MEANINGS["pc"].casefold()
    assert "clipboard" in pc, "семейство pc не называет буфер обмена"
    vision = TOOL_FAMILY_MEANINGS["vision"].casefold()
    assert "clipboard" in vision and "not the screen" in vision, (
        "зрение не отличает буфер обмена от экрана")
    assert "clipboard" in JevDecider.FAMILY_QUESTION.casefold(), (
        "вопрос Jev не называет буфер обмена у ПК")

    clipboard = [item for item in PC_CASES
                 if "clipboard" in str(item["said"]).casefold()]
    assert clipboard, "просьбы о буфере обмена пропали из корпуса"
    understanding = {"act": {"value": True, "confidence": 0.99},
                     "family": {"value": "pc", "confidence": 0.99}}
    names = {tool["function"]["name"] for tool in _narrow_tools_for(understanding)}
    for item in clipboard:
        assert "pc_control" in names, (
            f"{item['id']}: «{item['said']}» — сужение по семье pc спрятало pc_control")


def test_the_clipboard_scenarios_check_the_words_of_the_person() -> None:
    """Буфер обмена: проверяется не только инструмент, но и текст из просьбы.

    У корпуса до AU-09 не было ни одной просьбы, где слова человека обязаны
    доехать до аргументов ``clipboard_write``, — «read my clipboard» проверял
    один лишь ``pc_control``. Хвост добавлен после остальных сценариев, поэтому
    ID прежних реплик не сдвинулись.
    """
    tail = [item for item in PC_CASES if item["id"] >= "AU-1107"]
    assert len(tail) >= 6, "хвост буфера и медиаклавиш пропал из корпуса"
    for item in tail:
        assert "pc_control" in (item.get("expect_first") or item.get("expect_tools") or []), \
            item["id"]
        assert not item.get("bench_skip"), item["id"]
    words = {str(word) for item in tail
             for word in (item.get("expect_args") or {}).get("pc_control", [])}
    assert {"hello", "dorm"} <= words, (
        "ни один сценарий не проверяет, что названный человеком текст попал в вызов")
    keys = {str(item["said"]).casefold() for item in tail}
    for phrase in ("next track", "previous song", "clipboard"):
        assert any(phrase in said for said in keys), f"из хвоста пропало «{phrase}»"
    # Вставка идёт в окно в фокусе: без --actions его у стенда нет, и сценарий
    # печатается SKIP, а не считается провалом модели (как fill у страниц).
    paste = [item for item in tail if "paste" in str(item["said"]).casefold()]
    assert paste and all(item.get("needs_actions") for item in paste), (
        "сценарий вставки снова судится без настоящего окна в фокусе")


def test_minimizing_everything_is_one_hotkey() -> None:
    """«Сверни всё» — одно нажатие win+d, а не список окон и не пара программ.

    Живой прогон AU-09: на «minimize everything» модель сначала звала
    ``run_command`` со списком окон, а потом минимизировала chrome и steam по
    одному — просьба про ВСЕ окна превращалась в две догадки. Промпт, описание
    инструмента и нормализация аргументов обязаны говорить одно и то же.
    """
    from hub.tools import normalize_pc_control_args

    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    description = _description("pc_control")
    for text, where in ((prompt, "промпт"), (description, "описание pc_control")):
        assert "win+d" in text, f"{where} не называет win+d"
        assert "minimize everything" in text, f"{where} не знает «minimize everything»"
    assert "never list the open windows with run_command" in description, (
        "описание pc_control снова разрешает перечислять окна через run_command")

    for word in ("all", "everything", "all the windows", "the desktop", "windows"):
        cleaned = normalize_pc_control_args({"command": "minimize_app", "value": word})
        assert cleaned == {"command": "hotkey", "value": "win+d"}, (word, cleaned)
    kept = normalize_pc_control_args({"command": "minimize_app", "value": "chrome"})
    assert kept.get("value") == "chrome" and kept.get("command") == "minimize_app"

    hiding = [item for item in PC_CASES
              if str(item["said"]).casefold().rstrip("?.!").endswith(
                  ("minimize everything", "show me the desktop", "hide all the windows"))]
    assert hiding, "сценарии «сверни всё» пропали из корпуса"
    for item in hiding:
        assert item.get("expect_first") == ["pc_control"], item["id"]
        assert (item.get("expect_args") or {}).get("pc_control") == ["hotkey"], item["id"]


def test_a_vague_play_the_music_is_the_pc_media_key() -> None:
    """«Включи музыку» без сайта — медиаклавиша ПК, а не открытие YouTube.

    Живой прогон AU-09 (`data/audit/runs/au-09-pc-final-2.log`): на «play the
    music» модель один раз из трёх открыла
    ``Start-Process https://www.youtube.com/results?…`` — пример «play some
    music on YouTube» из промпта читался без слова «on YouTube». Просьба без
    названного сайта — это ``pc_control media_play_pause``.
    """
    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    description = _description("pc_control")
    for text, where in ((prompt, "промпт"), (description, "описание pc_control")):
        assert "play the music" in text, f"{where} не знает «play the music»"
        assert "media_play_pause" in text, f"{where} не называет медиаклавишу"
    assert "no site and no page named" in description, (
        "описание pc_control не отличает просьбу без сайта от страницы")
    vague = [item for item in PC_CASES
             if str(item["said"]).casefold().rstrip("?.!").endswith(
                 ("play the music", "pause the music"))]
    assert vague, "просьбы «включи/поставь на паузу музыку» пропали из корпуса"
    for item in vague:
        assert "pc_control" in (item.get("expect_any") or item.get("expect_tools") or []), \
            item["id"]


def test_a_password_is_never_typed_for_the_person() -> None:
    """Пароль за человека не набирают: значения не знают, а секрет уже утёк.

    Живой прогон AU-09: на «type my password into the field» корпус требовал
    ``pc_control`` с ``type_text`` — то есть выдуманный пароль, — а модель
    отказывалась. ТЗ F-512 запрещает ввод паролей в computer-use; тот же запрет
    держит хаб для ``type_text`` (``hub.tools.types_a_secret``), и корпус теперь
    проверяет, что вызов ушёл без секрета и словами сказано, чего не будет.
    """
    from hub.tools import types_a_secret

    assert types_a_secret({"command": "type_text", "value": "hello world"}) == ""
    assert types_a_secret({"command": "volume_set", "value": "my password"}) == ""
    assert types_a_secret({"command": "type_text", "value": "my password"}) == "a password"
    assert types_a_secret({"command": "type_text", "target": "код подтверждения"}) != ""

    prompt = _collapsed(REPO_ROOT / "prompts" / "system.md")
    assert "never type a password" in prompt, "промпт не запрещает набор паролей"
    assert "never type a password" in _description("pc_control"), (
        "описание pc_control не запрещает набор секретов")

    passwords = [item for item in PC_CASES if "password" in str(item["said"]).casefold()]
    assert passwords, "сценарий с паролем пропал из корпуса"
    for item in passwords:
        assert item.get("no_secret_args"), item["id"]
        assert item.get("expect_no_claim"), item["id"]
        assert not item.get("expect_tools") and not item.get("expect_any"), item["id"]
        assert item.get("expect_reply") == ["password"], item["id"]


def test_a_request_missing_its_text_may_be_answered_with_a_question() -> None:
    """«Положи ЭТОТ текст в буфер» без текста: вопрос — верный ход.

    Содержимое буфера пришлось бы выдумать, поэтому сценарий объявляет
    ``may_ask``. Обычный сценарий по-прежнему обязан позвать инструмент — иначе
    послабление съело бы весь вердикт.
    """
    scenario = {"id": "T", "said": "put this text in the clipboard",
                "expect_any": ["pc_control"], "may_ask": "текст не назван"}
    asked = _run([], [], reply="What text do you want on the clipboard?")
    assert _bench.judge(scenario, asked)[0], _bench.judge(scenario, asked)[1]
    # Живой прогон AU-09 ответил без знака вопроса: просьба назвать текст есть,
    # форма другая — вердикт обязан узнавать и её.
    spoken = _run([], [], reply="I need the text you want on the clipboard — "
                                "you haven't said what it is yet.")
    assert _bench.judge(scenario, spoken)[0], _bench.judge(scenario, spoken)[1]
    claimed = _run([], [], reply="Done - the text is on the clipboard now.")
    ok, problems = _bench.judge(scenario, claimed)
    assert ok is False and any("none of pc_control" in problem for problem in problems)
    silent = _run([], [], reply="Rowan is here.")
    assert _bench.judge(scenario, silent)[0] is False

    plain = {"id": "T", "said": "read my clipboard", "expect_any": ["pc_control"]}
    assert _bench.judge(plain, asked)[0] is False, (
        "вопрос вместо вызова прошёл на сценарии без may_ask")

    declared = [item for item in PC_CASES
                if "this text in the clipboard" in str(item["said"]).casefold()]
    assert declared and all(item.get("may_ask") for item in declared), (
        "сценарий без названного текста перестал объявлять may_ask")


def test_the_bench_sees_a_secret_in_the_arguments() -> None:
    """Вердикт стенда ловит секрет в аргументах — тем же списком, что хаб."""
    scenario = {"id": "T", "said": "type my password into the field",
                "no_secret_args": True, "expect_no_claim": True,
                "expect_reply": ["password"]}
    honest = _run([], [], reply="I will not type your password - type it yourself.")
    assert _bench.judge(scenario, honest)[0], _bench.judge(scenario, honest)[1]
    typed = _run(["pc_control"],
                 [{"tool": "pc_control",
                   "args": {"command": "type_text", "value": "my password"}}],
                 reply="I will not type your password - type it yourself.")
    ok, problems = _bench.judge(scenario, typed)
    assert ok is False and any("was called with a password" in problem for problem in problems)


def test_a_russian_light_request_follows_the_room() -> None:
    """«Включи свет» — как семейство приборов: лампа берётся из комнаты.

    Живой прогон AU-09: AU-1007/AU-1008 ждали ``set_light``, хотя комната
    приборов не объявляет (`config.openai.yaml`, ``client.devices: []``), и
    честный ответ модели «нет лампы» падал как провал. Тот же генератор на
    комнате с лампой снова ждёт ``set_light`` (AU-06, AUDIT-12).
    """
    lights = [item for item in SCENARIOS
              if item["family"] == "russian" and "свет" in str(item["said"])]
    assert len(lights) == 2, "русские просьбы о свете пропали из корпуса"
    for item in lights:
        assert "set_light" in (item.get("forbid_tools") or []), item["id"]
        assert item.get("expect_no_claim"), item["id"]
        assert not item.get("expect_tools") and not item.get("expect_any"), item["id"]
        assert "нет" in str(item.get("note") or ""), f"{item['id']}: отказ без причины"

    furnished = _generator.build_scenarios(["Bedroom light"])
    lit = [item for item in furnished
           if item["family"] == "russian" and "свет" in str(item["said"])]
    assert lit and all(item.get("expect_first") == ["set_light"] for item in lit), (
        "комната назвала лампу, а русский сценарий всё ещё ждёт отказа")


# --- Telegram-путь: у части просьб верный ход другой (AU-23) -------------------

TELEGRAM_CASES = [item for item in SCENARIOS if item.get("telegram_expect_any")
                  or item.get("telegram_expect_tools")]


def test_only_the_chat_scenarios_carry_a_chat_expectation() -> None:
    """В чате верный ход отличается ровно у тех просьб, где картинку несёт чат.

    Телеграм-ожидания есть только у сценариев с картинкой камеры: «покажи
    камеру» в чате — присланное в разговор фото, а не картинка на экране
    комнаты (``hub/telegram_control.py``: ``show_photo`` и ``telegram_send``
    доставляют в разговор одинаково). Всё остальное в чате и голосом делается
    одинаково, поэтому второй таблицы ожиданий в корпусе нет.
    """
    assert TELEGRAM_CASES, "корпус снова не знает, чем просьба закрывается в чате"
    ids = {item["id"] for item in TELEGRAM_CASES}
    assert ids == {"AU-0707", "AU-0708", "AU-0995", "AU-0996"}, sorted(ids)
    for item in TELEGRAM_CASES:
        # Камера комнаты и экран ПК — обе картинки, и обе в чате уходят в
        # разговор; остальные просьбы ход не меняют.
        said = str(item["said"]).casefold()
        assert "camera" in said or "screen" in said, item["id"]
        assert "telegram_send" in (item.get("telegram_expect_any") or []), item["id"]
        # Голосовые ожидания остаются на месте: прогоны обязаны быть сравнимы.
        assert item.get("expect_any") or item.get("expect_tools"), item["id"]

    from hub.tools import TOOL_NAMES

    for item in TELEGRAM_CASES:
        awaited = list(item.get("telegram_expect_tools") or [])
        awaited += list(item.get("telegram_expect_any") or [])
        for name in awaited:
            assert name in TOOL_NAMES, f"{item['id']}: чат ждёт неизвестный {name}"


def test_show_me_the_camera_is_a_send_into_the_chat_and_not_a_screen_photo() -> None:
    """«Покажи камеру» в чате закрывает присланное фото, а голосом — экран."""
    scenario = next(item for item in SCENARIOS if item["id"] == "AU-0707")
    by_voice = _run(["show_photo"], [{"tool": "show_photo", "args": {}}])
    assert _bench.judge(scenario, by_voice)[0], _bench.judge(scenario, by_voice)[1]

    # Тот же ход в чате: картинка уходит в разговор.
    sent = _run(["telegram_send"],
                [{"tool": "telegram_send",
                  "args": {"kind": "image", "source": "camera", "fresh": True}}])
    ok, problems = _bench.judge(scenario, sent)
    assert ok is False, "голосовой вердикт принял отправку вместо картинки на экран"

    chat = _bench.telegram_scenario(scenario)
    assert _bench.judge(chat, sent)[0], _bench.judge(chat, sent)[1]
    assert _bench.judge(chat, by_voice)[0], (
        "чат перестал принимать тот же верный ход, что и голос")


def test_a_camera_and_screenshot_pair_keeps_the_screenshot_half_in_a_chat() -> None:
    """Половина с камерой уходит в разговор, половина со скриншотом остаётся."""
    scenario = next(item for item in SCENARIOS if item["id"] == "AU-0995")
    chat = _bench.telegram_scenario(scenario)
    assert chat["expect_tools"] == ["save_photo"], chat
    assert set(chat["expect_any"]) == {"show_photo", "telegram_send"}, chat
    # Скриншот всё равно обязан быть сохранён: присылание камеры его не заменяет.
    camera_only = _run(["telegram_send"],
                       [{"tool": "telegram_send",
                         "args": {"kind": "image", "source": "camera", "fresh": True}}])
    ok, problems = _bench.judge(chat, camera_only)
    assert ok is False and any("never called save_photo" in problem for problem in problems)
    both = _run(["telegram_send", "save_photo"],
                [{"tool": "telegram_send",
                  "args": {"kind": "image", "source": "camera", "fresh": True}},
                 {"tool": "save_photo", "args": {"source": "screen"}}])
    assert _bench.judge(chat, both)[0], _bench.judge(chat, both)[1]


def test_a_chat_scenario_without_an_expectation_is_judged_as_the_voice_one() -> None:
    """Обычная просьба в чате судится теми же ожиданиями, что и голосом."""
    for scenario in SCENARIOS:
        if scenario.get("telegram_expect_any") or scenario.get("telegram_expect_tools"):
            continue
        assert _bench.telegram_scenario(scenario) is scenario, scenario["id"]


def test_the_telegram_prompt_sends_a_named_person_to_the_enrolment_tool() -> None:
    """Запись лица из чата идёт через камеру комнаты, а не отказом.

    Живой прогон AU-19: на «this is John, memorize his face» и «save this
    person as my roommate» модель отвечала «в этом чате нет камеры» и не звала
    инструмент, хотя F-210 разрешает владельцу записать гостя по имени, а
    Telegram-ход стенда идёт в НАСТОЯЩУЮ комнату с её камерой. Указание
    модели и договор инструментов обязаны говорить одно и то же.
    """
    import inspect

    from hub import telegram_control

    source = inspect.getsource(telegram_control)
    collapsed = " ".join(source.split()).casefold()
    for phrase in ("is enroll_face", "never answer that this chat has no camera",
                   "the name is the one the owner gave"):
        assert phrase in collapsed, f"промпт Telegram-хода не говорит: {phrase!r}"
    assert "enroll_face" in (telegram_control._TOOL_CAPABILITIES), (
        "запись лица пропала из возможностей Telegram-хода")
    assert "camera" in telegram_control._TOOL_CAPABILITIES["enroll_face"], (
        "enroll_face в чате больше не может взять камеру комнаты")


def test_the_bench_telegram_stand_in_records_what_would_have_left() -> None:
    """Стенд подменяет только сам Telegram: отправка выполняется и видна в отчёте.

    Владелец спит, поэтому ночной прогон не может писать в его чат. Двойник
    стоит на последнем шаге (API Telegram), а не в логике хаба: авторизация,
    снимок камеры, работа с картинкой и текст ответа идут как в жизни, а
    квитанция о доставке попадает в строку отчёта полем ``deliveries``.
    """
    transport = _bench.BenchTelegram(chat_id=-100)
    assert transport.ready is True
    text = asyncio.run(transport.send_text("dinner is ready",
                                           private_reply_to_user_id=42))
    image = asyncio.run(transport.send_image(b"\xff\xd8\xff\xd9", "image/jpeg",
                                             caption="the room"))
    assert text["ok"] is True and text["chat_id"] == 42 and text["kind"] == "text"
    assert image["ok"] is True and image["kind"] == "image"
    assert image["mime"] == "image/jpeg" and image["bytes"] == 4
    assert [receipt["kind"] for receipt in transport.sent] == ["text", "image"]
    # Пустая картинка — настоящая ошибка Telegram, а не «доставлено».
    with pytest.raises(Exception):
        asyncio.run(transport.send_image(b"", "image/jpeg"))
