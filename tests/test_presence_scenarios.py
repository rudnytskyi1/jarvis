"""ТЗ сценарии 2 и 4 (фаза 2): приветствие по имени и фото незнакомца.

Сценарий 2: друг заходит в комнату СТОЯ СПИНОЙ к камере. Пока лица нет, имени
нет — значит и приветствия быть не должно (и уж точно не «здравствуйте,
незнакомец»). Когда он поворачивается, лицо опознаётся, и приветствие звучит
ПО ИМЕНИ.

Сценарий 4: незнакомец в комнате, владельца там нет — фотография уходит в
Telegram, и уходит она вовремя (ТЗ 15.3: незнакомец → Telegram ≤ 5 с).

Оба сценария идут по настоящему коду хаба: `Connection._match_presence`,
настоящие `RoomState`/`PresenceTracker`, настоящая БД хаба (f-204/F-301
пишут в неё), настоящий `PresenceAlerts` с настоящим рабочим циклом. Стенд
замеров и подставные движки живут в `scripts/measure_presence_latency.py`, и
эти тесты гоняют ровно его — то, что меряют, и то, что проверяют, не может
разойтись.
"""
from __future__ import annotations

import asyncio
import json

from hub import app as hub_app
from scripts import measure_presence_latency as presence_mod

BODY = presence_mod.BODY
MAX = presence_mod.FRIEND


def test_a_back_to_the_camera_is_neither_a_stranger_nor_a_name(monkeypatch):
    """Сценарий 2: пока лица нет, Rowan молчит."""
    engine = presence_mod.Engine({presence_mod.FRIEND_JPEG: [presence_mod.photo(
        presence_mod.FRIEND_FACE)]}, {presence_mod.FRIEND_JPEG: MAX})
    with presence_mod.room(engine) as (connection, socket):
        async def run():
            await connection._match_presence([presence_mod.frame(presence_mod.BACK_JPEG,
                                                                tracks=[BODY])])
            assert set(connection.presence.present()) == set(), "спина — не человек с именем"
            assert connection.presence.has_fresh_unknown_face() is False, (
                "спина — не незнакомец: лица-то не было")
            assert connection._may_greet(*connection._greet_config()) is None
            await presence_mod.silent_for(connection, 0.4)

        asyncio.run(run())
        assert socket.said() == [], "не о чем было говорить"


def test_the_friend_is_greeted_by_name_when_he_turns_around(monkeypatch):
    """Сценарий 2: тот же трек, лицо появилось — и приветствие по имени."""
    engine = presence_mod.Engine({presence_mod.FRIEND_JPEG: [presence_mod.photo(
        presence_mod.FRIEND_FACE)]}, {presence_mod.FRIEND_JPEG: MAX})
    with presence_mod.room(engine) as (connection, socket):
        async def run() -> float:
            await connection._match_presence([presence_mod.frame(presence_mod.BACK_JPEG,
                                                                tracks=[BODY])])
            assert socket.said() == [], "спиной он ещё никто"
            await connection._match_presence([presence_mod.frame(presence_mod.FRIEND_JPEG,
                                                                tracks=[BODY])])
            # Личность удержана на ТОМ ЖЕ треке (F-204), а не заведена заново.
            assert connection.room.tracks[BODY["id"]]["name"] == MAX
            return await presence_mod.speak_greeting(connection, socket)

        waited = asyncio.run(run())

    said = socket.said()
    assert len(said) == 1, f"одно приветствие, а не {len(said)}"
    assert MAX in said[0], f"приветствие по имени, а не {said[0]!r}"
    assert waited < presence_mod.GREETING_BUDGET_S, (
        f"приветствие заняло {waited:.3f} с против бюджета "
        f"{presence_mod.GREETING_BUDGET_S} с")


def test_the_greeting_measurement_names_the_friend_and_keeps_the_budget():
    report = presence_mod.measure_greeting(repeats=2)
    assert report["named"] is True
    assert all(MAX in line for line in report["said"])
    assert report["waits_s"]["count"] == 2
    assert report["within_budget"] is True
    assert report["waits_s"]["max_s"] < presence_mod.GREETING_BUDGET_S


def test_an_unknown_face_reaches_the_owner_in_telegram_with_a_photo():
    """Сценарий 4: настоящий PresenceAlerts, подставной только транспорт."""
    report = presence_mod.measure_stranger_alert(repeats=1)
    assert report["delivered"] is True
    delivery = report["deliveries"][0]
    assert delivery["kind"] == "image"
    assert delivery["chat_id"] == presence_mod.OWNER, "фото уходит владельцу"
    assert "неопознанный человек" in delivery["caption"]
    assert delivery["bytes"] > 0
    assert report["within_budget"] is True


def test_the_alert_still_needs_the_stranger_to_stay_in_frame():
    """Короткий кадр — не тревога: правило ТЗ требует стабильности."""
    report = presence_mod.measure_stranger_alert(repeats=1, min_stable_s=0.0)
    assert report["delivered"] is True, "нулевая стабильность доставляет сразу"
    quick = report["waits_s"]["max_s"]
    slow = presence_mod.measure_stranger_alert(repeats=1, min_stable_s=2.0)
    assert slow["waits_s"]["max_s"] > quick, (
        "требование «человек стабильно в кадре» должно быть видно в замере")
    assert slow["within_budget"] is True


def test_the_command_prints_both_budgets_and_fails_over_budget(capsys):
    assert presence_mod.main(["--repeats", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["greeting"]["budget_s"] == 2.0
    assert report["stranger_alert"]["budget_s"] == 5.0
    assert report["within_budget"] is True

    # An honest failure: 5.5 s of required stability cannot fit the 5 s budget.
    assert presence_mod.main(["--repeats", "1", "--min-stable-s", "5.5"]) == 1
    over = json.loads(capsys.readouterr().out)
    assert over["stranger_alert"]["within_budget"] is False
    assert over["within_budget"] is False


def test_the_hub_globals_are_put_back_after_a_measurement():
    """A measurement must not leave the hub pointed at a deleted database."""
    before = (hub_app._hub_conn, hub_app._face, hub_app._hub_db_path)
    presence_mod.measure_greeting(repeats=1)
    assert (hub_app._hub_conn, hub_app._face, hub_app._hub_db_path) == before
