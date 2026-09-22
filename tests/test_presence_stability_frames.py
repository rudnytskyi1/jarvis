"""Присутствие считается по кадрам, а не по секундам (владелец, 22.09).

«Я только что быстро перед камерой прошёл и из-за stable presence 1 секунда
оно не заметило меня. Может stable presence с секунд надо заменить на кадры
(например 2 кадра в секунду)».
"""
from __future__ import annotations

import asyncio

import pytest

from hub.presence_alerts import validate_rule
from tests.test_presence_alerts import engine, provider


def test_frames_are_the_default_gate_and_seconds_are_off():
    rule = validate_rule({'enabled': True})
    assert rule['min_frames'] == 2
    assert rule['min_stable_s'] == 0
    assert validate_rule({'min_frames': 5})['min_frames'] == 5
    with pytest.raises(ValueError):
        validate_rule({'min_frames': 0})
    with pytest.raises(ValueError):
        validate_rule({'min_frames': 1.5})


def test_a_quick_pass_before_the_camera_alerts(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])

    async def run():
        api = provider()
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'cooldown_s': 10})
        alerts.start()
        # Полсекунды перед камерой: секунда стабильности это проглатывала.
        for stamp in (1000., 1000.5):
            clock[0] = stamp
            assert alerts.observe(persons=1, jpeg=b'jpeg')
            await alerts.drain()
        api.send_image.assert_awaited_once()
        await alerts.close()
    asyncio.run(run())


def test_one_frame_is_still_not_presence(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])

    async def run():
        api = provider()
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'cooldown_s': 10})
        alerts.start()
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        api.send_image.assert_not_awaited()
        await alerts.close()
    asyncio.run(run())


def test_seconds_can_still_be_demanded_on_top_of_frames(tmp_path, monkeypatch):
    clock = [1000.]
    monkeypatch.setattr('hub.presence_alerts.time.time', lambda: clock[0])

    async def run():
        api = provider()
        alerts = engine(tmp_path, api)
        alerts.save_rule({'enabled': True, 'min_stable_s': 2, 'cooldown_s': 10})
        alerts.start()
        for stamp in (1000., 1000.5, 1001.):
            clock[0] = stamp
            alerts.observe(persons=1, jpeg=b'jpeg')
            await alerts.drain()
        # Кадры уже есть, но правило дополнительно требует две секунды.
        api.send_image.assert_not_awaited()
        clock[0] = 1002.5
        alerts.observe(persons=1, jpeg=b'jpeg')
        await alerts.drain()
        api.send_image.assert_awaited_once()
        await alerts.close()
    asyncio.run(run())
