"""client/camera.py: FP16 goes out under the name this Ultralytics knows.

Ultralytics 8.4 renamed the ``half`` argument to ``quantize``. The old name
still works, but every prediction printed

    WARNING 'half' is deprecated and will be removed in the future. Use
    'quantize' instead.

on the room PC, and its console file was 482 lines of exactly that warning and
nothing else - the owner could not read a single line of the client's own log.
"""
import sys
import types

from client import camera


def _fake_ultralytics(monkeypatch, overrides):
    module = types.ModuleType('ultralytics.cfg')
    module.DEFAULT_CFG_DICT = overrides
    monkeypatch.setitem(sys.modules, 'ultralytics', types.ModuleType('ultralytics'))
    monkeypatch.setitem(sys.modules, 'ultralytics.cfg', module)
    camera._fp16_arg_name.cache_clear()


def test_new_build_gets_quantize(monkeypatch):
    _fake_ultralytics(monkeypatch, {'quantize': None, 'imgsz': 640})
    try:
        assert camera.precision_kwargs(True) == {'quantize': 16}
        # False must CLEAR the precision, which the new name spells None.
        assert camera.precision_kwargs(False) == {'quantize': None}
    finally:
        camera._fp16_arg_name.cache_clear()


def test_older_build_keeps_half(monkeypatch):
    _fake_ultralytics(monkeypatch, {'half': False, 'imgsz': 640})
    try:
        assert camera.precision_kwargs(True) == {'half': True}
        assert camera.precision_kwargs(False) == {'half': False}
    finally:
        camera._fp16_arg_name.cache_clear()


def test_a_client_without_ultralytics_still_starts(monkeypatch):
    monkeypatch.setitem(sys.modules, 'ultralytics.cfg', None)
    camera._fp16_arg_name.cache_clear()
    try:
        assert camera.precision_kwargs(True) == {'half': True}
    finally:
        camera._fp16_arg_name.cache_clear()
