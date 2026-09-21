"""Desktop image transport and Win32 verification without changing a desktop."""

import asyncio
import base64
import ctypes
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from client.actions import wallpaper
from client.actions.dispatcher import TOOL_SET_WALLPAPER, Dispatcher


def image_bytes(image_format='PNG'):
    output = BytesIO()
    Image.new('RGB', (24, 16), color=(30, 60, 90)).save(output, format=image_format)
    return output.getvalue()


class FakeSystemParameters:
    def __init__(self, *, set_ok=True, get_ok=True, reported=None):
        self.calls = []
        self.set_ok = set_ok
        self.get_ok = get_ok
        self.reported = reported
        self.path = None

    def __call__(self, action, parameter, pointer, flags):
        self.calls.append((action, parameter, flags))
        if action == wallpaper.SPI_SETDESKWALLPAPER:
            self.path = ctypes.wstring_at(pointer)
            return self.set_ok
        assert action == wallpaper.SPI_GETDESKWALLPAPER
        value = self.path if self.reported is None else self.reported
        buffer = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_wchar))
        for index, character in enumerate(value + '\0'):
            buffer[index] = character
        return self.get_ok


def fake_user32(**kwargs):
    return SimpleNamespace(SystemParametersInfoW=FakeSystemParameters(**kwargs))


@pytest.mark.parametrize('image_format,extension', [('PNG', '.png'), ('JPEG', '.jpg')])
def test_preserves_image_bytes_and_applies_unicode_local_file(tmp_path, image_format, extension):
    data = image_bytes(image_format)
    native = fake_user32()
    result = wallpaper.set_wallpaper(
        {'image_base64': base64.b64encode(data).decode()},
        folder=tmp_path / 'Роуэн фото', user32=native,
    )
    path = Path(result['path'])
    assert path.read_bytes() == data
    assert path.suffix == extension
    assert path.parent == tmp_path / 'Роуэн фото'
    assert result['applied'] is True and result['verified'] is True
    spi = native.SystemParametersInfoW
    assert spi.path == str(path)
    assert spi.calls == [(0x14, 0, 3), (0x73, 32768, 0)]
    assert spi.argtypes[2] is ctypes.c_void_p
    assert spi.restype is wallpaper.wintypes.BOOL


def test_webp_is_converted_to_lossless_png(tmp_path):
    native = fake_user32()
    result = wallpaper.set_wallpaper(
        {'image_base64': base64.b64encode(image_bytes('WEBP')).decode()},
        folder=tmp_path, user32=native,
    )
    assert result['format'] == 'PNG'
    with Image.open(result['path']) as image:
        assert image.format == 'PNG'
        assert image.size == (24, 16)


@pytest.mark.parametrize('args', [
    {'path': 'C:/brain-only/generated.png'},
    {'image_base64': 'not base64'},
    {'image_base64': base64.b64encode(b'not an image').decode()},
    {'image_base64': ['not', 'text']},
])
def test_invalid_or_missing_image_never_touches_windows(tmp_path, args):
    native = fake_user32()
    with pytest.raises((ValueError, OSError)):
        wallpaper.set_wallpaper(args, folder=tmp_path, user32=native)
    assert native.SystemParametersInfoW.calls == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('options,expected_calls', [
    ({'set_ok': False}, 1),
    ({'get_ok': False}, 2),
    ({'reported': 'C:/old wallpaper.png'}, 2),
    ({'reported': ''}, 2),
])
def test_windows_failure_and_readback_mismatch_are_not_success(tmp_path, options, expected_calls):
    native = fake_user32(**options)
    with pytest.raises(OSError):
        wallpaper.set_wallpaper(
            {'image_base64': base64.b64encode(image_bytes()).decode()},
            folder=tmp_path, user32=native,
        )
    assert len(native.SystemParametersInfoW.calls) == expected_calls


def test_missing_local_file_never_calls_windows(tmp_path):
    native = fake_user32()
    with pytest.raises(FileNotFoundError):
        wallpaper.apply_windows_wallpaper(tmp_path / 'missing.png', user32=native)
    assert native.SystemParametersInfoW.calls == []


def test_repeated_image_uses_same_persistent_file(tmp_path):
    args = {'image_base64': base64.b64encode(image_bytes()).decode()}
    first = wallpaper.set_wallpaper(args, folder=tmp_path, user32=fake_user32())
    second = wallpaper.set_wallpaper(args, folder=tmp_path, user32=fake_user32())
    assert first['path'] == second['path']
    assert list(tmp_path.iterdir()) == [Path(first['path'])]


@pytest.mark.parametrize('set_ok', [True, False])
def test_dispatcher_reports_verified_action_and_redacts_image(monkeypatch, tmp_path, caplog, set_ok):
    native = fake_user32(set_ok=set_ok)
    apply = wallpaper.apply_windows_wallpaper
    monkeypatch.setattr(wallpaper, 'wallpaper_folder', lambda: tmp_path)
    monkeypatch.setattr(wallpaper, 'apply_windows_wallpaper', lambda path, **kwargs: apply(path, user32=native))
    encoded = base64.b64encode(image_bytes()).decode()
    with caplog.at_level('INFO'):
        ok, error, output = asyncio.run(Dispatcher({}, None).execute({
            'id': 'wallpaper-1', 'tool': TOOL_SET_WALLPAPER,
            'args': {'image_base64': encoded},
        }))
    assert encoded not in caplog.text
    assert '<image omitted>' in caplog.text
    if set_ok:
        assert ok and error is None
        assert json.loads(output)['verified'] is True
    else:
        assert not ok and 'Windows refused' in error
        assert output is None
