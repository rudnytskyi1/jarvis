"""Photo replacement and Win32 ordering without touching real windows."""
import base64
import ctypes
import os
import weakref
from ctypes import wintypes
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from client import viewer
from client.actions import photos


def native_window_api(monkeypatch):
    api = SimpleNamespace(
        FindWindowW=Mock(return_value=2**40 + 123),
        IsWindow=Mock(return_value=1),
        GetWindowLongW=Mock(return_value=viewer._WS_CAPTION | viewer._WS_THICKFRAME),
        SetWindowLongW=Mock(return_value=1),
        SetWindowPos=Mock(return_value=1),
        IsIconic=Mock(return_value=True),
        ShowWindow=Mock(return_value=1),
        SetForegroundWindow=Mock(return_value=1),
    )
    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=api), raising=False)
    return api


def test_new_photo_restores_own_window_and_raises_it_above_older_windows(monkeypatch):
    api = native_window_api(monkeypatch)
    viewer._make_borderless_topmost(viewer.WINDOW_NAME)
    assert str(os.getpid()) in viewer.WINDOW_NAME
    api.FindWindowW.assert_called_once_with(None, viewer.WINDOW_NAME)
    assert api.FindWindowW.restype is wintypes.HWND
    assert api.SetWindowPos.argtypes[0] is wintypes.HWND
    hwnd = 2**40 + 123
    api.ShowWindow.assert_called_once_with(hwnd, viewer._SW_RESTORE)
    api.SetForegroundWindow.assert_called_once_with(hwnd)
    position = api.SetWindowPos.call_args.args
    assert position[:2] == (hwnd, viewer._HWND_TOPMOST)
    assert position[-1] & viewer._SWP_SHOWWINDOW


def test_passive_topmost_refresh_does_not_keep_stealing_focus(monkeypatch):
    api = native_window_api(monkeypatch)
    viewer._keep_on_top(viewer.WINDOW_NAME)
    assert api.SetWindowPos.call_args.args[-1] & viewer._SWP_NOACTIVATE
    api.SetForegroundWindow.assert_not_called()
    api.ShowWindow.assert_not_called()


def test_missing_window_is_ignored(monkeypatch):
    api = native_window_api(monkeypatch)
    api.FindWindowW.return_value = None
    viewer._make_borderless_topmost(viewer.WINDOW_NAME)
    viewer._keep_on_top(viewer.WINDOW_NAME)
    api.SetWindowPos.assert_not_called()
    api.SetForegroundWindow.assert_not_called()


def test_the_photo_goes_right_below_the_hud_instead_of_fighting_it(monkeypatch):
    """Фото и оверлей не должны по очереди захватывать верх — это и есть мигание."""
    api = native_window_api(monkeypatch)
    hud = 2**40 + 777
    viewer.set_overlay_window(lambda: hud)
    try:
        viewer._keep_on_top(viewer.WINDOW_NAME)
    finally:
        viewer.set_overlay_window(None)
    position = api.SetWindowPos.call_args.args
    assert position[1] == hud
    assert position[-1] & viewer._SWP_NOACTIVATE
    api.SetForegroundWindow.assert_not_called()


def test_a_closed_hud_falls_back_to_a_plain_topmost_photo(monkeypatch):
    api = native_window_api(monkeypatch)
    api.IsWindow.return_value = 0
    viewer.set_overlay_window(lambda: 4242)
    try:
        viewer._keep_on_top(viewer.WINDOW_NAME)
    finally:
        viewer.set_overlay_window(None)
    assert api.SetWindowPos.call_args.args[1] == viewer._HWND_TOPMOST


def test_a_broken_overlay_provider_still_shows_the_photo(monkeypatch):
    api = native_window_api(monkeypatch)

    def boom():
        raise RuntimeError('the Qt thread is gone')

    viewer.set_overlay_window(boom)
    try:
        viewer._keep_on_top(viewer.WINDOW_NAME)
    finally:
        viewer.set_overlay_window(None)
    assert api.SetWindowPos.call_args.args[1] == viewer._HWND_TOPMOST


def test_without_an_overlay_the_photo_keeps_the_old_behaviour(monkeypatch):
    api = native_window_api(monkeypatch)
    viewer.set_overlay_window(None)
    viewer._keep_on_top(viewer.WINDOW_NAME)
    assert api.SetWindowPos.call_args.args[1] == viewer._HWND_TOPMOST


def test_backlogged_photos_show_only_latest_in_reused_window(monkeypatch):
    photo_viewer = viewer.ImageViewer()
    native = SimpleNamespace(WINDOW_NORMAL=0, WND_PROP_FULLSCREEN=0, WINDOW_FULLSCREEN=1,
                             namedWindow=Mock(), setWindowProperty=Mock(), waitKey=Mock(),
                             destroyAllWindows=Mock())
    native.imshow = Mock(side_effect=lambda *_: photo_viewer._queue.put(viewer._STOP))
    photo_viewer._cv2 = native
    photo_viewer._decode = Mock(side_effect=lambda _, data: data)
    monkeypatch.setattr(viewer, '_fit_to_screen', lambda _, frame: frame)
    monkeypatch.setattr(viewer, '_make_borderless_topmost', Mock())
    monkeypatch.setattr(viewer, '_keep_on_top', Mock())
    photo_viewer._queue.put(viewer._ShowRequest(b'old photo', 'old', 60))
    photo_viewer._queue.put(viewer._ShowRequest(b'new photo', 'new', 60))
    photo_viewer._run()
    native.imshow.assert_called_once_with(viewer.WINDOW_NAME, b'new photo')
    photo_viewer._decode.assert_called_once_with(native, b'new photo')
    native.namedWindow.assert_called_once_with(viewer.WINDOW_NAME, native.WINDOW_NORMAL)
    viewer._make_borderless_topmost.assert_called_once_with(viewer.WINDOW_NAME)


def test_external_photo_dismisses_registered_viewer_through_its_queue(monkeypatch):
    active = viewer.ImageViewer()
    active.hide = Mock()
    monkeypatch.setattr(viewer, '_viewers', weakref.WeakSet([active]))
    viewer.dismiss_for_external_photo()
    active.hide.assert_called_once_with(wait=True)


def test_saving_and_opening_photo_dismisses_old_rowan_image_before_windows_open(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new('RGB', (32, 24), 'blue').save(output, 'JPEG')
    calls = []
    monkeypatch.setattr(viewer, 'dismiss_for_external_photo', lambda: calls.append('hide old Rowan image'))
    result = photos.save_photo({'jpeg_base64': base64.b64encode(output.getvalue()).decode(), 'open': True},
        desktop=tmp_path, opener=lambda _: calls.append('open saved photo'))
    assert calls == ['hide old Rowan image', 'open saved photo']
    assert result['saved'] is True and result['opened'] is True


def test_save_without_open_preserves_current_display(tmp_path, monkeypatch):
    output = BytesIO()
    Image.new('RGB', (32, 24), 'blue').save(output, 'JPEG')
    hide = Mock()
    opener = Mock()
    monkeypatch.setattr(viewer, 'dismiss_for_external_photo', hide)
    result = photos.save_photo({'jpeg_base64': base64.b64encode(output.getvalue()).decode(), 'open': False},
                              desktop=tmp_path, opener=opener)
    hide.assert_not_called()
    opener.assert_not_called()
    assert result['saved'] is True and result['opened'] is False
