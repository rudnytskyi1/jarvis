from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from client.camera import CameraService, CameraUnavailable


def test_rtsp_uses_native_resolution_and_bounded_reads():
    camera = CameraService(SimpleNamespace(stream_url='rtsp://name:secret@192.0.2.1:554/main'))
    capture = Mock()
    capture.isOpened.return_value = True
    cv = Mock(CAP_FFMPEG=1900, CAP_PROP_OPEN_TIMEOUT_MSEC=53, CAP_PROP_READ_TIMEOUT_MSEC=54)
    cv.VideoCapture.return_value = capture
    assert camera._open_capture(cv) is capture
    assert cv.VideoCapture.call_args.args[2] == [53, 4000, 54, 2500]
    capture.set.assert_not_called()


def test_network_error_does_not_expose_camera_password():
    camera = CameraService(SimpleNamespace(stream_url='rtsp://name:secret@192.0.2.1:554/main'))
    cv = Mock(CAP_FFMPEG=1900, CAP_PROP_OPEN_TIMEOUT_MSEC=53, CAP_PROP_READ_TIMEOUT_MSEC=54)
    cv.VideoCapture.side_effect = RuntimeError(camera.stream_url)
    with pytest.raises(CameraUnavailable) as error:
        camera._open_capture(cv)
    assert 'secret' not in str(error.value)


def test_usb_can_request_4k():
    camera = CameraService(SimpleNamespace(width=3840, height=2160))
    cv = Mock(CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4)
    capture = Mock()
    capture.get.return_value = 3840
    camera._request_resolution(cv, capture)
    assert capture.set.call_args_list[0].args == (3, 3840)
    assert capture.set.call_args_list[1].args == (4, 2160)


def test_missing_opencv_does_not_enter_network_reconnect_loop():
    camera = CameraService(SimpleNamespace(enabled=True, stream_url='rtsp://192.0.2.1/main'))
    camera._import_cv2 = Mock(side_effect=CameraUnavailable('OpenCV missing'))
    camera._fail = Mock()
    camera._capture_loop()
    camera._fail.assert_called_once_with('OpenCV missing')


def test_network_reconnect_discards_old_frames_and_recovers():
    camera = CameraService(SimpleNamespace(enabled=True, stream_url='rtsp://192.0.2.1/main'))
    camera._import_cv2 = Mock(return_value=Mock())
    old, new = Mock(), Mock()
    old.read.return_value = (False, None)
    def recovered():
        camera._stop_event.set()
        return True, 'fresh'
    new.read.side_effect = recovered
    def open_again():
        assert camera._frame is None
        return new
    camera._open_capture = Mock(side_effect=[old, CameraUnavailable('offline'), new])
    camera._frame, camera._frame_ts = 'old', 1.0
    # No real-time sleep in the retry test.
    camera._stop_event.wait = Mock(return_value=False)
    camera._capture_loop()
    assert camera._frame == 'fresh'
    assert camera._frame_ts > 1
    assert camera._open_capture.call_count == 3
    old.release.assert_called_once()
