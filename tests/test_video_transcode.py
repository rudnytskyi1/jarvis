"""Alert videos must open on a phone, not only in desktop Telegram (ТЗ F-702).
The room client writes MPEG-4 Part 2 (OpenCV's ``mp4v``), which the phone apps
refuse, so the hub re-encodes each clip to H.264 with its metadata first. These
tests cover the conversion itself and the wiring that puts its result into the
Telegram call.
"""
import pytest

from hub import video_transcode
from hub.video_transcode import is_phone_ready, phone_ready_mp4

ffmpeg = pytest.mark.skipif(video_transcode.ffmpeg_path() is None,
                            reason='this machine has no ffmpeg to convert with')


def mpeg4_clip(tmp_path, seconds=1.0, fps=8, size=(160, 120)):
    """A clip produced exactly the way the room client produces one."""
    cv2 = pytest.importorskip('cv2')
    import numpy as np
    path = tmp_path / 'room-clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        pytest.skip('this OpenCV build cannot write MP4')
    for index in range(int(seconds * fps)):
        frame = np.full((size[1], size[0], 3), index % 255, dtype='uint8')
        writer.write(frame)
    writer.release()
    data = path.read_bytes()
    assert b'ftyp' in data[:64]
    return data


def test_a_clip_the_room_recorded_is_not_phone_ready(tmp_path):
    assert is_phone_ready(mpeg4_clip(tmp_path)) is False


@ffmpeg
def test_the_hub_converts_it_to_h264(tmp_path):
    converted = phone_ready_mp4(mpeg4_clip(tmp_path))
    assert converted is not None, 'a real clip must convert'
    assert is_phone_ready(converted) is True
    assert b'ftyp' in converted[:64], 'still an MP4 container'


def test_a_clip_that_is_already_h264_is_left_alone():
    already = b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 64 + b'avc1' + b'\x00' * 32
    assert is_phone_ready(already) is True
    assert phone_ready_mp4(already) is already


def test_without_an_encoder_the_original_is_all_we_have(monkeypatch, tmp_path):
    monkeypatch.setattr(video_transcode, 'ffmpeg_path', lambda: None)
    assert phone_ready_mp4(mpeg4_clip(tmp_path)) is None


def test_something_that_is_not_a_video_never_raises():
    assert phone_ready_mp4(b'not a video at all, honestly') is None
    assert phone_ready_mp4(b'') is None
