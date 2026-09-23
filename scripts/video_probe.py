"""Почему запись уведомления не открывается на телефоне: прогон перекодирования.

Владелец 2026-09-23: «опять видео в телеге на мобилке не грузятся». Клип пишется
клиентом через OpenCV (`mp4v`, MPEG-4 Part 2), который телефоны не играют, и хаб
обязан перекодировать его в H.264 перед отправкой (`hub/video_transcode.py`).
Этот скрипт делает ровно такой клип и показывает ПОШАГОВО, где перекодирование
ломается — с полным трейсбеком, а не одной строкой в логе:

    python scripts/video_probe.py
    python scripts/video_probe.py "C:/path/to/clip.mp4"   # проверить настоящий клип
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def make_clip() -> bytes:
    """Клип тем же способом, что и клиент комнаты (OpenCV, ``mp4v``)."""
    import cv2
    import numpy as np

    path = Path(tempfile.gettempdir()) / "rowan-video-probe.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (320, 240))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = 40
    for index in range(16):
        frame[:, : index * 10 % 320] = 200
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def main() -> int:
    sys.path.insert(0, str(REPO))
    from hub import video_transcode as vt

    if len(sys.argv) > 1:
        data = Path(sys.argv[1]).read_bytes()
        print(f"клип: {sys.argv[1]} ({len(data)} байт)")
    else:
        data = make_clip()
        print(f"клип: свежий, как пишет комната ({len(data)} байт)")
    print(f"H.264 уже внутри: {vt.is_phone_ready(data)}")
    print(f"ffmpeg: {vt.ffmpeg_path()}")
    print(f"ROWAN_FFMPEG в окружении: {vt.os.environ.get(vt.FFMPEG_ENV)!r}")

    encoder = vt.ffmpeg_path()
    if encoder:
        try:
            ready = subprocess.run([encoder, "-version"], capture_output=True, timeout=30)
            print(f"запуск ffmpeg: rc={ready.returncode}")
        except Exception:  # noqa: BLE001 - это и есть предмет проверки
            print("запуск ffmpeg упал:")
            traceback.print_exc()
        try:
            with tempfile.TemporaryDirectory(prefix="rowan-probe-") as folder:
                source = Path(folder) / "in.mp4"
                target = Path(folder) / "out.mp4"
                source.write_bytes(data)
                print(f"исходник записан: {source.stat().st_size} байт")
                done = subprocess.run(
                    [encoder, "-y", "-loglevel", "error", "-i", str(source),
                     *vt._ARGS, str(target)],
                    capture_output=True, timeout=120)
                print(f"ffmpeg: rc={done.returncode}, stderr={done.stderr[:200]!r}")
                if target.is_file():
                    print(f"результат: {target.stat().st_size} байт, "
                          f"H.264 внутри: {vt.is_phone_ready(target.read_bytes())}")
        except Exception:  # noqa: BLE001 - показываем настоящую причину
            print("шаг перекодирования упал:")
            traceback.print_exc()

    result = vt.phone_ready_mp4(data)
    if result is None:
        trace_av(data)
        print("phone_ready_mp4 -> None: клип уйдёт как есть, телефон его не откроет")
        return 1
    print(f"phone_ready_mp4 -> {len(result)} байт, "
          f"H.264 внутри: {vt.is_phone_ready(result)}")
    return 0


def trace_av(data: bytes) -> None:
    """Пошаговый разбор пути PyAV: где именно он отказывается кодировать."""
    try:
        import av
    except ImportError:
        print("PyAV (`av`) не установлен — остаётся только ffmpeg")
        return
    import io

    variants = [("в память, без movflags", {}, None),
                ("в память, +faststart", {}, {"movflags": "+faststart"}),
                ("в файл, +faststart", {"preset": "veryfast", "crf": "26"}, {"movflags": "+faststart"}),
                ("в файл, без movflags", {"preset": "veryfast", "crf": "26"}, None)]
    for label, options, muxer_options in variants:
        try:
            with av.open(io.BytesIO(data), mode="r") as source:
                stream = source.streams.video[0]
                if "файл" in label:
                    target = Path(tempfile.gettempdir()) / f"rowan-av-{len(label)}.mp4"
                    out = target
                    opened = av.open(str(target), mode="w", format="mp4",
                                     options=dict(muxer_options or {}))
                else:
                    out = io.BytesIO()
                    opened = av.open(out, mode="w", format="mp4",
                                     options=dict(muxer_options or {}))
                with opened as output:
                    encoded = output.add_stream("libx264", rate=stream.average_rate)
                    encoded.width = stream.codec_context.width
                    encoded.height = stream.codec_context.height
                    encoded.pix_fmt = "yuv420p"
                    encoded.options = dict(options)
                    for frame in source.decode(stream):
                        for packet in encoded.encode(frame):
                            output.mux(packet)
                    for packet in encoded.encode(None):
                        output.mux(packet)
                text = out.getvalue() if isinstance(out, io.BytesIO) else out.read_bytes()
            from hub import video_transcode as vt
            print(f"PyAV {label}: {len(text)} байт, H.264={vt.is_phone_ready(text)}")
        except Exception as exc:  # noqa: BLE001 - это и есть предмет проверки
            print(f"PyAV {label}: упало {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
