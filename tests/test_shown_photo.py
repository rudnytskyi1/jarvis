"""A picture for the screen is opened by Windows' photo app, not by us.

Владелец 2026-09-24: «просто когда картинку показывает на экране можно ее
сохранить как .temp куда-то и открыть в приложении фото? и проблема
исправлена» - про мигающий оверлей: two always-on-top windows traded the top
slot for as long as a picture was up.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from client.actions import photos

JPEG = b"\xff\xd8\xff\xe0" + b"rowan" * 8


def test_the_picture_is_written_to_a_temp_file_and_opened(tmp_path):
    opened: list[str] = []

    path = photos.show_pushed_photo(JPEG, 'Created with Nano Banana 2',
                                    opener=opened.append, folder=tmp_path)

    assert path.parent == tmp_path and path.suffix == '.jpg'
    assert path.read_bytes() == JPEG
    assert opened == [str(path)], 'картинку открывает системное приложение фото'


def test_the_temp_copy_leaves_the_desktop_alone(tmp_path):
    path = photos.shown_photo_path(JPEG, 'photo', folder=tmp_path)
    assert path.parent == tmp_path, 'это одноразовая копия, а не файл на рабочем столе'


def test_two_pictures_in_the_same_second_do_not_overwrite_each_other(tmp_path):
    stamp = datetime(2026, 9, 24, 0, 5, 0)

    first = photos.shown_photo_path(JPEG, 'one', folder=tmp_path, now=stamp)
    second = photos.shown_photo_path(JPEG, 'two', folder=tmp_path, now=stamp)

    assert first != second and second.exists()


def test_an_empty_picture_is_refused(tmp_path):
    with pytest.raises(ValueError):
        photos.shown_photo_path(b'', 'nothing', folder=tmp_path)


def test_close_the_photo_targets_only_our_own_window(tmp_path):
    photos.show_pushed_photo(JPEG, 'rowan', opener=lambda _: None, folder=tmp_path)
    mine = photos._shown_paths[-1].name
    listed = [SimpleNamespace(hwnd=1, pid=10, title=f'{mine} - Photos'),
              SimpleNamespace(hwnd=2, pid=11, title='Untitled - Notepad')]
    closed: list[list] = []

    assert photos.close_shown_photos(windows=listed, closer=lambda rows: closed.append(rows) or True)

    assert [row['hwnd'] for row in closed[0]] == [1], 'чужое окно закрывать нельзя'


def test_close_the_photo_says_no_when_nothing_was_shown():
    photos._shown_paths.clear()
    assert photos.close_shown_photos(windows=[], closer=lambda rows: True) is False
