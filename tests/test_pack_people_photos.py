"""The people-photo packer: categories, dedupe and the archive it writes."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


packer = _load('pack_people_photos', REPO_ROOT / 'scripts' / 'pack-people-photos.py')


def _archive(tmp_path: Path) -> Path:
    """A tiny training archive with one event per person and category."""
    archive = tmp_path / 'training_archive'
    rows = [
        ('person-1', 'Anton', 'face.png', b'face-of-anton', 1_700_000_000),
        ('person-1', 'Anton', 'body.png', b'body-of-anton', 1_700_000_100),
        ('person-1', 'Anton', 'body.png', b'body-of-anton', 1_700_000_200),  # same bytes
        ('person-2', 'John', 'body.png', b'body-of-john', 1_700_000_300),
        ('unknown', 'unknown', 'body.png', b'body-of-a-stranger', 1_700_000_400),
        ('person-1', 'Anton', 'original.jpg', b'scene-frame', 1_700_000_500),
    ]
    events = []
    for index, (person_id, person, key, payload, captured) in enumerate(rows):
        folder = archive / '2026-01-01' / f'{person}--{index:02d}' / 'events' / f'e{index:02d}'
        folder.mkdir(parents=True)
        (folder / key).write_bytes(payload)
        relative = f'2026-01-01/{person}--{index:02d}/events/e{index:02d}/{key}'
        events.append((person_id, 'appearance', captured,
                       json.dumps({'person': person, 'files': {key: {
                           'path': relative, 'bytes': len(payload),
                           'sha256': f'digest-{payload.decode()}'}}})))
    conn = sqlite3.connect(archive / 'index.sqlite3')
    conn.execute('create table events (id integer primary key, person_id text, kind text,'
                 ' captured_at real, folder text, record text)')
    conn.executemany('insert into events (person_id, kind, captured_at, record)'
                     ' values (?,?,?,?)', events)
    conn.commit()
    conn.close()
    return archive


def test_faces_and_bodies_land_in_their_own_folders(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    out = tmp_path / 'people.zip'

    report = packer.build_archive(out=out, people=None, include_gallery=False)

    with zipfile.ZipFile(out) as bundle:
        names = bundle.namelist()
        readme = bundle.read('README.md').decode('utf-8')
        index = bundle.read('index.csv').decode('utf-8')
    assert 'Anton/faces/' in '\n'.join(names)
    assert 'Anton/body/' in '\n'.join(names)
    assert 'John/body/' in '\n'.join(names)
    assert 'unknown/body/' in '\n'.join(names)
    assert not any(name.startswith('Anton/scenes/') for name in names), \
        'кадр комнаты в архив людей не попадает без --with-scenes'
    assert 'faces/' in readme and 'body/' in readme
    assert 'Anton / body | 1' in readme, 'повтор по sha256 не попадает в архив дважды'
    assert index.count('Anton/body/') == 1
    assert report['files'] == 4
    assert report['counts'] == {'Anton/body': 1, 'Anton/faces': 1,
                                'John/body': 1, 'unknown/body': 1}


def test_scenes_come_only_when_asked(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    out = tmp_path / 'people.zip'

    packer.build_archive(out=out, people=None, with_scenes=True, include_gallery=False)

    with zipfile.ZipFile(out) as bundle:
        assert any(name.startswith('Anton/scenes/') for name in bundle.namelist())


def test_named_and_unknown_can_be_kept_apart(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    monkeypatch.setattr(packer, 'GALLERY', tmp_path / 'no-gallery')
    named, unknown = packer.known_people()
    assert named == {'Anton', 'John'} and unknown == {'unknown'}

    packer.build_archive(out=tmp_path / 'named.zip', people=named, include_gallery=False)
    packer.build_archive(out=tmp_path / 'unknown.zip', people=unknown,
                         include_gallery=False)

    with zipfile.ZipFile(tmp_path / 'named.zip') as bundle:
        assert all('/' in name and not name.startswith('unknown/')
                   for name in bundle.namelist() if name.endswith('.png'))
    with zipfile.ZipFile(tmp_path / 'unknown.zip') as bundle:
        assert [name for name in bundle.namelist() if name.endswith('.png')] == \
            [name for name in bundle.namelist() if name.startswith('unknown/')]


def test_the_limit_caps_one_category(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    out = tmp_path / 'people.zip'
    report = packer.build_archive(out=out, people=None, limit=1, include_gallery=False)
    assert all(number <= 1 for number in report['counts'].values())


def test_a_missing_gallery_is_not_a_crash(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    assert list(packer.iter_gallery(tmp_path / 'nowhere')) == []


def test_a_folder_name_windows_cannot_use_is_replaced():
    assert packer._safe('John: the/system?') == 'John_ the_system_'
    assert packer._safe('   ..   ') == 'unknown'


def test_a_gallery_without_sha256_still_gets_a_file_name(tmp_path, monkeypatch):
    """Галерея профилей не хранит sha256: в имя не должен попадать путь."""
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    gallery = tmp_path / 'appearance'
    gallery.mkdir()
    (gallery / 'anton-face.jpg').write_bytes(b'reference-face')
    conn = sqlite3.connect(gallery / 'gallery.sqlite3')
    conn.execute('create table samples (person text, captured_name text,'
                 ' captured_at real, face_path text, body_path text)')
    conn.execute("insert into samples values ('Anton', 'Anton', 1700000000,"
                 " 'anton-face.jpg', '')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(packer, 'GALLERY', gallery)
    out = tmp_path / 'people.zip'

    packer.build_archive(out=out, people={'Anton'})

    with zipfile.ZipFile(out) as bundle:
        names = [name for name in bundle.namelist() if name.endswith('.jpg')]
    assert names == ['Anton/profile/'
                     f'{packer._stamp(1700000000)}_'
                     f'{packer._fallback_digest(gallery / "anton-face.jpg")[:8]}.jpg']
    assert all(not set(name) & set('<>:"|?*') for name in names), \
        'в именах внутри архива не может быть символов, запрещённых Windows'


def test_every_entry_name_stays_portable(tmp_path, monkeypatch):
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    monkeypatch.setattr(packer, 'GALLERY', tmp_path / 'no-gallery')
    out = tmp_path / 'people.zip'

    packer.build_archive(out=out, people=None, include_gallery=False)

    with zipfile.ZipFile(out) as bundle:
        for name in bundle.namelist():
            assert name.startswith(('Anton/', 'John/', 'unknown/',
                                    'index.csv', 'README.md')), name
            assert not set(name) & set('<>:"|?*\\')


def test_a_big_group_is_cut_into_parts_without_losing_photos(tmp_path, monkeypatch):
    """Части не пересекаются и вместе дают ровно те же снимки, что один архив."""
    archive = _archive(tmp_path)
    monkeypatch.setattr(packer, 'ARCHIVE', archive)
    monkeypatch.setattr(packer, 'GALLERY', tmp_path / 'no-gallery')
    whole = tmp_path / 'whole.zip'
    packer.build_archive(out=whole, people=None, include_gallery=False)

    items = packer._collect(people=None, include_gallery=False)
    parts = packer._share_out(items, 2)
    volumes = [sum(item[2].stat().st_size for item in part) for part in parts]

    assert len(parts) == 2
    assert sum(len(part) for part in parts) == 4, 'дедуп общий на все части'
    digests = [digest for part in parts for _p, _c, _path, _w, digest in part]
    assert len(digests) == len(set(digests)), 'один снимок не попадает в две части'
    assert abs(volumes[0] - volumes[1]) <= max(volumes), 'части примерно равны'

    with zipfile.ZipFile(whole) as bundle:
        in_whole = {name for name in bundle.namelist() if name.endswith('.png')}
    for number, part in enumerate(parts, start=1):
        out = tmp_path / f'part{number}.zip'
        packer.build_archive(out=out, items=part, note=f'Часть {number} из 2.')
        with zipfile.ZipFile(out) as bundle:
            names = {name for name in bundle.namelist() if name.endswith('.png')}
            assert f'Часть {number} из 2.' in bundle.read('README.md').decode('utf-8')
        assert names <= in_whole
    together = set()
    for part in parts:
        for _p, _c, path, _w, _d in part:
            together.add(path)
    assert len(together) == 4
