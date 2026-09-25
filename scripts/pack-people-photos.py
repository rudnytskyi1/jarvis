"""Pack the photos of people into ZIP archives, by person and by category.

Владелец 2026-09-24: «можешь упаковать архив фото всех людей по категориям (не
только лица но и обрезаный с фото в полный рост)».

Фото людей лежат в двух местах:

* ``data/training_archive/<дата>/<Имя>--<хеш>/events/<время>-<id>/`` — на каждое
  наблюдение кадр сцены (``original.jpg``), обрезанное тело (``body.png``) и,
  если лицо было видно, лицо (``face.png``); индекс —
  ``data/training_archive/index.sqlite3`` (таблица ``events``, поле ``record``
  с путями, размерами и sha256 каждого файла);
* ``data/appearance/<профиль>/<сэмпл>-face.jpg`` — эталонные снимки профилей
  (галерея ``data/appearance/gallery.sqlite3``).

Что делает упаковщик: раскладывает снимки по человеку и категории
(``<Имя>/faces/`` и ``<Имя>/body/``), выбрасывает байт-в-байт повторы по
sha256, кладёт внутрь ``index.csv`` (что именно и когда снято) и ``README.md``.
Кадры сцены (``original.jpg``) в архив НЕ попадают по умолчанию: это 28 ГБ
интерьера, а не люди (``--with-scenes`` включает их).

Примеры:

    python scripts/pack-people-photos.py                  # все, двумя архивами
    python scripts/pack-people-photos.py --named-only      # только узнанные
    python scripts/pack-people-photos.py --unknown-only --split 2   # unknown на два
    python scripts/pack-people-photos.py --limit 500       # не больше 500 на категорию
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sqlite3
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data' / 'training_archive'
#: Эталонные снимки профилей; путь берётся на момент вызова (см. iter_gallery).
GALLERY = ROOT / 'data' / 'appearance'
#: Как узнаём файл наблюдения по имени в записи события.
FACE_KEYS = ('face.png', 'face.jpg')
BODY_KEYS = ('body.png', 'body.jpg', 'legacy_crop.jpg')
SCENE_KEYS = ('original.jpg',)


def _category_of(key: str, *, with_scenes: bool) -> str:
    if key in FACE_KEYS:
        return 'faces'
    if key in BODY_KEYS:
        return 'body'
    if with_scenes and key in SCENE_KEYS:
        return 'scenes'
    return ''


def _stamp(captured_at) -> str:
    try:
        moment = dt.datetime.fromtimestamp(float(captured_at))
    except (TypeError, ValueError, OSError):
        return 'unknown-time'
    return moment.strftime('%Y-%m-%d_%H%M%S')


def iter_archive(archive: Path | None = None, *, with_scenes: bool = False):
    """Every photo of a person: ``(person, category, path, captured_at, sha256)``.

    Oldest first inside a person, so the archive reads as a story and two runs
    produce the same ZIP. ``archive`` defaults to :data:`ARCHIVE` at CALL time,
    not at import time: a caller (or a test) that points the module somewhere
    else must not still read the real training archive.
    """
    archive = archive or ARCHIVE
    conn = sqlite3.connect(f'{(archive / "index.sqlite3").resolve().as_uri()}?mode=ro',
                           uri=True)
    try:
        rows = conn.execute('select person_id, record, captured_at from events'
                            ' order by captured_at').fetchall()
    finally:
        conn.close()
    for person_id, record, captured_at in rows:
        try:
            payload = json.loads(record or '{}')
        except ValueError:
            continue
        person = ' '.join(str(payload.get('person') or person_id or 'unknown').split())
        files = payload.get('files') or {}
        for key, entry in files.items():
            category = _category_of(key, with_scenes=with_scenes)
            if not category or not isinstance(entry, dict):
                continue
            relative = str(entry.get('path') or '')
            if not relative:
                continue
            yield (person or 'unknown', category, archive / relative,
                   entry.get('captured_at') or captured_at,
                   str(entry.get('sha256') or ''))


def iter_gallery(gallery: Path | None = None):
    """The reference photo of every profile, as ``(person, 'profile', path, …)``."""
    gallery = gallery or GALLERY
    database = gallery / 'gallery.sqlite3'
    if not database.exists():
        return
    conn = sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True)
    try:
        columns = [row[1] for row in conn.execute('pragma table_info(samples)')]
        if not columns:
            return
        rows = conn.execute('select * from samples').fetchall()
    except sqlite3.Error:
        return
    finally:
        conn.close()
    for row in rows:
        item = dict(zip(columns, row))
        person = ' '.join(str(item.get('captured_name') or item.get('person')
                               or 'unknown').split()) or 'unknown'
        for field, category in (('face_path', 'profile'), ('body_path', 'body')):
            relative = str(item.get(field) or '')
            if not relative:
                continue
            yield (person, category, gallery / relative, item.get('captured_at'),
                   str(item.get('sha256') or ''))


def _unique(items):
    """Drop byte-identical copies: one photo twice is not an archive."""
    seen: set[str] = set()
    for person, category, path, when, digest in items:
        if not path.exists():
            continue
        key = digest or _fallback_digest(path)
        if key in seen:
            continue
        seen.add(key)
        yield person, category, path, when, key


def _fallback_digest(path: Path) -> str:
    """A hex digest for the files no sha256 was recorded for (profile gallery).

    Хешируем не файл, а его путь и размер: галерея профилей хранит тысячи
    снимков, и читать их целиком только ради имени файла незачем. Хеш при этом
    остаётся шестнадцатеричным, поэтому из него получается годное имя файла
    (в сами имена путей не попадает).
    """
    key = f'{path}:{path.stat().st_size}'
    return hashlib.sha256(key.encode('utf-8', 'replace')).hexdigest()


def _safe(name: str) -> str:
    """A folder name a ZIP and Windows Explorer both accept."""
    cleaned = ''.join('_' if char in '<>:"/\\|?*' else char for char in str(name))
    cleaned = ' '.join(cleaned.split()).strip(' .')
    return (cleaned or 'unknown')[:60]


def _collect(*, people: set[str] | None = None, with_scenes: bool = False,
             include_gallery: bool = True) -> list:
    """Every photo that belongs in an archive, before dedupe."""
    items = list(iter_archive(with_scenes=with_scenes))
    if include_gallery:
        items.extend(iter_gallery())
    if people is not None:
        items = [item for item in items if item[0] in people]
    return items


def _share_out(items: list, parts: int) -> list[list]:
    """Split the photos into ``parts`` archives of roughly equal volume.

    Файлы разные по весу (лицо — 10 КБ, тело — 120 КБ), поэтому режем не по
    количеству, а по байтам: тяжёлые кладём первыми в самую лёгкую часть.
    Дедуп здесь общий на все части, иначе один снимок попал бы в две.
    """
    unique = list(_unique(items))
    buckets: list[list] = [[] for _ in range(parts)]
    totals = [0] * parts
    for item in sorted(unique, key=lambda entry: entry[2].stat().st_size,
                       reverse=True):
        lightest = totals.index(min(totals))
        buckets[lightest].append(item)
        totals[lightest] += item[2].stat().st_size
    for bucket in buckets:
        bucket.sort(key=lambda entry: (str(entry[3]), str(entry[2])))
    return buckets


def build_archive(*, out: Path, people: set[str] | None = None, limit: int = 0,
                  with_scenes: bool = False, include_gallery: bool = True,
                  unknown_name: str = 'unknown', items: list | None = None,
                  note: str = '') -> dict:
    """Write one ZIP; return what went in (counts, bytes, the file)."""
    if items is None:
        items = _collect(people=people, with_scenes=with_scenes,
                         include_gallery=include_gallery)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[tuple[str, str], int] = {}
    index = io.StringIO()
    writer = csv.writer(index)
    writer.writerow(['person', 'category', 'file_in_zip', 'captured_at', 'sha256', 'bytes'])
    written = 0
    total_bytes = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for person, category, path, when, digest in _unique(items):
            if limit and counts.get((person, category), 0) >= limit:
                continue
            counts[(person, category)] = counts.get((person, category), 0) + 1
            suffix = path.suffix.lower() or '.jpg'
            stamp = _safe(digest[:8]) or 'no-hash'
            name = f'{_safe(person)}/{category}/{_stamp(when)}_{stamp}{suffix}'
            bundle.write(path, name)
            size = path.stat().st_size
            written += 1
            total_bytes += size
            writer.writerow([person, category, name, when, digest, size])
        bundle.writestr('index.csv', index.getvalue())
        bundle.writestr('README.md',
                        _readme(counts, written, total_bytes, unknown_name, note))
    return {'zip': str(out), 'files': written, 'bytes': total_bytes,
            'counts': {f'{person}/{category}': number
                       for (person, category), number in sorted(counts.items())}}


def _readme(counts, written: int, total_bytes: int, unknown_name: str,
            note: str = '') -> str:
    lines = [
        '# Photos of people',
        '',
        *([note, ''] if note else []),
        'Раскладка: `<человек>/<категория>/<дата>_<время>_<хеш>.png`.',
        '',
        '* `faces/` — обрезанные лица (то, по чему система узнаёт);',
        '* `body/` — обрезанные фото в полный рост (тело целиком);',
        '* `profile/` — эталонный снимок профиля из галереи;',
        '* `scenes/` — кадр комнаты целиком (только если просили `--with-scenes`).',
        '',
        f'Папка `{_safe(unknown_name)}/` — люди, которых система не узнала.',
        'Одинаковые снимки (совпадающий sha256) лежат в архиве один раз.',
        '',
        f'Всего файлов: {written}, объём: {total_bytes / 1e9:.2f} GB.',
        '',
        '| человек / категория | файлов |',
        '|---|---|',
    ]
    for (person, category), number in sorted(counts.items()):
        lines.append(f'| {person} / {category} | {number} |')
    lines.append('')
    return '\n'.join(lines)


def known_people() -> tuple[set[str], set[str]]:
    """Names the archive knows, split into named people and the unknown bucket."""
    names = {person for person, *_rest in iter_archive()}
    names |= {person for person, *_rest in iter_gallery()}
    named = {name for name in names if name != 'unknown'}
    return named, names - named


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=str(ROOT / 'data' / 'people'))
    parser.add_argument('--named-only', action='store_true',
                        help='только люди с именем, без папки unknown')
    parser.add_argument('--unknown-only', action='store_true',
                        help='только те, кого не узнали')
    parser.add_argument('--single', metavar='NAME.zip',
                        help='один архив со всеми вместо двух групп')
    parser.add_argument('--limit', type=int, default=0,
                        help='максимум файлов на человека и категорию (0 — все)')
    parser.add_argument('--with-scenes', action='store_true',
                        help='включить кадры комнаты целиком (десятки ГБ)')
    parser.add_argument('--no-gallery', action='store_true',
                        help='не брать эталонные снимки профилей')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='резать каждую группу на N архивов равного объёма')
    parser.add_argument('--out-name', metavar='STEM',
                        help='имя архива без даты и расширения (для одной группы)')
    arguments = parser.parse_args()

    named, unknown = known_people()
    out_dir = Path(arguments.out_dir)
    stamp = dt.datetime.now().strftime('%Y%m%d')
    gallery = not arguments.no_gallery
    split = max(1, arguments.split)
    report = []

    def emit(stem: str, people: set[str] | None, *, with_gallery: bool) -> None:
        """One group of people, cut into `split` archives when asked."""
        items = _collect(people=people, with_scenes=arguments.with_scenes,
                         include_gallery=with_gallery)
        buckets = _share_out(items, split) if split > 1 else [items]
        for number, bucket in enumerate(buckets, start=1):
            part = f'-part{number}' if split > 1 else ''
            note = ''
            if split > 1:
                note = (f'Часть {number} из {split}: остальные фото лежат в '
                        f'соседних архивах {stem}-part*.zip.')
            report.append(build_archive(out=out_dir / f'{stem}{part}.zip',
                                        people=people, limit=arguments.limit,
                                        include_gallery=with_gallery,
                                        items=bucket, note=note))

    if arguments.single:
        emit(Path(arguments.single).stem, None, with_gallery=gallery)
    else:
        if not arguments.unknown_only and named:
            emit(arguments.out_name or f'people-named-{stamp}', named,
                 with_gallery=gallery)
        if not arguments.named_only and unknown:
            stem = arguments.out_name if arguments.out_name and arguments.unknown_only \
                else f'people-unknown-{stamp}'
            emit(stem, unknown, with_gallery=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
