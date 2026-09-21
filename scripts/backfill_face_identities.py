"""Index archived face observations under persistent IDs without loading models.

Original events and media stay in place. Only face identity/index files are
written; dry runs open the source SQLite database read-only and emit counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hub.training_archive import TrainingArchive

_PAGE_SIZE = 128
_KINDS = {'appearance', 'enrollment_face'}
_UNKNOWN = {'', 'unknown', 'anonymous', 'unidentified', 'guest', 'none', 'null'}


def _digest(value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                     separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _observation(record):
    """Return a frame key and face, without trusting historical person labels."""
    if not isinstance(record, dict) or not isinstance(record.get('id'), str):
        raise ValueError('Invalid archive event')
    metadata = record.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('Invalid archive metadata')
    face = metadata.get('face_observation')
    if face is None or face == {}:
        return None
    if not isinstance(face, dict):
        raise ValueError('Invalid face observation')
    box = [float(value) for value in face.get('box', ())]
    if (len(box) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('Invalid face box')
    captured_at = float(record['captured_at'])
    if not math.isfinite(captured_at):
        raise ValueError('Invalid archive capture time')
    files = record.get('files', {})
    if not isinstance(files, dict):
        raise ValueError('Invalid archive media index')
    original_hash = ''
    for name in ('original.jpg', 'original.jpeg', 'original.png', 'original.webp'):
        if name in files:
            original_hash = files[name].get('sha256', '')
            if not isinstance(original_hash, str):
                raise ValueError('Invalid original media hash')
            break
    frame_id = metadata.get('frame_id')
    if frame_id is None or frame_id == '':
        # Old enrollments may lack a frame ID. A captured timestamp and original
        # hash still group simultaneous observations; no-media records stand alone.
        frame_id = f'captured:{captured_at!r}' if original_hash else f'event:{record["id"]}'
    frame_key = str(metadata.get('face_frame_key') or f'{frame_id}:{original_hash}')
    source_id = str(metadata.get('client_id') or '')
    selected = dict(face)
    # Neither a copied recognition candidate nor a historical name is enrollment.
    selected.pop('confirmed_name', None)
    person = record.get('person')
    if isinstance(person, str) and person.strip().casefold() not in _UNKNOWN:
        accepted = (record.get('kind') == 'enrollment_face'
                    and metadata.get('status') == 'accepted_enrollment')
        manual_match = (metadata.get('identity_source') == 'manual_face_match'
                        and float(metadata.get('match_score') or 0) >= .60
                        and float(selected.get('score') or 0) >= .80)
        if accepted or manual_match:
            selected['confirmed_name'] = person
    # Do not include detection quality/name in the duplicate key: the exact face
    # box and vector identify repeated records from the same captured image.
    observation_key = _digest([selected.get('box'), selected.get('embedding')])
    return (source_id, frame_key), selected, observation_key, captured_at


def _source_rows(db, maximum):
    """Finish each paged SELECT before writers can update the archive index."""
    has_index = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='face_events'"
    ).fetchone() is not None
    indexed = ('(SELECT face_id FROM face_events f WHERE f.event_id=e.id)'
               if has_index else 'NULL')
    after = 0
    while after < maximum:
        rows = db.execute(
            f'SELECT e.rowid,e.kind,e.record,{indexed} AS indexed_face_id '
            'FROM events e WHERE e.rowid>? AND e.rowid<=? ORDER BY e.rowid LIMIT ?',
            (after, maximum, _PAGE_SIZE)).fetchall()
        if not rows:
            break
        after = rows[-1]['rowid']
        yield from rows


def _frame_records(db, rowids):
    records = []
    for offset in range(0, len(rowids), _PAGE_SIZE):
        batch = rowids[offset:offset + _PAGE_SIZE]
        marks = ','.join('?' for _ in batch)
        records.extend(json.loads(row[0]) for row in db.execute(
            f'SELECT record FROM events WHERE rowid IN ({marks}) ORDER BY rowid', batch
        ).fetchall())
    if len(records) != len(rowids):
        raise ValueError('The archive snapshot changed during backfill')
    return records


def backfill(archive, *, limit=None, dry_run=False):
    """Return counts, holding JSON/vectors for at most one captured frame.

    The compact grouping index contains only row IDs and existing assignments.
    ``limit`` counts pending events and stops after a complete frame, so a frame
    crossing the limit is included in full. Appended source rows wait for the
    next run. Repeating a completed run performs no identity or index writes.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError('Limit must be a nonnegative integer')
    if not isinstance(archive, TrainingArchive):
        archive = TrainingArchive(archive)
    stats = dict(scanned=0, eligible=0, frames=0, distinct_observations=0,
                 duplicate_observations=0, missing_embedding=0, planned=0,
                 indexed=0, already_indexed=0, skipped_no_face=0,
                 skipped_other_kind=0, skipped_invalid=0, deferred_by_limit=0,
                 deferred_incomplete_frame=0,
                 errors=0, snapshot_max_rowid=0, dry_run=bool(dry_run),
                 paid_api_requests=0)
    database = archive.database
    if not database.exists():
        return stats
    if not database.resolve().is_relative_to(archive.root):
        raise ValueError('Archive index escaped its root')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        maximum = db.execute('SELECT coalesce(max(rowid),0) FROM events').fetchone()[0]
        stats['snapshot_max_rowid'] = maximum
        groups = {}
        for row in _source_rows(db, maximum):
            stats['scanned'] += 1
            if row['kind'] not in _KINDS:
                stats['skipped_other_kind'] += 1
                continue
            try:
                observation = _observation(json.loads(row['record']))
                if observation is None:
                    stats['skipped_no_face'] += 1
                    continue
                key, face, _, _ = observation
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                stats['skipped_invalid'] += 1
                continue
            stats['eligible'] += 1
            if not face.get('embedding'):
                stats['missing_embedding'] += 1
            indexed_face_id = row['indexed_face_id']
            if indexed_face_id:
                stats['already_indexed'] += 1
            groups.setdefault(key, []).append((row['rowid'], indexed_face_id))

        # Dict insertion order preserves each frame's earliest source row, and
        # row IDs preserve face ordering even when frame events are interleaved.
        for (source_id, frame_key), group in groups.items():
            pending = sum(not face_id for _, face_id in group)
            if not pending:
                continue
            if limit is not None and stats['planned'] >= limit:
                stats['deferred_by_limit'] += pending
                continue
            try:
                records = _frame_records(db, [rowid for rowid, _ in group])
                faces, indices, positions, captured = [], [], {}, None
                for record in records:
                    _, face, key, moment = _observation(record)
                    if captured is None:
                        captured = moment
                    if key not in positions:
                        positions[key] = len(faces)
                        faces.append(face)
                    else:
                        previous = faces[positions[key]]
                        if face.get('confirmed_name') and not previous.get('confirmed_name'):
                            previous['confirmed_name'] = face['confirmed_name']
                    indices.append(positions[key])
                stats['planned'] += pending
                stats['frames'] += 1
                stats['distinct_observations'] += len(faces)
                stats['duplicate_observations'] += len(records) - len(faces)
                saved = [record['metadata'].get('face_identity') for record in records]
                if any(saved):
                    # Live saves can fail after assigning a full frame but before
                    # recording every face. Reuse their durable assignments: the
                    # retained records cannot reconstruct that full matcher batch.
                    if any(not existing and not isinstance(assignment, dict)
                           for (_, existing), assignment in zip(group, saved)):
                        stats['deferred_incomplete_frame'] += pending
                        continue
                    if dry_run:
                        continue
                    per_record = [assignment if isinstance(assignment, dict)
                                  else {'face_id': existing}
                                  for (_, existing), assignment in zip(group, saved)]
                else:
                    if dry_run:
                        continue
                    assignments = archive.assign_face_ids(faces, frame_id=frame_key,
                        captured_at=captured, source_id=source_id)
                    if len(assignments) != len(faces):
                        raise ValueError('Face assignment count mismatch')
                    per_record = [assignments[index] for index in indices]
                # A partially indexed batch must reproduce existing assignments;
                # never attach a remaining face after an ordering discrepancy.
                if any(existing and assignment.get('face_id') != existing
                       for (_, existing), assignment in zip(group, per_record)):
                    raise ValueError('Previously indexed frame assignment changed')
                for record, (_, existing), assignment in zip(records, group, per_record):
                    if existing:
                        continue
                    try:
                        if archive.index_face_event(record, assignment):
                            stats['indexed'] += 1
                        else:
                            stats['already_indexed'] += 1
                    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
                        stats['errors'] += 1
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
                stats['errors'] += 1
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO_ROOT / 'data' / 'training_archive',
                        help='Existing training archive directory')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int,
                        help='Maximum pending events, including the complete final frame')
    parser.add_argument('--report', type=Path, help='Optional counts-only JSON report')
    args = parser.parse_args(argv)
    try:
        result = backfill(args.root, limit=args.limit, dry_run=args.dry_run)
        output = json.dumps(result, sort_keys=True, indent=2) + '\n'
        if args.report:
            # Reports are explicit outputs, including when inspecting a dry run.
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x', encoding='utf-8') as report:
                report.write(output)
    except (OSError, ValueError, sqlite3.Error):
        print(json.dumps({'errors': 1, 'indexed': 0, 'paid_api_requests': 0}))
        return 1
    print(output, end='')
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
