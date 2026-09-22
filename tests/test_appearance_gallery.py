"""Identity admission, persistent references, and bounded adaptive matching."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from hub.appearance import AppearanceGallery, LearnedProfiles


def scene(seed=1):
    image = np.random.default_rng(seed).integers(0, 256, (320, 320, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def face(vector=None, **extra):
    return dict(box=[.2, .1, .55, .45], embedding=vector if vector is not None else [1., 0., 0.], score=.95, **extra)


def track(name="Anton", **extra):
    return dict(id="room:1", name=name, box=[.1, 0., .7, 1.], **extra)


def confirm(gallery, jpeg=None, row=None, detected=None, profiles=None, start=100, faces=None):
    jpeg = scene() if jpeg is None else jpeg
    detected = face() if detected is None else detected
    row = track() if row is None else row
    profiles = {"Anton": [[1., 0., 0.]]} if profiles is None else profiles
    answer = None
    for now in (start, start + .5, start + 1):
        answer = gallery.observe(jpeg, row, detected, profiles, faces=faces, now=now)
    return answer


def sample_count(gallery):
    with sqlite3.connect(gallery.database) as db:
        return db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]


def test_constructor_and_empty_reads_are_lazy(tmp_path):
    root = tmp_path / "not-created"
    gallery = AppearanceGallery(root)
    assert gallery.references("Anton") == []
    assert gallery.list_people() == []
    assert gallery.learned_profiles({"Anton": [[1, 0, 0]]}) == {"Anton": [[1, 0, 0]]}
    assert not root.exists()


def test_confirmation_counts_distinct_frames_not_seconds(tmp_path):
    """Owner's report (2026-09-22): a quick pass-by was never learned.

    Requiring a full second of presence meant somebody who crossed the room in
    half a second was seen but never kept. Three distinct frames of one burst
    are three observations of the same person; the same frame repeated is not.
    """
    gallery = AppearanceGallery(tmp_path)
    jpeg, row, detected, profiles = scene(), track(), face(), {"Anton": [[1, 0, 0]]}
    first = gallery.observe(jpeg, row, detected, profiles, now=10, frame_id='p1')
    assert first is None
    assert gallery.observe(jpeg, row, detected, profiles, now=10.05, frame_id='p1') is None
    assert gallery.observe(jpeg, row, detected, profiles, now=10.2, frame_id='p2') is None
    accepted = gallery.observe(jpeg, row, detected, profiles, now=10.45, frame_id='p3')
    assert accepted["name"] == "Anton"
    assert accepted["quality"]["identity_score"] == pytest.approx(1)
    assert sample_count(gallery) == 1


def test_a_frames_gap_restarts_the_confirmation(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    profiles = {"Anton": [[1, 0, 0]]}
    assert gallery.observe(scene(), track(), face(), profiles, now=10, frame_id='p1') is None
    assert gallery.observe(scene(), track(), face(), profiles, now=10.5, frame_id='p2') is None
    # Six seconds later this is a new sighting, not the third frame.
    assert gallery.observe(scene(), track(), face(), profiles, now=16.5, frame_id='p3') is None
    assert gallery.observe(scene(), track(), face(), profiles, now=17, frame_id='p4') is None
    assert gallery.observe(scene(), track(), face(), profiles, now=17.5, frame_id='p5')


def test_repeated_same_timestamp_does_not_confirm(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    for _ in range(10):
        assert gallery.observe(scene(), track(), face(), {"Anton": [[1, 0, 0]]}, now=1) is None
    assert not gallery.database.exists()


@pytest.mark.parametrize("detected,profiles", [
    (dict(face(), score=.79), {"Anton": [[1, 0, 0]]}),
    (face([.59, .8074, 0]), {"Anton": [[1, 0, 0]]}),
    (face(), {"Anton": [[1, 0, 0]], "John": [[.999, .01, 0]]}),
    (face(), {"John": [[1, 0, 0]]}),
    (dict(face(), box=[.1, .1, .2, .2]), {"Anton": [[1, 0, 0]]}),
    (face([float("nan"), 0, 0]), {"Anton": [[1, 0, 0]]}),
])
def test_weak_ambiguous_small_or_wrong_identity_cannot_teach(tmp_path, detected, profiles):
    gallery = AppearanceGallery(tmp_path)
    assert confirm(gallery, detected=detected, profiles=profiles) is None
    assert not gallery.database.exists()


def test_blur_cannot_teach(tmp_path):
    blurred = cv2.imencode(".jpg", np.full((320, 320, 3), 120, dtype=np.uint8))[1].tobytes()
    gallery = AppearanceGallery(tmp_path)
    assert confirm(gallery, jpeg=blurred) is None


def test_rejected_frame_breaks_identity_confirmation(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    profiles = {"Anton": [[1, 0, 0]]}
    for now in (10, 10.5):
        assert gallery.observe(scene(), track(), face(), profiles, now=now) is None
    assert gallery.observe(scene(), track(), face([0, 1, 0]), profiles, now=10.7) is None
    assert gallery.observe(scene(), track(), face(), profiles, now=11) is None
    assert gallery.observe(scene(), track(), face(), profiles, now=11.5) is None
    assert gallery.observe(scene(), track(), face(), profiles, now=12)


def test_different_identity_on_same_track_restarts_confirmation(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    profiles = {"Anton": [[1, 0, 0]], "John": [[0, 1, 0]]}
    for now in (10, 10.5):
        assert gallery.observe(scene(), track(), face(), profiles, now=now) is None
    assert gallery.observe(scene(), track("John"), face([0, 1, 0]), profiles, now=10.7) is None
    assert gallery.observe(scene(), track(), face(), profiles, now=11) is None
    assert gallery.observe(scene(), track(), face(), profiles, now=11.5) is None
    assert gallery.observe(scene(), track(), face(), profiles, now=12)


def test_body_reference_requires_unambiguous_ownership(tmp_path):
    own_face = face()
    gallery = AppearanceGallery(tmp_path)
    assert confirm(gallery, detected=own_face, faces=[own_face])
    refs = AppearanceGallery(tmp_path).references("anton")
    assert [r["kind"] for r in refs] == ["face", "body"]
    assert refs[0]["jpeg"].startswith(b"\xff\xd8")
    assert refs[0]["captured_at"] == 101
    assert "most recent saved appearance" in refs[0]["label"]
    assert "no evidence that they are in the room" in refs[0]["label"]


def test_the_newest_appearance_wins_over_an_older_better_frame(tmp_path):
    """Owner's report (2026-09-22): the generated look was from another day.

    The gallery used to rank identity quality (with a small age penalty), so a
    two-day-old frame could supply today's clothes. References now follow the
    owner's newest usable photograph.
    """
    gallery = AppearanceGallery(tmp_path)
    assert gallery.enroll(scene(1), "Anton", face(), now=100_000)
    assert confirm(gallery, jpeg=scene(2), detected=face([.70, np.sqrt(1 - .49), 0]),
                   start=103_600)  # an hour later, still a confirmed identity
    refs = gallery.references("Anton")
    assert refs[0]["captured_at"] == 103_601
    assert refs[0]["quality"]["identity_score"] == pytest.approx(.70)


def test_a_weak_newest_frame_does_not_become_the_reference(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    assert gallery.enroll(scene(1), "Anton", face(), now=100_000)
    assert confirm(gallery, jpeg=scene(2), detected=face([.62, np.sqrt(1 - .62 ** 2), 0]),
                   start=103_600)
    refs = gallery.references("Anton")
    assert refs[0]["captured_at"] == 100_000
    assert refs[0]["quality"]["identity_score"] == pytest.approx(1.0)


def test_the_body_reference_comes_from_the_same_moment_as_the_face(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    own_face = face()
    assert confirm(gallery, jpeg=scene(1), detected=own_face, faces=[own_face], start=100)
    assert confirm(gallery, jpeg=scene(2), detected=own_face, faces=[own_face], start=300)
    refs = gallery.references("Anton")
    assert [r["kind"] for r in refs] == ["face", "body"]
    assert refs[0]["sample_id"] == refs[1]["sample_id"]


@pytest.mark.parametrize("mode", ["missing_faces", "empty_faces", "other_face", "overlapping_bodies"])
def test_ambiguous_bodies_never_become_references(tmp_path, mode):
    own_face = face()
    faces = [own_face]
    row = track()
    if mode == "missing_faces":
        faces = None
    elif mode == "empty_faces":
        faces = []
    elif mode == "other_face":
        faces.append(dict(face([0, 1, 0]), box=[.56, .5, .68, .7]))
    else:
        row["body_unambiguous"] = False
    gallery = AppearanceGallery(tmp_path)
    assert confirm(gallery, row=row, faces=faces)
    assert [r["kind"] for r in gallery.references("Anton")] == ["face"]


def test_retains_every_accepted_image_across_days_and_restart(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    for index in range(30):
        assert gallery.enroll(scene(index), "Anton", face(), now=index * 86400)
    files = list(tmp_path.rglob("*.jpg"))
    assert len(files) == 30
    newer = AppearanceGallery(tmp_path)
    assert newer.list_people() == ["Anton"]
    assert newer.references("Anton")[0]["captured_at"] >= 27 * 86400
    assert sample_count(newer) == 30
    assert all(p.exists() for p in files)


def test_interval_and_diversity_stop_identical_frame_spam(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    assert confirm(gallery, start=100)
    assert confirm(gallery, jpeg=scene(2), start=120) is None  # within cooldown
    assert confirm(gallery, start=200) is None  # same face/appearance
    assert confirm(gallery, jpeg=scene(2), start=300)
    assert sample_count(gallery) == 2


def test_automatic_samples_cannot_become_their_own_manual_anchors(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    manual = {"Anton": [[1., 0., 0.]]}
    accepted = face([.65, np.sqrt(1 - .65 ** 2), 0])
    assert confirm(gallery, detected=accepted, profiles=manual)
    learned = gallery.learned_profiles(manual)
    assert isinstance(learned, LearnedProfiles)
    assert len(learned["Anton"]) == 2
    drifting = face([.4, np.sqrt(1 - .4 ** 2), 0])
    assert np.dot(drifting["embedding"], accepted["embedding"]) > .9
    assert confirm(gallery, detected=drifting, profiles=learned, start=300) is None
    assert sample_count(gallery) == 1
    assert manual == {"Anton": [[1., 0., 0.]]}


def test_active_vectors_bounded_and_invalidated_by_manual_reenrollment(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    anchor = np.zeros(32)
    anchor[0] = 1
    for index in range(1, 21):
        vector = anchor * .8
        vector[index] = .6
        assert gallery.enroll(scene(index), "Anton", face(vector), now=index * 86400)
    learned = gallery.learned_profiles({"Anton": [anchor.tolist()]})
    assert len(learned["Anton"]) == 1 + gallery.ACTIVE_LIMIT
    assert sample_count(gallery) == 20
    other = np.zeros(32)
    other[-1] = 1
    assert gallery.learned_profiles({"Anton": [other.tolist()]}) == {"Anton": [other.tolist()]}
    assert gallery.learned_profiles({}) == {}


def test_profile_cache_invalidates_across_instances(tmp_path, monkeypatch):
    reader, writer = AppearanceGallery(tmp_path), AppearanceGallery(tmp_path)
    manual = {"Anton": [[1, 0, 0]]}
    assert reader.learned_profiles(manual) == manual
    assert writer.enroll(scene(), "Anton", face(), now=10)
    assert len(reader.learned_profiles(manual)["Anton"]) == 2
    monkeypatch.setattr(reader, "_rows", lambda *a, **k: pytest.fail("cached vectors should avoid I/O"))
    cached = reader.learned_profiles(manual)
    cached["Anton"][0][0] = 999
    assert reader.learned_profiles(manual)["Anton"][0][0] == 1


def test_concurrent_connections_cannot_bypass_capture_interval(tmp_path):
    def save(_):
        return confirm(AppearanceGallery(tmp_path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(save, range(4)))
    assert sum(r is not None for r in results) == 1
    assert sample_count(AppearanceGallery(tmp_path)) == 1


def test_rename_preserves_history_and_confirmed_merge_keeps_files(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    assert gallery.enroll(scene(), "Old Name", face(), now=10)
    before = set(tmp_path.rglob("*.jpg"))
    assert gallery.rename("Old Name", "Anton")
    refs = gallery.references("Anton")
    assert refs[0]["name"] == "Anton"
    assert refs[0]["captured_name"] == "Old Name"
    assert not gallery.references("Old Name")
    assert gallery.enroll(scene(2), "John", face([0, 1, 0]), now=20)
    with pytest.raises(ValueError):
        gallery.rename("Anton", "John")
    assert gallery.rename("Anton", "John", allow_merge=True)
    assert gallery.list_people() == ["John"]
    assert sample_count(gallery) == 2
    assert all(p.exists() for p in before)


def test_invalid_name_cannot_escape_archive_and_missing_corrupt_files_are_skipped(tmp_path):
    gallery = AppearanceGallery(tmp_path / "archive")
    sample = gallery.enroll(scene(), "../../Anton", face(), now=10)
    assert sample is not None  # labels never enter filesystem paths
    assert not (tmp_path / "Anton").exists()
    path = gallery.root / sample["face_path"]
    path.write_bytes(b"not an image")
    assert gallery.references("../../Anton") == []
    path.unlink()
    assert gallery.list_people() == []


def test_reference_reader_rejects_database_path_traversal(tmp_path):
    gallery = AppearanceGallery(tmp_path / "archive")
    assert gallery.enroll(scene(), "Anton", face(), now=10)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(scene())
    with sqlite3.connect(gallery.database) as db:
        db.execute("UPDATE samples SET face_path='../outside.jpg'")
    assert gallery.references("Anton") == []


def test_corrupt_database_keeps_voice_path_readers_nonfatal(tmp_path):
    (tmp_path / "gallery.sqlite3").write_bytes(b"bad database")
    gallery = AppearanceGallery(tmp_path)
    assert gallery.references("Anton") == []
    assert gallery.list_people() == []
    assert gallery.learned_profiles({"Anton": [[1, 0, 0]]}) == {"Anton": [[1, 0, 0]]}


def test_corrupt_sample_timestamp_does_not_break_reference_lookup(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    assert gallery.enroll(scene(), "Anton", face(), now=10)
    with sqlite3.connect(gallery.database) as db:
        db.execute("UPDATE samples SET captured_at=?", (1e100,))
    assert gallery.references("Anton") == []


def test_recreated_or_corrected_profile_cannot_reference_previous_person(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    original = gallery.enroll(scene(), "Anton", face(), now=10)
    old_files = set(tmp_path.rglob("*.jpg"))
    assert gallery.references("Anton", profiles={"Anton": [[1, 0, 0]]})
    # Removing enrollment, or reusing its display name for another person,
    # must leave the historical archive intact without leaking its identity.
    assert gallery.references("Anton", profiles={}) == []
    assert gallery.references("Anton", profiles={"Anton": [[0, 1, 0]]}) == []
    assert gallery.list_people(profiles={}) == []
    assert gallery.list_people(profiles={"Anton": [[0, 1, 0]]}) == []
    replacement = gallery.enroll(scene(2), "Anton", face([0, 1, 0]), now=20)
    refs = gallery.references("Anton", profiles={"Anton": [[0, 1, 0]]})
    assert {r["sample_id"] for r in refs} == {replacement["sample_id"]}
    assert original["sample_id"] not in {r["sample_id"] for r in refs}
    assert sample_count(gallery) == 2
    assert all(path.exists() for path in old_files)
    assert gallery.list_people(profiles={"Anton": [[0, 1, 0]]}) == ["Anton"]


def test_reference_revalidation_requires_runner_up_margin_and_manual_anchors(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    automatic = face([.65, np.sqrt(1 - .65 ** 2), 0])
    assert confirm(gallery, detected=automatic)
    learned = gallery.learned_profiles({"Anton": [[1, 0, 0]]})
    # Adaptive matching vectors cannot make another drifting image eligible
    # as an identity reference when it no longer matches manual enrollment.
    drifting = face([.4, np.sqrt(1 - .4 ** 2), 0])
    drift = gallery.enroll(scene(2), "Anton", drifting, now=200)
    refs = gallery.references("Anton", profiles=learned)
    assert drift["sample_id"] not in {r["sample_id"] for r in refs}
    ambiguous = {"Anton": [[1, 0, 0]], "John": [[1, 0, 0]]}
    assert gallery.references("Anton", profiles=ambiguous) == []


def test_malformed_latest_embedding_does_not_block_future_capture(tmp_path):
    gallery = AppearanceGallery(tmp_path)
    assert confirm(gallery)
    previous_files = set(tmp_path.rglob("*.jpg"))
    with sqlite3.connect(gallery.database) as db:
        db.execute("UPDATE samples SET embedding='broken json'")
    assert gallery.references("Anton", profiles={"Anton": [[1, 0, 0]]}) == []
    accepted = confirm(gallery, start=200)
    assert accepted
    assert sample_count(gallery) == 2
    assert all(path.exists() for path in previous_files)
    refs = gallery.references("Anton", profiles={"Anton": [[1, 0, 0]]})
    assert {r["sample_id"] for r in refs} == {accepted["sample_id"]}


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_body_without_usable_face_never_becomes_identity_reference(tmp_path, damage):
    gallery = AppearanceGallery(tmp_path)
    detected = face()
    accepted = confirm(gallery, detected=detected, faces=[detected])
    assert {r["kind"] for r in gallery.references("Anton")} == {"face", "body"}
    path = gallery.root / accepted["face_path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not a jpeg")
    assert (gallery.root / accepted["body_path"]).exists()
    assert gallery.references("Anton", profiles={"Anton": [[1, 0, 0]]}) == []
    assert gallery.references("Anton") == []
    assert gallery.list_people(profiles={"Anton": [[1, 0, 0]]}) == []
