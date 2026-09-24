"""The 3D build: silhouettes, the carve, the mesh and the files it writes."""
from __future__ import annotations

import json

import numpy as np
import pytest

from hub import three_d


def a_photo(height: int = 240, width: int = 96, *, person: tuple[int, int, int, int] | None = None):
    """A plain bright room with a darker person standing in it."""
    image = np.full((height, width, 3), 210, dtype=np.uint8)
    if person is not None:
        left, top, right, bottom = person
        image[top:bottom, left:right] = (120, 40, 10)  # BGR: a blue shirt
    return image


def test_a_silhouette_is_the_person_and_not_the_room():
    image = a_photo(person=(28, 30, 68, 220))
    mask = three_d.silhouette(image)
    assert mask is not None
    rows, columns = np.nonzero(mask)
    assert 20 <= columns.min() <= 40 and 55 <= columns.max() <= 75, "ширина человека"
    assert 20 <= rows.min() <= 45 and 205 <= rows.max() <= 235, "рост человека"
    assert mask.mean() < 0.5, "комната не попала в маску"


def test_a_crop_too_small_or_too_empty_is_refused():
    assert three_d.silhouette(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    assert three_d.silhouette(a_photo(person=None)) is None, "пустая комната — не человек"


def test_a_big_crop_is_segmented_small_and_scaled_back():
    """A build of a thousand frames cannot pay 1.8 s of GrabCut per frame."""
    big = a_photo(height=960, width=400, person=(120, 120, 280, 880))
    assert max(big.shape[:2]) > three_d.MASK_MAX_SIDE, "тест должен попадать в ветку уменьшения"
    mask = three_d.silhouette(big)
    assert mask is not None and mask.shape == big.shape[:2], "маска возвращается в исходном размере"
    rows, columns = np.nonzero(mask)
    assert 80 <= columns.min() <= 160 and 240 <= columns.max() <= 320, "ширина человека"
    assert 80 <= rows.min() <= 200 and 800 <= rows.max() <= 940, "рост человека"


def test_the_head_angle_comes_out_of_five_points():
    looking = [[10.0, 10.0], [30.0, 10.0], [20.0, 20.0], [12.0, 30.0], [28.0, 30.0]]
    assert three_d.yaw_from_keypoints(looking) == pytest.approx(0.0, abs=0.5)
    turned = [[10.0, 10.0], [30.0, 10.0], [28.0, 20.0], [12.0, 30.0], [28.0, 30.0]]
    assert three_d.yaw_from_keypoints(turned) > 20.0, "нос ушёл к глазу — голова повёрнута"
    assert three_d.yaw_from_keypoints([[0, 0], [1, 1]]) is None
    assert three_d.yaw_from_keypoints(looking[:3]) is None, "мало точек"
    assert three_d.yaw_from_keypoints([[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]]) is None, \
        "точки в одной точке — не лицо"


def _a_body(*, span: float = 60.0, nose: bool = True, nose_offset: float = 0.0,
            eyes: bool = True, ears: tuple[int, ...] = ()) -> dict:
    """A 17-point skeleton 70 px from the shoulders to the hips, COCO order.

    The shoulder span is what the angle is read from, so the test drives it
    directly: 60 px of span over 70 px of torso is a person squarely facing the
    camera, and a narrower line is somebody who turned. The face is switched on
    and off the way a camera sees it: from the front the nose and the eyes are
    there, from the side one ear, from behind both ears and no face at all.
    """
    points = np.zeros((17, 2), dtype=np.float32)
    confidence = np.full(17, 0.9, dtype=np.float32)
    confidence[0:5] = 0.1                      # nothing of the face is sure yet
    points[5] = (50.0 - span / 2.0, 80.0)      # left shoulder
    points[6] = (50.0 + span / 2.0, 80.0)      # right shoulder
    points[11] = (36.0, 150.0)                 # left hip
    points[12] = (64.0, 150.0)                 # right hip
    if nose:
        points[0] = (50.0 + nose_offset, 60.0)
        confidence[0] = 0.9
    if eyes:
        points[1], points[2] = (45.0, 62.0), (55.0, 62.0)
        confidence[1] = confidence[2] = 0.9
    if 3 in ears:
        points[3] = (38.0, 62.0)
        confidence[3] = 0.9
    if 4 in ears:
        points[4] = (62.0, 62.0)
        confidence[4] = 0.9
    return {"points": points, "confidence": confidence}


def test_the_body_gives_the_angle_when_there_is_no_face():
    square_on, where = three_d.yaw_from_body(_a_body())
    assert where == "front" and square_on == pytest.approx(0.0, abs=2.0), \
        "плечи во всю ширину — человек смотрит в камеру"
    turned, where = three_d.yaw_from_body(_a_body(span=40.0, nose_offset=8.0))
    assert where == "front" and turned > 30.0, "плечи уже — человек повернулся, нос вправо"
    other, _ = three_d.yaw_from_body(_a_body(span=40.0, nose_offset=-8.0))
    assert other == pytest.approx(-turned, abs=1.0), "поворот в другую сторону — тот же угол со знаком минус"


def test_the_side_and_the_back_are_read_from_the_ears():
    # Only the right ear shows: the person turned towards the right of the frame.
    side, where = three_d.yaw_from_body(_a_body(span=20.0, nose=False, eyes=False, ears=(4,)))
    assert where == "side" and side > 55.0, "узкая линия плеч с одним ухом — профиль"
    flipped, _ = three_d.yaw_from_body(_a_body(span=20.0, nose=False, eyes=False, ears=(3,)))
    assert flipped == pytest.approx(-side, abs=1.0), "видно другое ухо — поворот в другую сторону"
    # Both ears and no face at all: the person stands with their back to the camera.
    back, where = three_d.yaw_from_body(_a_body(nose=False, eyes=False, ears=(3, 4)))
    assert where == "back" and abs(back) == pytest.approx(180.0, abs=5.0), \
        "спина — это разворот, а не отсутствие угла"
    away, where = three_d.yaw_from_body(_a_body(span=40.0, nose=False, eyes=False, ears=(3, 4)))
    assert where == "back" and 90.0 < abs(away) < 180.0, "спина вполоборота — меньше 180°"


def test_a_body_angle_needs_a_body():
    assert three_d.yaw_from_body(None) == (None, "")
    assert three_d.yaw_from_body({"points": np.zeros((17, 2)), "confidence": np.zeros(17)}) \
        == (None, ""), "нет плеч — нет угла"
    assert three_d.yaw_from_body(_a_body(nose=False, eyes=False)) == (None, ""), \
        "ни лица, ни ушей — угол не выдумывается"
    empty = {"points": np.zeros((4, 2), dtype=np.float32),
             "confidence": np.zeros(4, dtype=np.float32)}
    assert three_d.yaw_from_body(empty) == (None, ""), "скелета нет — угла нет"


def test_the_body_angle_does_not_care_how_far_away_the_person_stood():
    near = _a_body(span=34.0, nose_offset=6.0)
    far = {"points": near["points"] * 2.0 + np.array((90.0, 40.0), dtype=np.float32),
           "confidence": near["confidence"]}
    first, _ = three_d.yaw_from_body(near)
    second, _ = three_d.yaw_from_body(far)
    assert first is not None and second == pytest.approx(first, abs=1.0), \
        "те же плечи вдвое дальше — тот же угол"



def _bar_mask(shape=(200, 100), *, left=40, right=60, top=20, bottom=190):
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def test_the_carve_keeps_what_every_view_agrees_on():
    front = _bar_mask()
    side = _bar_mask()
    grid = three_d.carve([(front, 0.0), (side, 90.0)], resolution=48)
    assert grid.any(), "колонна должна выжить в обоих ракурсах"
    rows, columns, depths = np.nonzero(grid)
    assert rows.min() < 5 and rows.max() > 40, "тело стоит во весь рост"
    assert columns.max() - columns.min() <= 6 and depths.max() - depths.min() <= 6, \
        "узкая колонна, а не куб"

    narrow = _bar_mask(left=42, right=44)
    carved = three_d.carve([(front, 0.0), (narrow, 90.0)], resolution=48)
    assert carved.sum() < grid.sum(), "пересечение не может стать больше одной проекции"
    assert not three_d.carve([(front, 0.0), (np.zeros_like(front), 40.0)]).any()


def test_views_are_chosen_by_angle_not_by_time():
    views = [three_d.View(sample_id=f"s{index}", yaw_deg=yaw)
             for index, yaw in enumerate([0.0, 4.0, 6.0, 40.0, 43.0, 80.0])]
    chosen = three_d.select_views(views, limit=8, spread_deg=12.0)
    # One frame per angle group (the middle of each group), 12 degrees apart.
    assert [view.sample_id for view in chosen] == ["s1", "s4", "s5"]
    assert [view.yaw_deg for view in chosen] == [4.0, 43.0, 80.0]
    assert len(three_d.select_views(views, limit=2)) == 2
    assert three_d.select_views([], limit=4) == []


def test_the_mesh_and_its_files_are_written(tmp_path):
    grid = three_d.carve([(_bar_mask(), 0.0), (_bar_mask(), 60.0)], resolution=48)
    vertices, faces = three_d.mesh_of(grid, resolution=48)
    assert len(vertices) > 100 and len(faces) > 100
    assert vertices[:, 0].min() >= -0.01 and vertices[:, 0].max() <= 1.01, "рост в 0..1"
    colours = three_d.colour_of(vertices, [(a_photo(), _bar_mask(), 0.0)])
    assert colours.shape == (len(vertices), 3) and colours.dtype == np.uint8

    three_d.save_model(tmp_path / "model", vertices, faces, colours, meta={"person": "Test"})
    payload = (tmp_path / "model" / "model.bin").read_bytes()
    meta = json.loads((tmp_path / "model" / "model.json").read_text(encoding="utf-8"))
    assert len(payload) == 6 * meta["points"] == 6 * len(vertices)
    assert meta["triangles"] == len(faces) and meta["meta"]["person"] == "Test"
    header = (tmp_path / "model" / "model.ply").read_bytes().split(b"end_header\n")[0]
    assert header.startswith(b"ply\nformat binary_little_endian 1.0")
    assert f"element vertex {len(vertices)}".encode() in header


def test_an_empty_hull_is_an_error_not_an_empty_model(tmp_path):
    vertices = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        three_d.save_model(tmp_path / "model", vertices, np.zeros((0, 3)), np.zeros((0, 3)),
                           meta={})


def test_colours_come_from_the_frame_that_sees_the_vertex():
    image = a_photo(person=(28, 30, 68, 220))
    mask = _bar_mask(left=28, right=68, top=30, bottom=220)
    vertices = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)  # mid-height, centred
    painted = three_d.colour_of(vertices, [(image, mask, 0.0)])
    red, green, blue = (int(value) for value in painted[0])
    assert blue > 90 and blue > red, "цвет — из фотографии (синяя рубашка), не заглушка"
    grey = three_d.colour_of(vertices, [])
    assert tuple(int(value) for value in grey[0]) == (150, 150, 150)


# --- the photographic build -------------------------------------------------


def _skeleton(*, legs: float = 0.9, hips: float = 0.9, head: float = 0.9,
              box=(0, 0, 100, 300)) -> dict:
    """COCO keypoints with chosen confidences: 0 nose ... 16 right ankle."""
    points = np.zeros((17, 2), dtype=np.float32)
    confidence = np.full(17, 0.9, dtype=np.float32)
    confidence[0:3] = head
    confidence[11:13] = hips
    confidence[13:17] = legs
    points[:, 0] = np.linspace(40, 60, 17)
    points[:, 1] = np.linspace(20, 300, 17)
    return {"points": points, "confidence": confidence, "box": np.asarray(box, dtype=np.float32)}


def test_the_skeleton_tells_standing_from_sitting():
    assert three_d.standing_from_skeleton(_skeleton()) == (True, "ok")
    assert three_d.standing_from_skeleton(_skeleton(legs=0.05))[1] == "legs not visible (sitting or cut off)"
    assert three_d.standing_from_skeleton(_skeleton(hips=0.1))[1] == "only the upper body"
    assert three_d.standing_from_skeleton(_skeleton(head=0.1))[1] == "head not visible"
    assert three_d.standing_from_skeleton(_skeleton(box=(0, 0, 300, 200)))[1] == "not standing (wide body)"
    assert three_d.standing_from_skeleton(None)[1] == "no person found"


def test_the_body_frame_measures_the_person_not_the_crop():
    pose = _skeleton()
    frame = three_d.body_frame(pose)
    assert frame is not None
    centre_x, feet_y, height_px = frame
    assert 40 <= centre_x <= 60, "центр — по бёдрам"
    assert feet_y == pytest.approx(300, abs=1.0), "ноги стоят на полу кадра"
    assert height_px == pytest.approx(280, abs=1.0), "рост — от головы до ног"
    assert three_d.body_frame({"points": np.zeros((17, 2)), "confidence": np.zeros(17),
                               "box": np.zeros(4)}) is None


def test_a_frame_is_placed_by_the_skeleton_when_it_is_known():
    image = a_photo(person=(28, 30, 68, 220))
    mask = _bar_mask((240, 96), left=28, right=68, top=30, bottom=220)
    plain = three_d.frame_cloud(image, mask, 0.0, None, step=6)
    placed = three_d.frame_cloud(image, mask, 0.0, None, step=6, frame=(70.0, 220.0, 190.0))
    assert plain is not None and placed is not None
    # The skeleton says the centre is at x=70 while the mask's own centre is 48:
    # the points move by exactly that difference over the skeleton's height.
    assert float(placed[:, 0].mean()) == pytest.approx((48 - 70) / 190.0, abs=0.02)
    assert float(placed[:, 0].mean()) < float(plain[:, 0].mean()), "сдвиг к скелету"
    assert placed[:, 1].max() <= 1.01 and placed[:, 1].min() >= -0.01


def test_a_frame_becomes_points_of_its_own_pixels():
    image = a_photo(person=(28, 30, 68, 220))
    mask = _bar_mask((240, 96), left=28, right=68, top=30, bottom=220)
    depth = np.zeros(image.shape[:2], dtype=np.float32)
    depth[mask] = np.linspace(0.0, 1.0, int(mask.sum()), dtype=np.float32)

    straight = three_d.frame_cloud(image, mask, 0.0, depth, step=1)
    assert straight is not None and len(straight) == int(mask.sum())
    assert straight[:, 1].min() >= -0.01 and straight[:, 1].max() <= 1.01, "рост в 0..1"
    assert abs(float(straight[:, 0].mean())) < 0.02, "человек стоит по центру"
    blue = straight[:, 5].mean()
    assert blue > 90 and blue > straight[:, 3].mean(), "цвета — пиксели фотографии"
    assert straight[:, 2].std() > 0.01, "глубина даёт рельеф, а не плоскость"

    turned = three_d.frame_cloud(image, mask, 90.0, depth, step=1)
    assert abs(float(turned[:, 2].mean())) - abs(float(straight[:, 2].mean())) > -0.02
    assert not np.allclose(straight[:, :3], turned[:, :3]), "кадр разворачивается в свой ракурс"
    assert three_d.frame_cloud(image, mask[:20, :20], 0.0, None) is None, "мало пикселей"


def test_clouds_merge_into_cells_instead_of_stacking():
    image = a_photo(person=(28, 30, 68, 220))
    mask = _bar_mask((240, 96), left=28, right=68, top=30, bottom=220)
    cloud = three_d.frame_cloud(image, mask, 0.0, None, step=4)
    once = three_d.merge_clouds([cloud], voxel=three_d.CLOUD_VOXEL)
    twice = three_d.merge_clouds([cloud, cloud], voxel=three_d.CLOUD_VOXEL)
    assert len(once) < len(cloud), "соседние пиксели попадают в одну ячейку"
    assert len(twice) == len(once), "два одинаковых кадра не удваивают облако"
    assert three_d.merge_clouds([]).shape == (0, 6)
    coarse = three_d.merge_clouds([cloud], voxel=1.0)
    assert 0 < len(coarse) < len(once), "крупная ячейка — грубее и меньше"


def test_training_frames_come_out_of_the_archive_index(tmp_path):
    import sqlite3

    root = tmp_path / "training_archive"
    event = root / "2026-09-20" / "Anton--abc" / "events" / "001836-33b9"
    event.mkdir(parents=True)
    (event / "body.png").write_bytes(b"\x89PNG")
    (event / "original.jpg").write_bytes(b"\xff\xd8\xff")
    (event / "enrollment.wav").write_bytes(b"RIFF")
    connection = sqlite3.connect(str(root / "index.sqlite3"))
    connection.executescript(
        "CREATE TABLE identities(id TEXT PRIMARY KEY, name TEXT NOT NULL);"
        "CREATE TABLE events(person_id TEXT, kind TEXT, captured_at REAL, record TEXT);")
    connection.execute("INSERT INTO identities VALUES('p-anton', 'Anton')")
    connection.execute("INSERT INTO events VALUES(?,?,?,?)", (
        "p-anton", "appearance", 1_789_881_516.0, json.dumps({
            "kind": "appearance", "event_path": "2026-09-20/Anton--abc/events/001836-33b9/event.json",
            "files": ["body.png", "original.jpg", "enrollment.wav"]})))
    connection.execute("INSERT INTO events VALUES(?,?,?,?)", (
        "p-anton", "conversation", 1_789_881_600.0, json.dumps({
            "kind": "conversation", "event_path": "2026-09-20/Anton--abc/events/001900-x/event.json",
            "files": ["dialog.json"]})))
    connection.commit()
    connection.close()

    found = three_d.training_frames(root, "Anton", limit=50)
    assert [path.split("events")[-1] for path, _stamp in found] == ["\\001836-33b9\\body.png",
                                                                   "\\001836-33b9\\original.jpg"]
    assert three_d.training_frames(root, "Nobody", limit=5) == []
    assert three_d.training_frames(tmp_path / "nowhere", "Anton") == []


# --- поза: голова, руки и ноги ----------------------------------------------


def _posed_skeleton(*, arms: str = "down", legs: str = "straight", head: str = "up") -> dict:
    """A skeleton whose head, arms and legs can be moved on purpose."""
    points = np.zeros((17, 2), dtype=np.float32)
    points[0] = (50, 40)          # nose
    points[1:3] = [(45, 38), (55, 38)]
    points[5:7] = [(40, 80), (60, 80)]     # shoulders
    points[9:11] = [(38, 130), (62, 130)] if arms == "down" else [(20, 70), (80, 70)]
    points[11:13] = [(45, 150), (55, 150)]  # hips
    points[13:15] = [(45, 220), (55, 220)]  # knees
    points[15:17] = [(45, 290), (55, 290)]  # ankles
    if legs == "crossed":
        points[15:17] = [(30, 290), (70, 290)]
    if legs == "sitting":
        points[13:15] = [(45, 170), (55, 170)]
        points[15:17] = [(55, 160), (65, 160)]
    if head == "down":
        points[0] = (50, 150)
        points[1:3] = [(45, 148), (55, 148)]
    confidence = np.full(17, 0.9, dtype=np.float32)
    return {"points": points, "confidence": confidence,
            "box": np.asarray((20, 20, 80, 300), dtype=np.float32)}


def test_the_head_has_to_be_held_up():
    assert three_d.head_above_shoulders(_posed_skeleton()) is True
    assert three_d.head_above_shoulders(_posed_skeleton(head="down")) is False
    assert three_d.head_above_shoulders(None) is False


def test_the_pose_descriptor_catches_the_arms_and_the_legs():
    base = three_d.pose_shape(_posed_skeleton())
    assert base is not None
    assert three_d.pose_distance(base, three_d.pose_shape(_posed_skeleton())) == pytest.approx(0.0)
    assert three_d.pose_distance(base, three_d.pose_shape(_posed_skeleton(arms="up"))) > 0.2
    crossed = three_d.pose_distance(base, three_d.pose_shape(_posed_skeleton(legs="crossed")))
    sitting = three_d.pose_distance(base, three_d.pose_shape(_posed_skeleton(legs="sitting")))
    assert 0.0 < crossed < sitting, "шаг ноги меняет позу меньше, чем посадка"
    assert sitting > three_d.POSE_DISTANCE, "сидящая поза — это другая поза"
    assert three_d.pose_distance(base, None) == 1.0


def test_the_pose_descriptor_ignores_where_the_person_stands_in_the_frame():
    """Position and size are taken out, so a frame is judged by its pose alone."""
    moved = _posed_skeleton()
    moved["points"] = moved["points"] * 2.0 + np.array((300.0, 250.0), dtype=np.float32)
    assert three_d.pose_distance(three_d.pose_shape(_posed_skeleton()),
                                 three_d.pose_shape(moved)) == pytest.approx(0.0, abs=1e-5)
