"""A 3D build of one person out of the photographs the archive already keeps.

There is no depth sensor and no photogrammetry rig in a dorm room. What there is:
one fixed camera and a person who turned in front of it, which is exactly what
the appearance archive recorded (front, side, back — the owner asked for those
samples himself). This module turns that into geometry:

1. every selected frame contributes a **silhouette** — the body crop, cut out
   from its background by OpenCV's GrabCut — and an **angle** — the yaw of the
   face in the same frame;
2. the intersection of the back-projected silhouettes is the **visual hull** of
   the person, a real 3D body carved out of the photographs;
3. the hull becomes a **mesh** with marching cubes and is **coloured** by
   projecting every vertex back into the frames that saw it.

What this is not: a scanned human. The result is a blocky statue with the right
outline and the right colours, not folds, hair or a face. A sharp model needs a
generative image-to-3D network or a proper multi-camera capture; the honest
status of both is in ``docs/ADMIN_PANEL.md``.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: The pose model comes from ultralytics, which replaces ``PIL.Image.open`` with
#: a version that asks pip to install ``pi-heif`` when a picture will not open -
#: a blocking network call in the middle of a build (or a hub turn). Rowan never
#: installs packages while it runs, so the automatic install stays off; the
#: package itself is installed.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

log = logging.getLogger("jarvis.server.three_d")

#: How many voxels the tallest side of the hull is cut into.
RESOLUTION = 96
#: Physical half-width/depth of the carving box, in person-heights. A body is
#: about 0.45 m wide and 0.3 m deep; 0.4 of a 1.75 m person is generous and
#: keeps the arms of a wide pose inside the box.
HALF_EXTENT = 0.40
#: GrabCut needs a rectangle that is surely background and one that is surely
#: the person: the body crop is tight, so the margin is small.
MASK_MARGIN = 0.04
#: GrabCut is the slowest single step of a build (measured: 1.8 s on a full
#: body crop, which is hours over a couple of thousand frames), and its mask only
#: has to answer "is this pixel the person". Above this long side the crop is
#: segmented smaller and the mask is scaled back up, which costs a fraction.
MASK_MAX_SIDE = 480
#: The depth model that turns a photograph into relief. Small on purpose: 25M
#: parameters, ~99 MB, a fraction of a second per frame on the CPU, and it never
#: fights the LLM for the GPU.
DEPTH_MODEL_DIR = "data/models/depth-anything-v2-small"
#: How thick a person is, in person-heights, front to back. A body is about
#: 0.35 m deep; 0.22 of 1.75 m is that. Monocular depth carries relative relief
#: only, so the relief is scaled to this rather than measured in metres.
BODY_THICKNESS = 0.22
#: Voxel size of the merged cloud, in person-heights (6 mm of a 1.75 m person).
CLOUD_VOXEL = 0.006
#: How many points one build may keep, so the panel can still render it.
CLOUD_LIMIT = 1_500_000
#: What makes a frame good enough to remember how somebody looks: the whole
#: person in the picture, standing (tall, not a desk-wide blob), big enough in
#: the frame and not blurred. The owner asked for exactly this after a build
#: mixed him sitting at the desk with him standing in the room.
MIN_HEIGHT_SHARE = 0.35
MIN_ASPECT = 1.6
MIN_SHARPNESS = 25.0
#: How close two frames' clothes have to be to count as the same look.
LOOK_DISTANCE = 0.30
#: YOLO11-pose weights used to see the skeleton of the person in a frame. The
#: same family of model the room PCs already run for detection.
POSE_WEIGHTS = "data/models/yolo11x-pose.pt"
#: A keypoint this confident counts as seen (COCO order: 0 nose ... 16 right ankle).
POSE_MIN_CONFIDENCE = 0.5
#: How different two poses may be and still belong to the same build. The unit is
#: the torso: 0.15 means a joint sits a sixth of a torso away from where the
#: chosen pose has it — a hand on the hip or a leg crossed is much further.
POSE_DISTANCE = 0.15
#: The head has to sit above the shoulders, this many torsos up.
HEAD_MIN_ABOVE = 0.25
HEAD_MAX_ABOVE = 2.5
#: Shoulder span over torso length for somebody squarely facing the camera. Both
#: numbers come from the skeleton itself, so the ratio does not care how far away
#: the person stood or how big the crop is; a body seen from the side projects a
#: much narrower shoulder line, and that narrowing is the angle.
BODY_FRONT_RATIO = 0.85
#: Below this many degrees a body is square-on and the nose offset is noise.
BODY_SQUARE_DEG = 8.0


@dataclass(frozen=True)
class View:
    """One photograph used by the carve: where it is and from which angle."""

    sample_id: str
    yaw_deg: float
    path: str = ""
    captured_at: str = ""


def decode(path: str | Path) -> np.ndarray | None:
    """A BGR image, or ``None`` when the file is not a readable picture."""
    try:
        import cv2

        image = cv2.imread(str(path))
    except Exception:  # a broken file must not stop the build
        log.debug("Could not read %s", path, exc_info=True)
        return None
    return image if image is not None and image.size else None


def silhouette(image: np.ndarray) -> np.ndarray | None:
    """The person inside one body crop, as a boolean mask.

    GrabCut is given the crop itself as the rectangle: the border is background,
    the middle is the person, and the colour models separate them. A crop that
    yields an implausible mask (too small, or filling everything) is refused
    rather than carved. A big crop is segmented at ``MASK_MAX_SIDE`` and the mask
    scaled back up, because the mask is a yes/no answer about pixels, not a
    measurement, and a thousand frames have to fit in one night.
    """
    import cv2

    height, width = image.shape[:2]
    if min(height, width) < 24:
        return None
    scale = 1.0
    small = image
    if max(height, width) > MASK_MAX_SIDE:
        scale = MASK_MAX_SIDE / float(max(height, width))
        small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                           interpolation=cv2.INTER_AREA)
    sheight, swidth = small.shape[:2]
    margin_x, margin_y = int(swidth * MASK_MARGIN), int(sheight * MASK_MARGIN)
    rectangle = (margin_x, margin_y, swidth - 2 * margin_x, sheight - 2 * margin_y)
    mask = np.zeros((sheight, swidth), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rectangle, background, foreground, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        log.debug("GrabCut refused a %dx%d crop", swidth, sheight)
        return None
    person = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    # Keep the largest connected blob: a lamp or a poster on the wall is not him.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(person.astype(np.uint8), 8)
    if count <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    person = labels == biggest
    area = float(person.sum()) / float(sheight * swidth)
    if not 0.06 <= area <= 0.97:
        return None
    if scale != 1.0:
        person = cv2.resize(person.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    return person


def yaw_from_keypoints(keypoints: Any) -> float | None:
    """The face's yaw in degrees from five landmarks, no 3D model needed.

    The nose sits between the eyes when the person looks at the camera and moves
    towards one eye as the head turns. The usual approximation is the nose
    offset over half the eye distance, taken through ``arcsin``.
    """
    try:
        points = np.asarray(keypoints, dtype=np.float64).reshape(5, 2)
    except (TypeError, ValueError):
        return None
    left_eye, right_eye, nose = points[0], points[1], points[2]
    eye_span = float(np.linalg.norm(left_eye - right_eye))
    if not math.isfinite(eye_span) or eye_span < 4.0:
        return None
    middle = (left_eye + right_eye) / 2.0
    offset = float(nose[0] - middle[0])
    ratio = max(-1.0, min(1.0, offset / (eye_span / 2.0)))
    return math.degrees(math.asin(ratio))


def yaw_from_body(pose: dict[str, Any] | None) -> tuple[float | None, str]:
    """The body's angle in the same degrees as :func:`yaw_from_keypoints`.

    The owner's complaint was that a person who walked past the camera turning
    away was thrown out of a build with "no face to read the angle from": the
    angle used to come from the face alone, and a back of the head has no eyes
    to measure. The skeleton of the same frame carries the answer without a
    single new model: the shoulder line shrinks towards the torso length as
    somebody turns, and which side of the head is towards the camera says which
    way they turned.

    Returns ``(degrees, where the angle came from)``: ``front`` when the nose
    and an eye were visible, ``back`` when both ears were and the face was not,
    ``side`` when only one ear showed. 0 is squarely facing the camera, the sign
    follows :func:`yaw_from_keypoints` (positive: the nose towards the right of
    the frame), and a person seen from behind is near 180. ``(None, "")`` means
    the skeleton is too thin to say - a build then skips the frame, as before.
    """
    if not pose:
        return None, ""
    points = np.asarray(pose.get("points"), dtype=float)
    confidence = np.asarray(pose.get("confidence"), dtype=float)
    if points.ndim != 2 or len(points) < 17 or len(confidence) < 17:
        return None, ""
    seen = confidence >= POSE_MIN_CONFIDENCE
    shoulders = points[5:7][seen[5:7]]          # COCO: 5 left, 6 right
    hips = points[11:13][seen[11:13]]           # 11 left, 12 right
    if len(shoulders) < 2 or not len(hips):
        return None, ""
    torso = float(np.linalg.norm(shoulders.mean(axis=0) - hips.mean(axis=0)))
    span = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    if not math.isfinite(torso) or torso < 8.0:
        return None, ""
    turned = math.degrees(math.acos(max(0.0, min(1.0, span / torso / BODY_FRONT_RATIO))))
    nose, eyes = bool(seen[0]), [index for index in (1, 2) if seen[index]]
    ears = [index for index in (3, 4) if seen[index]]   # 3 left, 4 right
    if nose and eyes:
        middle = float(points[eyes].mean(axis=0)[0])
        sign = 1.0 if float(points[0][0]) >= middle else -1.0
        if turned < BODY_SQUARE_DEG:
            sign = 1.0                          # square-on: the offset is noise
        return sign * turned, "front"
    if len(ears) == 1:
        # Facing the camera, the ear on the side the nose moves towards hides:
        # only the other one is left visible, which is the turn's sign.
        sign = 1.0 if ears[0] == 4 else -1.0
        return sign * turned, "side"
    if len(ears) == 2 and not nose:
        # Both ears and no face: seen from behind. A body squarely away projects
        # the same shoulder span as one squarely towards the camera, so the
        # angle is the mirror of the turn the ears show.
        ears_middle = float(points[[3, 4]].mean(axis=0)[0])
        sign = 1.0 if ears_middle >= float(shoulders[:, 0].mean()) else -1.0
        angle = 180.0 - sign * turned
        return (angle - 360.0 if angle > 180.0 else angle), "back"
    return None, ""


def face_yaw(image: np.ndarray, engine: Any, path: str = "") -> float | None:
    """The yaw of the face in one body crop, or ``None`` when there is no face."""
    if engine is None:
        return None
    try:
        import cv2

        encoded = cv2.imencode(".jpg", image)[1].tobytes()
        faces = engine.located_faces(encoded)
    except Exception:
        log.debug("Face pose failed for %s", path, exc_info=True)
        return None
    if not faces:
        return None
    face = faces[0]
    pose = face.get("pose")
    if isinstance(pose, (list, tuple)) and len(pose) == 3:
        try:
            value = float(pose[1])
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            return value
    return yaw_from_keypoints(face.get("landmarks"))


def select_views(candidates: Sequence[View], limit: int = 16,
                 spread_deg: float = 12.0) -> list[View]:
    """Keep the frames that cover the widest range of angles, not the newest.

    ``candidates`` carry their own yaw. Angles closer than ``spread_deg`` are the
    same point of view for a carve, so only the sharpest-looking frame of each
    group is kept, and those are then thinned to ``limit`` evenly across the
    range that was actually captured.
    """
    if limit < 1 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda view: view.yaw_deg)
    groups: list[list[View]] = []
    for view in ordered:
        if groups and abs(view.yaw_deg - groups[-1][-1].yaw_deg) < spread_deg:
            groups[-1].append(view)
        else:
            groups.append([view])
    picked = [group[len(group) // 2] for group in groups]
    if len(picked) <= limit:
        return picked
    step = len(picked) / float(limit)
    return [picked[int(index * step)] for index in range(limit)]


def carve(masks: Sequence[tuple[np.ndarray, float]], *, resolution: int = RESOLUTION,
          half_extent: float = HALF_EXTENT) -> np.ndarray:
    """The visual hull: voxels that every silhouette sees as the person.

    Each mask is the person's silhouette in one frame, ``yaw`` its angle in
    degrees. Voxels are carved in person-heights: ``y`` runs 0 (feet) to 1 (top),
    ``x`` and ``z`` span ``±half_extent`` around the body's centre.
    """
    height = max(8, int(resolution))
    width = depth = max(8, int(round(height * half_extent)))
    y = (np.arange(height) + 0.5) / height
    x = (np.arange(width) + 0.5) / width * (2 * half_extent) - half_extent
    z = (np.arange(depth) + 0.5) / depth * (2 * half_extent) - half_extent
    grid = np.ones((height, width, depth), dtype=bool)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    for mask, yaw in masks:
        if mask is None or not mask.any():
            return np.zeros_like(grid)
        rows, columns = np.nonzero(mask)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
        person_height = max(4, bottom - top + 1)
        yaw_rad = math.radians(float(yaw))
        # The person's own width in this frame is what the silhouette shows.
        along = (xx[None, :, :] * math.cos(yaw_rad) + zz[None, :, :] * math.sin(yaw_rad))
        columns_px = (left + right) / 2.0 + along * person_height
        rows_px = (bottom - y * person_height)[:, None, None]
        valid = ((columns_px >= -1) & (columns_px <= mask.shape[1])
                 & (rows_px >= -1) & (rows_px <= mask.shape[0]))
        column_index = np.clip(np.rint(columns_px).astype(np.int64), 0, mask.shape[1] - 1)
        row_index = np.clip(np.rint(rows_px).astype(np.int64), 0, mask.shape[0] - 1)
        seen = mask[row_index, column_index] & valid
        # ``seen`` is (height, width, depth) already: the x/z grid is its last two axes.
        grid &= seen
        if not grid.any():
            break
    return grid


def _smoothed(grid: np.ndarray) -> np.ndarray:
    """One 3x3x3 opening, so single voxels of noise do not become spikes."""
    try:
        from scipy import ndimage
    except Exception:  # scipy is present in this deployment; stay usable without it
        return grid
    return ndimage.binary_opening(grid, structure=np.ones((3, 3, 3)))


def mesh_of(grid: np.ndarray, *, resolution: int = RESOLUTION,
            half_extent: float = HALF_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of a carved hull, in person-heights."""
    from skimage import measure

    solid = _smoothed(grid)
    if not solid.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    padded = np.pad(solid.astype(np.float32), 1)
    vertices, faces, _normals, _values = measure.marching_cubes(padded, level=0.5)
    vertices -= 1.0  # undo the padding
    height, width, depth = solid.shape
    vertices[:, 0] = (vertices[:, 0] + 0.5) / height * 1.0
    vertices[:, 1] = (vertices[:, 1] + 0.5) / width * (2 * half_extent) - half_extent
    vertices[:, 2] = (vertices[:, 2] + 0.5) / depth * (2 * half_extent) - half_extent
    return vertices.astype(np.float32), faces.astype(np.int64)


def colour_of(vertices: np.ndarray, views: Iterable[tuple[np.ndarray, np.ndarray, float]],
              *, half_extent: float = HALF_EXTENT) -> np.ndarray:
    """Sample every vertex from the frames that see it, and average the colours.

    ``views`` are ``(image, mask, yaw)``. A vertex no frame colours comes back
    as mid-grey: the hull's own shape is still true, only its paint is missing.
    """
    colours = np.zeros((len(vertices), 3), dtype=np.float32)
    votes = np.zeros(len(vertices), dtype=np.float32)
    for image, mask, yaw in views:
        if mask is None or not mask.any():
            continue
        rows, columns = np.nonzero(mask)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
        person_height = max(4, bottom - top + 1)
        yaw_rad = math.radians(float(yaw))
        along = vertices[:, 1] * math.cos(yaw_rad) + vertices[:, 2] * math.sin(yaw_rad)
        columns_px = (left + right) / 2.0 + along * person_height
        rows_px = bottom - vertices[:, 0] * person_height
        valid = ((columns_px >= 0) & (columns_px <= mask.shape[1] - 1)
                 & (rows_px >= 0) & (rows_px <= mask.shape[0] - 1))
        if not valid.any():
            continue
        column_index = np.clip(np.rint(columns_px).astype(np.int64), 0, mask.shape[1] - 1)
        row_index = np.clip(np.rint(rows_px).astype(np.int64), 0, mask.shape[0] - 1)
        seen = valid & mask[row_index, column_index]
        if not seen.any():
            continue
        sampled = image[row_index[seen], column_index[seen]][:, ::-1].astype(np.float32)
        colours[seen] += sampled
        votes[seen] += 1.0
    painted = votes > 0
    colours[painted] /= votes[painted][:, None]
    colours[~painted] = 150.0
    return colours.astype(np.uint8)


# --- the photographic build: relief from a depth model ----------------------


#: The loaded pose model and the device it was loaded on: a build may ask for the
#: card and a later one for the CPU, so the model is reloaded when that changes.
_POSE: Any = None
_POSE_DEVICE: str = ""
#: Same for the depth model.
_DEPTH_DEVICE: str = ""


def runtime_device(preferred: str = "auto") -> str:
    """``cuda`` when this machine can and the caller allows it, else ``cpu``.

    The build uses the GPU when it is asked to: a pose pass that costs 200 ms on
    the CPU costs a few on the card, and there are thousands of frames. The hub
    itself keeps the GPU for the rooms, so the default stays ``auto`` and the
    panel offers the choice.
    """
    if preferred in {"cpu", "cuda"}:
        return preferred if preferred == "cpu" else ("cuda" if _cuda_available() else "cpu")
    return "cuda" if _cuda_available() else "cpu"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pose_model(weights: str = POSE_WEIGHTS, device: str = "auto") -> Any:
    """YOLO11-pose, loaded once, on the CPU.

    The hub's GPU belongs to the rooms (the LLM and Whisper); the skeleton of one
    frame costs a fraction of a second on the CPU, and a build is a background
    command, not a service.
    """
    global _POSE, _POSE_DEVICE
    resolved = runtime_device(device)
    if _POSE is not None and _POSE_DEVICE == resolved:
        return _POSE
    _POSE = None
    if _POSE is not None:
        return _POSE
    path = Path(weights)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        log.warning("No pose weights at %s: frames are judged by silhouette only", path)
        return None
    try:
        from ultralytics import YOLO

        _POSE = YOLO(str(path))
        _POSE_DEVICE = resolved
    except Exception:
        log.warning("YOLO-pose is unavailable: frames are judged by silhouette only",
                    exc_info=True)
        _POSE = False
    return _POSE


def skeleton(image: np.ndarray, model: Any = None) -> dict[str, Any] | None:
    """The skeleton of the biggest person in one frame, or ``None``.

    ``points`` are the COCO keypoints in pixels, ``confidence`` their scores,
    ``box`` the person's box. The model that made them is the one the room PCs
    already run for detection, so nothing new has to be trusted.
    """
    weights = model if model is not None else pose_model()
    if not weights or image is None or not getattr(image, "size", 0):
        return None
    try:
        result = weights.predict(image, verbose=False,
                                 device=_POSE_DEVICE or "cpu", conf=0.35)[0]
    except Exception:
        log.debug("The pose model failed on one frame", exc_info=True)
        return None
    if result.keypoints is None or not len(result.keypoints):
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    points = result.keypoints.xy.cpu().numpy()
    raw = result.keypoints.conf
    confidence = raw.cpu().numpy() if raw is not None else np.ones(points.shape[:2],
                                                                  dtype=np.float32)
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    best = int(np.argmax(areas))
    return {"points": points[best], "confidence": confidence[best], "box": boxes[best]}


def standing_from_skeleton(pose: dict[str, Any] | None,
                           min_confidence: float = POSE_MIN_CONFIDENCE) -> tuple[bool, str]:
    """Does this skeleton stand up, whole, with the legs in the picture?

    The owner's complaint was that a build mixed him sitting at the desk with him
    standing in the room. A silhouette cannot tell those apart (the desk is part
    of the blob), but hips and legs can: sitting hides or bends them, and a crop
    that stops at the chest has no ankles at all.
    """
    if not pose:
        return False, "no person found"
    confidence = np.asarray(pose["confidence"], dtype=float)
    if len(confidence) < 17:
        return False, "no full skeleton"
    head = float(np.max(confidence[0:3]))          # nose, both eyes
    hips = float(np.min(confidence[11:13]))        # left/right hip
    legs = int(np.sum(confidence[13:17] >= min_confidence))  # knees and ankles
    if head < min_confidence:
        return False, "head not visible"
    if hips < min_confidence:
        return False, "only the upper body"
    if legs < 3:
        return False, "legs not visible (sitting or cut off)"
    box = np.asarray(pose["box"], dtype=float)
    height, width = float(box[3] - box[1]), float(box[2] - box[0])
    if width <= 0 or height / width < 1.5:
        return False, "not standing (wide body)"
    return True, "ok"


def body_frame(pose: dict[str, Any], *, min_confidence: float = POSE_MIN_CONFIDENCE
               ) -> tuple[float, float, float] | None:
    """Where the body stands in the picture: ``(centre_x, feet_y, height_px)``.

    Measured from the skeleton, not from the crop: the hips give the centre, the
    ankles the floor, and the head-to-ankle span the height. Frames aligned this
    way land on each other even when the person moved between them, which is what
    stops a build from smearing a body across the room.
    """
    if not pose:
        return None
    points = np.asarray(pose["points"], dtype=float)
    confidence = np.asarray(pose["confidence"], dtype=float)
    if len(points) < 17 or len(confidence) < 17:
        return None
    seen = points[confidence >= min_confidence]
    if len(seen) < 4:
        return None
    hips = points[11:13][confidence[11:13] >= min_confidence]
    centre_x = float(hips[:, 0].mean()) if len(hips) else float(seen[:, 0].mean())
    feet_y = float(seen[:, 1].max())
    height_px = float(seen[:, 1].max() - seen[:, 1].min())
    if height_px < 16:
        return None
    return centre_x, feet_y, height_px


def head_above_shoulders(pose: dict[str, Any] | None,
                         min_above: float = HEAD_MIN_ABOVE,
                         max_above: float = HEAD_MAX_ABOVE) -> bool:
    """Is the head held up, above the shoulders, the way a person looks standing?

    Yaw-proof on purpose: only the vertical offset matters, so a person facing
    away still passes while a person with their head bent down over a phone does
    not.
    """
    if not pose:
        return False
    points = np.asarray(pose["points"], dtype=float)
    confidence = np.asarray(pose["confidence"], dtype=float)
    if len(points) < 13 or len(confidence) < 13:
        return False
    shoulders = points[5:7][confidence[5:7] >= POSE_MIN_CONFIDENCE]
    hips = points[11:13][confidence[11:13] >= POSE_MIN_CONFIDENCE]
    head = points[0:3][confidence[0:3] >= POSE_MIN_CONFIDENCE]
    if len(shoulders) == 0 or len(hips) == 0 or len(head) == 0:
        return False
    torso = abs(float(shoulders[:, 1].mean()) - float(hips[:, 1].mean()))
    if torso < 8:
        return False
    above = (float(shoulders[:, 1].mean()) - float(head[:, 1].mean())) / torso
    return min_above <= above <= max_above


def pose_shape(pose: dict[str, Any] | None, *, min_confidence: float = POSE_MIN_CONFIDENCE
               ) -> tuple[np.ndarray, np.ndarray] | None:
    """How the body is posed, with rotation and distance taken out.

    Every pair of visible joints — head, arms, legs, all seventeen — contributes
    its distance, divided by the torso length. Two frames of the same pose give
    nearly the same square, whether the person turned to the camera or away, and
    a hand on the hip, an arm raised or a crossed leg changes it a lot. That is
    the comparison the owner asked for: it is not only "sits or stands", it is
    where the head, the arms and the legs are.
    """
    if not pose:
        return None
    points = np.asarray(pose["points"], dtype=float)
    confidence = np.asarray(pose["confidence"], dtype=float)
    if len(points) < 17 or len(confidence) < 17:
        return None
    present = confidence >= min_confidence
    if int(present.sum()) < 8:
        return None
    shoulders = points[5:7][present[5:7]]
    hips = points[11:13][present[11:13]]
    if len(shoulders) and len(hips):
        torso = float(np.linalg.norm(shoulders.mean(axis=0) - hips.mean(axis=0)))
    else:
        torso = 0.0
    if torso < 8:  # no usable torso: fall back to the widest span
        seen = points[present]
        torso = float(np.linalg.norm(seen[:, None, :] - seen[None, :, :], axis=-1).max())
    if torso < 8:
        return None
    grid = np.zeros((17, 17), dtype=np.float32)
    for first in range(17):
        if not present[first]:
            continue
        for second in range(17):
            if present[second]:
                grid[first, second] = np.linalg.norm(points[first] - points[second]) / torso
    return grid, present


def pose_distance(first: tuple[np.ndarray, np.ndarray] | None,
                  second: tuple[np.ndarray, np.ndarray] | None) -> float:
    """Difference between two poses over the joints both of them saw."""
    if first is None or second is None:
        return 1.0
    left, left_seen = first
    right, right_seen = second
    shared = np.logical_and(left_seen[:, None] & left_seen[None, :],
                            right_seen[:, None] & right_seen[None, :])
    if int(shared.sum()) < 12:
        return 1.0
    difference = (left - right)[shared]
    return float(np.sqrt(np.mean(np.square(difference))))


def standing_quality(image: np.ndarray, mask: np.ndarray, *, face: Any = None,
                     min_height_share: float = MIN_HEIGHT_SHARE,
                     min_aspect: float = MIN_ASPECT,
                     min_sharpness: float = MIN_SHARPNESS) -> tuple[bool, str]:
    """Is this frame a whole standing person worth remembering?

    A build has to remember *how somebody looks*, not every pose they were ever
    in: a person sitting at a desk gives a wide blob, half a person at the edge
    of the frame gives half a body, and a blurred frame gives a smudge. Each of
    those is refused here with the reason, so the panel can show why a build used
    fewer frames than it had.
    """
    import cv2

    rows, columns = np.nonzero(mask)
    if len(rows) < 64:
        return False, "no person in the frame"
    height, width = image.shape[:2]
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(columns.min()), int(columns.max())
    body_height, body_width = bottom - top + 1, right - left + 1
    if top <= 2 or bottom >= height - 3 or left <= 2 or right >= width - 3:
        return False, "cut off by the frame"
    if body_height < min_height_share * height:
        return False, "too small in the frame"
    if body_width and body_height / float(body_width) < min_aspect:
        return False, "not standing (wide blob)"
    region = image[top:bottom + 1, left:right + 1]
    grey = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    if sharpness < min_sharpness:
        return False, "blurred"
    if face is not None:
        box = face.get("box") if isinstance(face, dict) else None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            head_y = float(box[1]) * height
            if head_y > top + 0.3 * body_height:
                return False, "no head in the crop"
    return True, "ok"


def clothing_signature(image: np.ndarray | None) -> tuple[float, ...]:
    """A Lab histogram of the torso band of one body crop.

    Colour, not shape: two photographs of one shirt under different light land
    close together, another shirt lands far away. The band avoids the head and
    the legs, which move.
    """
    if image is None or not getattr(image, "size", 0):
        return ()
    try:
        import cv2

        small = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
        torso = small[17:74, 9:55]  # rows ~13-58%, columns ~14-86%
        if not torso.size:
            return ()
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        histogram = cv2.calcHist([lab], [0, 1, 2], None, [6, 5, 5],
                                 [0, 256, 0, 256, 0, 256]).ravel()
        total = float(histogram.sum())
        if not total:
            return ()
        return tuple(float(value) / total for value in histogram)
    except Exception:  # a broken JPEG must not take a build down
        log.debug("Could not read a body crop for its clothes", exc_info=True)
        return ()


def clothing_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Histogram intersection distance: 0 is the same clothes, 1 is nothing alike."""
    if not left or not right or len(left) != len(right):
        return 1.0
    return 1.0 - sum(min(one, other) for one, other in zip(left, right))


def blend_signatures(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    """The running signature of one look: mean of what it has seen, renormalised."""
    merged = [0.5 * one + 0.5 * other for one, other in zip(left, right)]
    total = sum(merged)
    return tuple(value / total for value in merged) if total else left


def dominant_look(frames: Sequence[dict[str, Any]], signatures: Sequence[tuple[float, ...]],
                  *, pick: str = "newest", distance: float = LOOK_DISTANCE) -> list[int]:
    """Which frames wear the clothes the person is wearing *now*.

    ``frames`` are in the order they arrived (newest first). Groups are built
    greedily by clothes, and the answer is the newest group that is not a
    one-frame accident (``newest``), the biggest group (``biggest``), or simply
    everything (``all``).
    """
    if pick == "all" or not frames:
        return list(range(len(frames)))
    groups: list[dict[str, Any]] = []
    for index, signature in enumerate(signatures):
        if not signature:
            continue
        group = next((item for item in groups
                      if clothing_distance(signature, item["signature"]) <= distance), None)
        if group is None:
            groups.append({"signature": signature, "members": [index]})
        else:
            group["members"].append(index)
            group["signature"] = blend_signatures(group["signature"], signature)
    if not groups:
        return list(range(len(frames)))
    if pick == "biggest":
        chosen = max(groups, key=lambda item: len(item["members"]))
    else:
        solid = [item for item in groups if len(item["members"]) >= 2] or groups
        chosen = min(solid, key=lambda item: min(item["members"]))
    return sorted(chosen["members"])


_DEPTH: tuple[Any, Any] | None = None


def depth_model(directory: str | Path | None = None,
                device: str = "auto") -> tuple[Any, Any] | None:
    """The depth processor and model, loaded once, or ``None`` without weights.

    The hub's GPU belongs to the LLM and Whisper; this model is small enough to
    run on the CPU (measured: ~0.2 s per frame), which means a build can run
    while the rooms keep talking.
    """
    global _DEPTH, _DEPTH_DEVICE
    resolved = runtime_device(device)
    if _DEPTH is not None and _DEPTH_DEVICE == resolved:
        return _DEPTH
    folder = Path(directory or DEPTH_MODEL_DIR)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent.parent / folder
    if not folder.exists():
        log.warning("No depth model at %s: the photographic build is unavailable", folder)
        return None
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(str(folder))
        model = AutoModelForDepthEstimation.from_pretrained(str(folder))
        model.eval()
        if resolved == "cuda":
            model = model.to("cuda")
    except Exception:
        log.exception("The depth model at %s could not be loaded", folder)
        return None
    _DEPTH = (processor, model)
    _DEPTH_DEVICE = resolved
    return _DEPTH


def depth_relief(image: np.ndarray, mask: np.ndarray, processor: Any, model: Any,
                 ) -> np.ndarray | None:
    """How close each pixel is to the camera, 0..1, scaled inside the person.

    Monocular depth has no metres: only the order and the relative differences
    are meaningful. Normalising inside the mask (2nd..98th percentile) removes
    the unknown distance to the camera and leaves the person's own relief — a
    nose in front of the cheeks, a chest in front of the shoulders.
    """
    import cv2
    import torch

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")
        if _DEPTH_DEVICE == "cuda":
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
        with torch.no_grad():
            predicted = model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=image.shape[:2], mode="bicubic",
            align_corners=False).squeeze().cpu().numpy()
    except Exception:
        log.exception("Depth estimation failed on one frame")
        return None
    inside = depth[mask] if mask.any() else np.array([])
    if inside.size < 32:
        return None
    low, high = (float(value) for value in np.percentile(inside, (2.0, 98.0)))
    if not high > low:
        return None
    return np.clip((depth - low) / (high - low), 0.0, 1.0)


def frame_cloud(image: np.ndarray, mask: np.ndarray, yaw_deg: float,
                depth: np.ndarray | None, *, thickness: float = BODY_THICKNESS,
                step: int = 2, frame: tuple[float, float, float] | None = None
                ) -> np.ndarray | None:
    """One frame as points ``(x, y, z, r, g, b)`` in the person's own frame.

    ``y`` runs 0 at the feet to 1 at the top of the head, ``x`` across the body
    and ``z`` front-to-back, both in person-heights. The frame's camera direction
    is its yaw, so the points are rotated into one common frame and every frame
    of a turn lands on the same body. ``frame`` is the skeleton's own measure of
    the body — ``(centre_x, feet_y, height_px)`` from :func:`body_frame` — and is
    what keeps a person who moved between frames from smearing.
    """
    rows, columns = np.nonzero(mask)
    if len(rows) < 64:
        return None
    keep = np.zeros(len(rows), dtype=bool)
    keep[::max(1, int(step))] = True
    rows, columns = rows[keep], columns[keep]
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(columns.min()), int(columns.max())
    if frame is not None:
        centre, floor, height_px = frame
    else:
        centre, floor = (left + right) / 2.0, bottom
        height_px = max(8, bottom - top + 1)
    if (right - left + 1) > 1.4 * height_px:  # a wall-to-wall blob is not a person
        return None
    along = (columns - centre) / height_px
    up = (floor - rows) / height_px
    if depth is None:
        relief = np.zeros(len(rows), dtype=np.float32)
    else:
        relief = (depth[rows, columns].astype(np.float32) - 0.5) * thickness
    radians = math.radians(float(yaw_deg))
    x = along * math.cos(radians) - relief * math.sin(radians)
    z = along * math.sin(radians) + relief * math.cos(radians)
    colours = image[rows, columns][:, ::-1].astype(np.uint8)  # BGR -> RGB
    points = np.empty((len(rows), 6), dtype=np.float32)
    points[:, 0], points[:, 1], points[:, 2] = x, up, z
    points[:, 3:6] = colours
    return points


def merge_clouds(clouds: Iterable[np.ndarray], *, voxel: float = CLOUD_VOXEL,
                 limit: int = CLOUD_LIMIT) -> np.ndarray:
    """Average the frames into one cloud: 6 mm cells, colours averaged per cell.

    Two photographs of one shirt should not become two surfaces. Cells also cap
    the size of the file the panel has to draw.
    """
    parts = [cloud for cloud in clouds if cloud is not None and len(cloud)]
    if not parts:
        return np.zeros((0, 6), dtype=np.float32)
    merged = np.concatenate(parts, axis=0)
    cells = np.rint(merged[:, :3] / max(1e-6, voxel)).astype(np.int64)
    _unique, index, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 6), dtype=np.float64)
    np.add.at(sums, index, merged)
    averaged = sums / counts[:, None]
    if len(averaged) > limit:  # keep the densest cells if a build overshoots
        order = np.argsort(-counts)[:limit]
        averaged = averaged[order]
    averaged[:, 3:6] = np.clip(averaged[:, 3:6], 0, 255)
    return averaged.astype(np.float32)


def training_frames(root: str | Path, person_name: str,
                    kinds: Sequence[str] = ("appearance", "appearance_legacy",
                                            "camera_request"),
                    limit: int = 4000) -> list[tuple[str, float]]:
    """Every stored image of one person in the training archive (F-303).

    The archive keeps one folder per event with the files that event produced
    (``face.jpg`` for an appearance sighting, ``original.jpg`` for a camera
    request), and its index knows which person each event belongs to. This is
    where the thousands of frames live; the appearance gallery is the curated
    handful.
    """
    import sqlite3

    database = Path(root) / "index.sqlite3"
    if not database.exists():
        return []
    try:
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        log.warning("The training archive index at %s is unreadable", database)
        return []
    try:
        rows = connection.execute(
            "SELECT e.captured_at, e.record FROM events e JOIN identities i ON i.id = e.person_id"
            " WHERE lower(i.name) = lower(?) ORDER BY e.captured_at DESC LIMIT ?",
            (str(person_name), int(max(1, limit)))).fetchall()
    except sqlite3.Error:
        log.warning("The training archive has no events for %s", person_name)
        return []
    finally:
        connection.close()
    wanted = {str(kind) for kind in kinds}
    found: list[tuple[str, float]] = []
    for captured_at, record in rows:
        try:
            payload = json.loads(record)
        except (TypeError, ValueError):
            continue
        if str(payload.get("kind") or "") not in wanted:
            continue
        event_path = str(payload.get("event_path") or "")
        if not event_path:
            continue
        folder = (Path(root) / event_path).parent
        for name in payload.get("files") or []:
            candidate = folder / str(name)
            if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if candidate.is_file():
                found.append((str(candidate), float(captured_at)))
    found.sort(key=lambda item: item[1], reverse=True)
    return found[:limit]


def save_model(directory: str | Path, vertices: np.ndarray, faces: np.ndarray,
               colours: np.ndarray, *, meta: dict[str, Any]) -> dict[str, str]:
    """Write the mesh for other tools and a compact point file for the panel.

    ``model.ply`` opens in Blender or MeshLab; ``model.json`` plus ``model.bin``
    are what the owner's panel renders — six bytes per point, no CDN, no engine.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(vertices) == 0:
        raise ValueError("Nothing was carved: no usable view had a silhouette")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    span = np.where(high - low > 1e-6, high - low, 1.0)
    scaled = np.clip((vertices - low) / span, 0.0, 1.0)
    quantised = np.rint(scaled * 255.0).astype(np.uint8)
    payload = np.concatenate([quantised, np.asarray(colours, dtype=np.uint8)], axis=1)
    (directory / "model.bin").write_bytes(payload.tobytes())
    (directory / "model.json").write_text(json.dumps({
        "points": int(len(vertices)), "triangles": int(len(faces)),
        "format": "uint8 xyz + rgb, 6 bytes per point, in model.bin",
        "bounds": {"low": [float(v) for v in low], "high": [float(v) for v in high]},
        "meta": meta,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_ply(directory / "model.ply", vertices, faces, colours)
    return {"bin": str(directory / "model.bin"), "json": str(directory / "model.json"),
            "ply": str(directory / "model.ply")}


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray,
               colours: np.ndarray) -> None:
    """A binary little-endian PLY with vertex colours: Blender reads it as-is."""
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]) + "\n"
    vertex_rows = np.empty(len(vertices), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                 ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertex_rows["x"], vertex_rows["y"], vertex_rows["z"] = (vertices[:, 0], vertices[:, 1],
                                                            vertices[:, 2])
    vertex_rows["r"], vertex_rows["g"], vertex_rows["b"] = (colours[:, 0], colours[:, 1],
                                                            colours[:, 2])
    face_rows = np.empty(len(faces), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"),
                                            ("c", "<i4")])
    face_rows["n"] = 3
    face_rows["a"], face_rows["b"], face_rows["c"] = faces[:, 0], faces[:, 1], faces[:, 2]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex_rows.tobytes())
        handle.write(face_rows.tobytes())


__all__ = ["BODY_THICKNESS", "CLOUD_VOXEL", "DEPTH_MODEL_DIR", "HALF_EXTENT", "POSE_DISTANCE",
           "RESOLUTION", "View", "body_frame", "carve", "clothing_distance",
           "clothing_signature", "colour_of", "decode", "depth_model", "depth_relief",
           "dominant_look", "face_yaw", "frame_cloud", "head_above_shoulders", "merge_clouds",
           "mesh_of", "pose_distance", "pose_model", "pose_shape", "save_model", "select_views",
           "silhouette", "skeleton", "standing_from_skeleton", "standing_quality",
           "training_frames", "yaw_from_keypoints"]
