"""Build the 3D model of one person out of the photographs the hub stores.

    python scripts/build-3d-profile.py --person Anton                     # everything
    python scripts/build-3d-profile.py --person Anton --source archive --limit 2000
    python scripts/build-3d-profile.py --person Anton --method hull

Two methods, two sources and a frame budget, because the owner asked for the
menu in the panel to choose exactly this:

* ``--method photos`` (default) — a depth model turns every frame into relief and
  every pixel of the person becomes a coloured 3D point, so the model is made of
  the real photographs, thousands of them if you ask for them;
* ``--method hull`` — silhouettes only, carved into a body (fast, blocky).

* ``--source archive`` — the training archive (F-303): every stored crop of the
  person, thousands of frames;
* ``--source gallery`` — the curated appearance archive (F-209), tens of frames;
* ``--source bodies`` — the per-track body crops in ``data/homes``;
* ``--source all`` (default) — all three, newest first.

The build is a command, never a service: it writes progress into
``status.json`` beside the model and stops when it is done, so the panel can
watch it without the hub's loop doing any of the work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from hub import three_d  # noqa: E402 - the path above is what makes ``hub`` importable
from hub.web_admin import WebAdminData  # noqa: E402 - the panel's own reader

SOURCES = ("all", "archive", "gallery", "bodies")
METHODS = ("photos", "hull")
#: How many prepared frames (silhouette + angle + clothes) one build keeps in
#: memory. A mask is a fraction of a megabyte, but twenty thousand of them are
#: gigabytes, so a longer build is thinned evenly instead of swapping.
PREPARED_LIMIT = 3000
#: A whole 1080p frame costs 0.57 s in insightface, an hour over a build, and the
#: yaw of a head is the same angle at 800 px: the frame is resized before the
#: face pass. Below that nothing changes.
FACE_MAX_SIDE = 800


def face_frame(image: np.ndarray) -> np.ndarray:
    """The frame the face pass reads: the same picture, at most ``FACE_MAX_SIDE``."""
    longest = int(max(image.shape[:2]))
    if longest <= FACE_MAX_SIDE:
        return image
    import cv2

    ratio = FACE_MAX_SIDE / float(longest)
    return cv2.resize(image, (max(1, int(image.shape[1] * ratio)),
                              max(1, int(image.shape[0] * ratio))),
                      interpolation=cv2.INTER_AREA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3D model of one archived person")
    parser.add_argument("--person", required=True, help="id, archive id, or name")
    parser.add_argument("--root", default=str(REPO / "data" / "hub.db"), help="path to hub.db")
    parser.add_argument("--archive", default=str(REPO / "data" / "training_archive"),
                        help="path to the training archive")
    parser.add_argument("--source", choices=SOURCES, default="all")
    parser.add_argument("--method", choices=METHODS, default="photos")
    parser.add_argument("--limit", type=int, default=600, help="how many frames to use")
    parser.add_argument("--views", type=int, default=16,
                        help="how many angles the silhouette method carves from")
    parser.add_argument("--resolution", type=int, default=three_d.RESOLUTION)
    parser.add_argument("--step", type=int, default=3,
                        help="keep every Nth pixel of a frame (photos method)")
    parser.add_argument("--standing", default="on", choices=("on", "off"),
                        help="keep only whole, standing frames (default on)")
    parser.add_argument("--look", default="newest", choices=("newest", "biggest", "all"),
                        help="which clothes to build from (default: the newest look)")
    parser.add_argument("--pose", default="same", choices=("same", "any"),
                        help="one pose only (default) or every pose mixed")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="run the models on the GPU when it is free (default auto)")
    parser.add_argument("--status", default="", help="where to write progress (JSON)")
    return parser.parse_args()


class Progress:
    """The file the panel polls: what is running, how far it got, how it ended."""

    def __init__(self, path: Path, total: int, options: dict) -> None:
        self.path = path
        self.total = total
        self.started = time.time()
        self.options = options

    def write(self, state: str, done: int = 0, **extra) -> None:
        payload = {"state": state, "done": int(done), "total": int(self.total),
                   "seconds": round(time.time() - self.started, 1),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(self.started)),
                   "options": self.options}
        payload.update(extra)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except OSError:
            pass


def find_person(data: WebAdminData, wanted: str) -> dict:
    people = data.profiles()
    key = wanted.strip().casefold()
    for person in people:
        if key in {str(person["profile_id"]).casefold(), str(person["gallery_id"]).casefold()}:
            return person
    for person in people:
        if str(person["name"]).strip().casefold() == key:
            return person
    names = ", ".join(sorted(person["name"] for person in people)) or "nobody"
    raise SystemExit(f"No profile matches {wanted!r}. The hub knows: {names}")


def archive_frames(archive: Path, name: str, limit: int) -> list[dict]:
    """Body crops of the training archive, each with the full frame for the pose.

    One event folder holds ``body.png``, ``face.png`` and ``original.jpg``; the
    body crop is what the model is built of, the original frame the head angle
    is read from (a face is easier to find in the whole scene than in a crop).
    """
    images = three_d.training_frames(archive, name, limit=max(1, limit) * 3)
    folders: dict[str, dict[str, str]] = {}
    for path, stamp in images:
        item = folders.setdefault(str(Path(path).parent), {})
        item[Path(path).name.lower()] = path
        item.setdefault("stamp", str(stamp))
    frames = []
    for folder in sorted(folders, key=lambda key: float(folders[key]["stamp"]), reverse=True):
        item = folders[folder]
        crop = item.get("body.png") or item.get("body.jpg") or item.get("original.jpg")
        if not crop:
            continue
        frames.append({"path": crop, "pose": item.get("original.jpg", crop),
                       "stamp": float(item["stamp"]), "source": "archive"})
        if len(frames) >= limit:
            break
    return frames


def gallery_frames(data: WebAdminData, person: dict, limit: int) -> list[dict]:
    frames = []
    for sample in data.appearance_samples(person["gallery_id"])[:limit]:
        path = data.gallery_root / sample["body_file"] if sample["body_file"] else None
        if path is None or not path.exists():
            continue
        frames.append({"path": str(path), "pose": str(path), "stamp": 0.0,
                       "source": "gallery"})
    return frames


def body_frames(data: WebAdminData, person: dict, limit: int) -> list[dict]:
    frames = []
    for path, stamp in data.body_crops(person["person_id"], limit=limit):
        frames.append({"path": path, "pose": path, "stamp": float(stamp),
                       "source": "bodies"})
    return frames


def gather(data: WebAdminData, person: dict, args: argparse.Namespace) -> list[dict]:
    # Three times as many frames as the model needs: the filters throw away
    # sitting and half-visible frames, and this hub's newest frames are mostly
    # somebody at a desk. Every source is asked for the wider pool, not only the
    # final slice, or "limit 600" would mean "the 600 newest frames, of which
    # eight are a standing person".
    wanted = max(args.limit, args.limit * 3)
    frames: list[dict] = []
    if args.source in {"all", "archive"}:
        frames += archive_frames(Path(args.archive), person["name"], wanted)
    if args.source in {"all", "gallery"} and person["gallery_id"]:
        frames += gallery_frames(data, person, wanted)
    if args.source in {"all", "bodies"} and person["person_id"]:
        frames += body_frames(data, person, wanted)
    seen: set[str] = set()
    unique = []
    for frame in frames:
        if frame["path"] in seen:
            continue
        seen.add(frame["path"])
        unique.append(frame)
    unique.sort(key=lambda frame: frame["stamp"], reverse=True)
    return unique[:wanted]


def prepare(frames: list[dict], engine: Any, progress: Progress, args) -> tuple[list, list, dict]:
    """Read every frame once: silhouette, head angle, and whether it is usable.

    The owner's complaint ("оно все смешивает когда я сижу") is answered here:
    frames that are not a whole standing person are dropped with a reason, and
    the frames are grouped by clothes so a build can take one look instead of
    every look mixed together.

    The angle of a frame comes from the face when there is one and from the
    skeleton when there is not (:func:`hub.three_d.yaw_from_body`), so somebody
    filmed from the side or from behind is no longer thrown away for having no
    eyes to measure.
    """
    usable, skipped, reasons = [], 0, {}
    angles_from: dict[str, int] = {}
    pose_engine = three_d.pose_model(device=args.device)
    progress.write("running", 0, note="reading frames")
    for index, frame in enumerate(frames, start=1):
        image = three_d.decode(frame["path"])
        mask = three_d.silhouette(image) if image is not None else None
        pose = three_d.skeleton(image, pose_engine) if image is not None else None
        body = three_d.body_frame(pose) if pose else None
        pose_image = three_d.decode(frame["pose"]) if frame["pose"] else None
        faces = []
        if pose_image is not None and engine is not None:
            try:
                import cv2

                encoded = cv2.imencode(".jpg", face_frame(pose_image))[1].tobytes()
                faces = engine.located_faces(encoded)
            except Exception:
                faces = []
        yaw = None
        if faces:
            face_pose = faces[0].get("pose")
            if isinstance(face_pose, (list, tuple)) and len(face_pose) == 3:
                yaw = float(face_pose[1])
            if yaw is None:
                yaw = three_d.yaw_from_keypoints(faces[0].get("landmarks"))
        angle_from = "face" if yaw is not None else ""
        if yaw is None:
            body_yaw, where = three_d.yaw_from_body(pose)
            if body_yaw is not None:
                yaw, angle_from = body_yaw, f"body ({where})"
        if image is None or mask is None:
            reason = "no silhouette"
        elif yaw is None:
            reason = "no face and no body to read the angle from"
        elif args.standing == "on" and pose_engine:
            ok, note = three_d.standing_from_skeleton(pose)
            if ok and not three_d.head_above_shoulders(pose):
                ok, note = False, "head not held up"
            reason = "ok" if ok else note
        elif args.standing == "on":
            ok, note = three_d.standing_quality(image, mask, face=faces[0] if faces else None)
            reason = "ok" if ok else note
        else:
            reason = "ok"
        if reason == "ok":
            usable.append({"path": frame["path"], "pose": frame["pose"],
                           "stamp": frame["stamp"], "source": frame["source"],
                           "mask": mask, "yaw": float(yaw),
                           "angle_from": angle_from,
                           "frame": body,
                           "shape": three_d.pose_shape(pose),
                           "clothes": three_d.clothing_signature(image)})
            angles_from[angle_from or "unknown"] = angles_from.get(angle_from or "unknown", 0) + 1
        else:
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
        if index % 10 == 0 or index == len(frames):
            progress.write("running", index, used=len(usable), skipped=skipped,
                           reasons=reasons, angles_from=angles_from)
    if args.look != "all" and len(usable) > 3:
        keep = three_d.dominant_look(usable, [frame["clothes"] for frame in usable],
                                     pick=args.look)
        dropped = len(usable) - len(keep)
        if len(keep) >= 3:
            usable = [usable[index] for index in keep]
            reasons[f"another look ({args.look})"] = dropped
    if args.pose == "same" and len(usable) > 6:
        # One pose only: the head, the arms and the legs in the same place in
        # every frame that builds the model, so the body cannot be smeared by
        # somebody who put a hand on the desk between frames.
        groups: list[dict] = []
        for index, frame in enumerate(usable):
            group = next((item for item in groups if three_d.pose_distance(
                frame["shape"], item["reference"]) <= three_d.POSE_DISTANCE), None)
            if group is None:
                groups.append({"reference": frame["shape"], "members": [index]})
            else:
                group["members"].append(index)
        solid = [item for item in groups if len(item["members"]) >= 2] or groups
        chosen = (max(solid, key=lambda item: len(item["members"])) if args.look == "biggest"
                  else min(solid, key=lambda item: min(item["members"])))
        keep = set(chosen["members"])
        if len(keep) >= 3:
            reasons["another pose (head, arms or legs)"] = len(usable) - len(keep)
            usable = [frame for index, frame in enumerate(usable) if index in keep]
    if len(usable) > PREPARED_LIMIT:
        step = len(usable) / float(PREPARED_LIMIT)
        thinned = [usable[int(index * step)] for index in range(PREPARED_LIMIT)]
        reasons["thinned to keep the build small"] = len(usable) - len(thinned)
        usable = thinned
    if len(usable) > args.limit:
        reasons["more than the ask (newest kept)"] = len(usable) - args.limit
        usable = usable[:args.limit]
    return usable, sorted(angles_from.items()), reasons


def build_photos(usable: list[dict], progress: Progress, args) -> tuple:
    """Every frame becomes real pixels in space; the frames merge into one body."""
    loaded = three_d.depth_model(device=args.device)
    if loaded is None:
        raise SystemExit("The depth model is missing: the photographic build needs "
                         f"{three_d.DEPTH_MODEL_DIR}")
    processor, model = loaded
    clouds, used, skipped, angles = [], 0, 0, []
    for index, frame in enumerate(usable, start=1):
        image, mask = three_d.decode(frame["path"]), frame["mask"]
        if image is None:
            skipped += 1
            continue
        relief = three_d.depth_relief(image, mask, processor, model)
        cloud = three_d.frame_cloud(image, mask, frame["yaw"], relief, step=args.step,
                                    frame=frame.get("frame"))
        if cloud is not None:
            clouds.append(cloud)
            used += 1
            angles.append(round(float(frame["yaw"]), 1))
        else:
            skipped += 1
        if index % 10 == 0 or index == len(usable):
            progress.write("running", index, used=used, skipped=skipped)
    merged = three_d.merge_clouds(clouds)
    if not len(merged):
        raise SystemExit("No frame gave a usable cloud: nothing was written.")
    vertices = merged[:, :3]
    colours = np.clip(merged[:, 3:6], 0, 255).astype(np.uint8)
    angles.sort()
    return (vertices, np.zeros((0, 3), dtype=np.int64), colours, used, skipped,
            [angles[0], angles[-1]] if angles else [])


def build_hull(usable: list[dict], progress: Progress, args) -> tuple:
    """Silhouettes only: fast, blocky, and it needs very few angles."""
    prepared = {}
    for frame in usable:
        image = three_d.decode(frame["path"])
        if image is not None:
            prepared[frame["path"]] = (image, frame["mask"], frame["yaw"])
    views = [three_d.View(sample_id=frame["path"], yaw_deg=float(frame["yaw"]),
                          path=frame["path"]) for frame in usable
             if frame["path"] in prepared]
    if len(views) < 3:
        raise SystemExit("Only %d usable view(s): a carve needs the person at several "
                         "angles, read from a face or a body. Nothing was written."
                         % len(views))
    chosen = three_d.select_views(views, args.views)
    progress.write("running", len(usable), used=len(chosen))
    grid = three_d.carve([(prepared[view.sample_id][1], view.yaw_deg) for view in chosen],
                         resolution=args.resolution)
    if not grid.any():
        raise SystemExit("Every view disagreed: the carved hull is empty. Nothing was written.")
    vertices, faces = three_d.mesh_of(grid, resolution=args.resolution)
    colours = three_d.colour_of(vertices, [prepared[view.sample_id] for view in chosen])
    return vertices, faces, colours, len(chosen), len(usable) - len(views), \
        sorted(round(view.yaw_deg, 1) for view in chosen)


def main() -> int:
    args = parse_args()
    data = WebAdminData(args.root)
    person = find_person(data, args.person)
    if not person["gallery_id"] and not person["person_id"]:
        raise SystemExit(f"{person['name']} has no archived frame to build from")
    directory = data.gallery_root / (person["gallery_id"] or person["person_id"]) / "model"
    frames = gather(data, person, args)
    if len(frames) < 3:
        raise SystemExit("Only %d frame(s) are available for %s with the source %r."
                         % (len(frames), person["name"], args.source))
    options = {"person": person["name"], "source": args.source, "method": args.method,
               "limit": args.limit, "step": args.step, "device": args.device}
    progress = Progress(Path(args.status) if args.status else directory / "status.json",
                        len(frames), options)
    progress.write("running", 0, device=three_d.runtime_device(args.device))
    try:
        from hub.face import FaceEngine
    except Exception as exc:  # pragma: no cover - the hub always ships it
        raise SystemExit(f"insightface is not importable: {exc}") from exc
    engine = FaceEngine()
    started = time.time()
    usable, angles_from, reasons = prepare(frames, engine, progress, args)
    if len(usable) < 3:
        raise SystemExit("Only %d frame(s) of %d are a whole standing person: %s"
                         % (len(usable), len(frames),
                            ", ".join(f"{count}× {reason}" for reason, count in reasons.items())
                            or "nothing usable"))
    if args.method == "hull":
        vertices, faces, colours, used, skipped, angles = build_hull(usable, progress, args)
    else:
        vertices, faces, colours, used, skipped, angles = build_photos(usable, progress, args)
    skipped = len(frames) - used
    meta = {"person": person["name"], "profile": person["gallery_id"] or person["person_id"],
            "source": args.source, "method": args.method,
            "device": three_d.runtime_device(args.device),
            "views": used, "of_frames": len(frames), "silhouettes": used,
            "yaw_range": angles, "resolution": args.resolution,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seconds": round(time.time() - started, 1),
            "skipped": skipped,
            "standing": args.standing == "on", "look": args.look,
            "pose": args.pose,
            "angles_from": dict(angles_from),
            "skip_reasons": reasons,
            "method_note": ("depth relief + real pixels" if args.method == "photos"
                            else "visual hull from silhouettes, coloured from the frames")}
    written = three_d.save_model(directory, vertices, faces, colours, meta=meta)
    progress.write("ok", len(frames), points=int(len(vertices)), triangles=int(len(faces)),
                   used=used, skipped=skipped, seconds=meta["seconds"],
                   device=meta["device"], angles_from=meta["angles_from"],
                   files={kind: str(path) for kind, path in written.items()})
    print(f"{person['name']}: {args.method} from {args.source}, {used} frame(s) of "
          f"{len(frames)}, {len(vertices)} points, {len(faces)} triangles, "
          f"{meta['seconds']} s on {meta['device']}")
    for kind, path in written.items():
        print(f"   {kind}: {path}")
    return 0


if __name__ == "__main__":
    def _report_failure(message: str) -> None:
        where = parse_args().status
        if not where:
            return
        try:
            Path(where).write_text(json.dumps(
                {"state": "error", "error": message, "seconds": 0.0},
                ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    try:
        code = main()
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else 1
        if code:
            _report_failure(str(stop.code))
    except Exception as error:  # keep the status file the panel polls honest
        _report_failure(f"{type(error).__name__}: {error}")
        print(f"BUILD FAILED: {error}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
