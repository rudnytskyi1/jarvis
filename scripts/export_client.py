"""Export the public client from an explicit allowlist, never from Git history."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "client/__init__.py", "client/main.py", "client/setup.py", "client/ws_client.py",
    "client/audio.py", "client/audio_processing.py", "client/attention.py",
    "client/barge_in.py",
    "client/vad.py", "client/wakeword.py", "client/voice_controls.py",
    "client/local_commands.py",
    "client/local_stt.py", "client/offline.py", "client/presence_buffer.py",
    "client/tts_cache.py",
    "client/tracking.py",
    "client/body_crops.py",
    "client/privacy.py",
    "client/ota.py",
    "client/vision_profile.py",
    "client/camera.py", "client/camera_clips.py", "client/frame_recording.py", "client/room-tracker.yaml",
    "client/screen.py", "client/viewer.py", "client/overlay.py", "client/overlay_web/chat.html",
    "client/actions/__init__.py", "client/actions/app_control.py", "client/actions/apps.py",
    "client/actions/browser.py", "client/actions/browser_desktop.py", "client/actions/dispatcher.py", "client/actions/photos.py", "client/actions/wallpaper.py",
    "client/actions/pc.py", "client/devices/__init__.py", "client/devices/base.py",
    "client/devices/registry.py", "client/devices/magichome.py", "client/devices/tuya.py",
    "client/devices/switchbot.py", "client/requirements.txt", "client/requirements-audio.txt",
    "client/requirements-browser.txt", "client/requirements-camera.txt", "client/requirements-overlay.txt",
    "common/__init__.py", "common/client_config.py", "common/ids.py", "common/protocol.py",
    "common/voice_commands.py", "common/recording.py",
    "common/body_crops.py",
)
TEMPLATE_FILES = (
    "README.md", ".gitignore", "config.example.yaml", "setup-client.bat", "start-client.bat",
    "scripts/install-client.ps1", "scripts/run-client.ps1", "scripts/update-client.ps1", "update-client.bat",
    "setup-client-conda.bat", "start-client-conda.bat", "scripts/client-python.ps1",
)
SECRET_PATTERNS = {
    "OpenAI-style credential": r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}",
    "Google API credential": r"\bAIza[0-9A-Za-z_-]{30,}",
    "Telegram bot credential": r"\b[0-9]{7,12}:[A-Za-z0-9_-]{30,}\b",
    "GitHub credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})",
    "AWS credential": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "private key": r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----",
    "embedded credential": r"(?i)(?:api[_-]?key|local_key|access_token|password|secret)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_./+=-]{16,}[\"']",
    "personal Windows path": r"[A-Za-z]:\\+Users\\+(?!Public\b|Default\b)[^\\\s\"']+",
    "private tunnel address": r"[A-Za-z0-9-]+\.ngrok(?:-free)?\.(?:app|dev|io)",
}


def _read_regular(root: Path, name: str) -> bytes:
    candidate = root / name
    # Refuse symlinks and Windows junctions anywhere inside the source path.
    for component in (candidate, *candidate.parents):
        if component == root:
            break
        info = component.lstat()
        if component.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"Refusing linked source path: {name}")
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"Not a regular source file: {name}")
    return candidate.read_bytes()


def audit_payloads(payloads: dict[str, bytes]) -> None:
    for name, payload in payloads.items():
        content = payload.decode("utf-8-sig")
        for label, pattern in SECRET_PATTERNS.items():
            if re.search(pattern, content):
                # Never display the matching value, even in a failed build log.
                raise ValueError(f"Export blocked: {label} in {name}")
        if not name.endswith(".py"):
            continue
        tree = ast.parse(content, filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
                if node.module in {"client", "common"}:
                    modules += [node.module + "." + alias.name for alias in node.names]
            for module in modules:
                top = module.split(".")[0]
                if top in {"server", "hub"}:
                    raise ValueError(f"Export blocked: brain dependency in {name}")
                if top not in {"client", "common"}:
                    continue
                target = module.replace(".", "/")
                if target + ".py" not in payloads and target + "/__init__.py" not in payloads:
                    raise ValueError(f"Missing client dependency {module} required by {name}")


def export_client(root: Path, destination: Path) -> tuple[Path, Path, int]:
    root = root.resolve()
    destination = destination.absolute()
    archive_path = destination.with_suffix(".zip")
    if destination.exists() or archive_path.exists():
        raise ValueError("Output already exists. Use a new --output directory; existing files are never merged or removed.")
    payloads = {name: _read_regular(root, name) for name in RUNTIME_FILES}
    payloads.update({name: _read_regular(root, "distribution/client/" + name) for name in TEMPLATE_FILES})
    audit_payloads(payloads)
    payloads["release-manifest.json"] = (json.dumps({
        "format": 1,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payloads.items())},
    }, indent=2) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="client-export-", dir=destination.parent) as temp:
        stage = Path(temp) / "client"
        stage.mkdir()
        for name, payload in payloads.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        staged_zip = Path(temp) / "client.zip"
        with zipfile.ZipFile(staged_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(payloads.items()):
                # Fixed metadata: a rebuild of the same sources is reproducible.
                info = zipfile.ZipInfo(destination.name + "/" + name, (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, payload)
        # Verify the actual ZIP bytes rather than just the input list.
        with zipfile.ZipFile(staged_zip) as archive:
            zipped = {name.split("/", 1)[1]: archive.read(name) for name in archive.namelist()}
            if zipped != payloads:
                raise ValueError("ZIP verification failed.")
            audit_payloads({name: data for name, data in zipped.items() if name != "release-manifest.json"})
        stage.rename(destination)
        staged_zip.rename(archive_path)
    return destination, archive_path, len(payloads)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/rowan-client")
    args = parser.parse_args()
    try:
        folder, archive, count = export_client(ROOT, args.output)
    except (ValueError, OSError, SyntaxError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(f"Client export verified: {count} files\nFolder: {folder}\nZIP: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
