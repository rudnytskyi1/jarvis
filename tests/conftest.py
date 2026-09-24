"""Shared test setup: make the repository root importable.

The sandbox denies access to directories created with POSIX mode ``0o700`` (see
AGENTS.md). On Windows the mode is otherwise ignored, so every ``os.mkdir`` /
``os.makedirs`` / ``Path.mkdir`` call is coerced to ``0o777`` before pytest or a
test creates a temporary directory. Temporary files are also pinned to a
directory inside the workspace, because ``%TEMP%`` may live outside the writable
root. This keeps the whole suite runnable in the sandbox without escalation.
"""
import os
import sys
from pathlib import Path

# ultralytics monkey-patches ``PIL.Image.open`` and, on any failure, asks pip to
# install ``pi-heif`` - inside a sandbox without a network that call blocks the
# whole run for tens of minutes (found on 2026-09-24: `pytest tests` hung at 55 %
# in tests/test_image_difficulty.py, py-spy showed `subprocess.check_output` ->
# `pip install pi-heif`). Rowan never installs packages at runtime, so the
# auto-install stays off for the suite; the package itself is installed.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    _sandbox_tmp = REPO_ROOT / "data" / ".pytest-tmp"
    _sandbox_tmp.mkdir(parents=True, exist_ok=True)
    for _name in ("TMPDIR", "TMP", "TEMP"):
        os.environ.setdefault(_name, str(_sandbox_tmp))
    # ``PYTEST_DEBUG_TEMPROOT`` is read lazily by pytest when the first tmp_path
    # fixture is requested, so setting it here is early enough.
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_sandbox_tmp))
    # ``hub.app`` opens ``data/hub.db`` for the audit, decision and turn traces.
    # Without this, every test that touches the hub gateway would write into the
    # owner's live database (it did: the trace of a turn lands there).
    os.environ.setdefault("ROWAN_HUB_DB", str(_sandbox_tmp / "hub.db"))

_original_mkdir = os.mkdir


def _mkdir_coerce(path, mode=0o777, *, dir_fd=None):
    return _original_mkdir(path, 0o777, dir_fd=dir_fd)


_original_makedirs = os.makedirs


def _makedirs_coerce(name, mode=0o777, exist_ok=False):
    return _original_makedirs(name, 0o777, exist_ok=exist_ok)


os.mkdir = _mkdir_coerce  # type: ignore[assignment]
os.makedirs = _makedirs_coerce  # type: ignore[assignment]

# ``pathlib.Path.mkdir`` calls the patched ``os`` functions, but pin it too so
# a direct reference cannot bypass the coercion above.
import pathlib  # noqa: E402

_original_path_mkdir = pathlib.Path.mkdir


def _path_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    return _original_path_mkdir(self, mode=0o777, parents=parents, exist_ok=exist_ok)


pathlib.Path.mkdir = _path_mkdir  # type: ignore[method-assign]
