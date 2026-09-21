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
