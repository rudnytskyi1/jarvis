# Vendored sqlite-vec extension

`vec0.dll` — loadable SQLite extension from the `sqlite-vec` 0.1.9 wheel
(`sqlite_vec-0.1.9-py3-none-win_amd64.whl`, upstream: <https://github.com/asg017/sqlite-vec>,
MIT / Apache-2.0 dual licensed).

The hub loads it from here when the `sqlite_vec` Python package is not
installed and `ROWAN_SQLITE_VEC_PATH` is not set (see `hub/vectors.py`).
Checked in because this store runs on Windows, where building the extension
from source is not an option and the local conda environment cannot install
into `site-packages` from the sandbox.
