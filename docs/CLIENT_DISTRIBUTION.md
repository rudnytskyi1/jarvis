# Standalone public client

One implementation is maintained in this checkout's `client/` and shared
`common/` files. GitHub is an automatically generated release of those exact
bytes, not a second edited source tree. Packaging templates contain only setup,
documentation and safe example settings. The room PC receives the same runtime
files and keeps its local config. Release updates do not carry API keys.

Publish a reviewed release to the owner-approved repository with:

```powershell
python scripts/publish_client.py --push
```

Without `--push`, the script prepares and verifies a local Git checkout only.
It exports through the fixed allowlist, checks hashes, clones the public repo
without private history, refuses unexpected tracked files and commits only the
generated release. Push is a normal fast-forward; no force-push is used. The
repository defaults to `https://github.com/rudnytskyi1/rowanai.git`.

On a second PC: clone the repository, run `setup-client.bat`, give the server
address and a unique workplace name, then `start-client.bat`. Later stop the
client and run `update-client.bat`; local settings/data remain in place.

For an existing Anaconda Python 3.11/3.12 environment, activate it in Anaconda
Prompt and run `setup-client-conda.bat`, then `start-client-conda.bat`. This uses
the selected environment directly without creating `.venv`. The interpreter
path is stored only in ignored `.rowan-python`; normal start/update scripts
honor that selection. An explicit `-Python C:\Path\To\env\python.exe` also works
with the Conda launchers. Setup preserves a backup when changing the selection.

Build from the full private checkout with:

```powershell
python scripts/export_client.py
```

The default outputs are `dist/rowan-client/` and `dist/rowan-client.zip`.
For another build use a new directory, for example
`python scripts/export_client.py --output dist/rowan-client-v2`.
The exporter refuses to overwrite/merge existing exports, so local setup files
or recordings from a previously tested copy cannot get swept into a release.

Publish **only that generated directory**, as a new repository with fresh Git
history. Do not push the full private checkout or copy its `.git` directory.
The exporter uses a fixed source allowlist and separate templates under
`distribution/client/`; it never reads the active configs, SSH keys, profiles,
recordings or Git history. `release-manifest.json` records SHA-256 hashes of
every included file. Both folder sources and the actual ZIP contents are
checked for recognized credential patterns and Python dependency closure.

`common/client_config.py` owns the client models and shared recording settings.
`common/config.py` reexports those models for compatibility with the brain.
The client loader accepts existing combined YAML but discards the server
section; the public package does not ship `common/config.py` or provider-model
configuration. Private config files are never rewritten by the export.

Standalone setup creates a virtual environment, asks for the endpoint and
microphone, optionally enables a camera, and downloads only the small Vosk
wake-word model. A fresh installation receives a unique client ID. Advanced
settings survive reconfiguration. The public default has local camera
recording disabled; this does not modify the owner's recording settings.

Server API keys remain server-side. Publishing client code is separate from
opening access to the brain: the existing WebSocket protocol does not provide
invitation tokens or per-installation authentication. Use a protected endpoint
or private network with trusted users; an invitation mechanism would need a
separate coordinated server/client change. The setup and README explain the
server's ability to request data and execute desktop actions.

Validation: `tests/test_client_distribution.py` exercises isolated imports with
brain modules blocked, client/legacy config compatibility, secret filtering,
exclusion of injected private files, URL validation, reconfiguration, unique
IDs and safe wake-model extraction. Installing dependencies from scratch and
testing physical devices on another PC are separate hardware checks.
