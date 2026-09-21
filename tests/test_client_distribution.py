"""Public-client boundary: no private inputs, server imports or unsafe setup writes."""
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from client.setup import MODEL_NAME, _unpack_model, save_settings, validate_server_url
from common.client_config import load_client_config
from common.config import ClientConfig, load_config
from scripts.export_client import RUNTIME_FILES, TEMPLATE_FILES, audit_payloads, export_client

ROOT = Path(__file__).resolve().parents[1]


def test_client_loader_matches_legacy_settings_without_server(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('server:\n  llm:\n    api_key: must-not-retain\nclient:\n  server_url: ws://localhost:8765/ws\n', encoding="utf-8")
    cfg = load_client_config(path)
    assert isinstance(cfg.client, ClientConfig)
    assert not hasattr(cfg, "server")
    assert "must-not-retain" not in cfg.model_dump_json()


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml", "config.openai.yaml"])
def test_existing_config_compatibility(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip("Private config is not present")
    assert load_client_config(path).client.model_dump() == load_config(path).client.model_dump()


def test_client_config_errors_do_not_echo_secrets(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text('client:\n  server_url: ws://localhost/ws\n  audio:\n    sample_rate: secret-value-not-a-number\n', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_client_config(path)
    assert "client.audio.sample_rate" in str(error.value)
    assert "secret-value" not in str(error.value)
    path.write_text('client: [\npassword: secret-value\n', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_client_config(path)
    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize("url", ["https://example.com/ws", "ws://u:secret@example.com/ws", "wss://example.com/ws?key=secret", "ws://localhost:bad/ws", "ws://", "ws://bad host/ws"])
def test_setup_rejects_credentials_and_invalid_urls(url):
    with pytest.raises(ValueError):
        validate_server_url(url)


def test_setup_preserves_existing_settings_and_identity(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"server": {"private": "keep-local"}, "client": {
        "server_url": "ws://old/ws", "client_id": "existing-room", "audio": {"output_device": 7},
        "camera": {"fps": 0, "frame_recording": {"enabled": True}}}}), encoding="utf-8")
    save_settings(path, server_url="wss://example.com/ws", input_device=4, camera_enabled=True, camera_index=1)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["server"] == {"private": "keep-local"}
    client = load_client_config(path).client
    assert (client.client_id, client.audio.output_device, client.audio.input_device) == ("existing-room", 7, 4)
    assert client.camera.fps == 0 and client.camera.frame_recording.enabled


def test_setup_unique_installations_and_no_server_keys(tmp_path):
    configs = []
    for index in range(2):
        path = tmp_path / f"config-{index}.yaml"
        save_settings(path, server_url="wss://example.com/ws", input_device=None, camera_enabled=False, camera_index=0)
        cfg = load_client_config(path)
        configs.append(cfg)
        assert set(yaml.safe_load(path.read_text())) == {"client"}
        assert not cfg.client.camera.frame_recording.enabled
    assert configs[0].client.client_id != configs[1].client.client_id


@pytest.mark.parametrize("entry", ["../outside.txt", f"{MODEL_NAME}/../escape.txt", "C:/escape.txt", f"{MODEL_NAME}\\..\\escape.txt"])
def test_model_download_cannot_extract_outside_model(tmp_path, entry):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, b"malicious")
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        _unpack_model(archive, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_export_refuses_secrets_without_echoing_them():
    secret = "sk-proj-" + "x" * 32
    with pytest.raises(ValueError) as error:
        audit_payloads({"client/main.py": f'key = "{secret}"'.encode()})
    assert "credential" in str(error.value)
    assert secret not in str(error.value)


def test_export_refuses_embedded_device_credential():
    with pytest.raises(ValueError, match="embedded credential"):
        audit_payloads({"client/config.py": b'local_key = "some_real_key_1234567890"'})


def test_export_rejects_bot_tokens_and_setup_preserves_backup(tmp_path):
    with pytest.raises(ValueError, match='Telegram bot credential'):
        audit_payloads({'README.md': ('1234567890:' + 'A' * 35).encode()})
    path = tmp_path / 'config.yaml'
    original = 'client:\n  server_url: ws://localhost/ws\n  client_id: saved-id\n'
    path.write_text(original, encoding='utf-8')
    save_settings(path, server_url='ws://localhost/ws', input_device=0,
                  camera_enabled=True, camera_index=1, workplace_name='Entrance',
                  camera_name='Wide view', camera_model='yolo11s.pt')
    assert next(tmp_path.glob('config.yaml.bak-*')).read_text(encoding='utf-8') == original
    cfg = load_client_config(path).client
    assert cfg.client_id == 'saved-id' and cfg.workplace_name == 'Entrance'
    assert cfg.camera.name == 'Wide view' and cfg.camera.model == 'yolo11s.pt'
    assert cfg.camera.fps == 0


@pytest.mark.parametrize("source", [b"import server.app", b"import hub.app",
                                    b"from common.config import load_config"])
def test_export_refuses_brain_dependencies(source):
    with pytest.raises(ValueError):
        audit_payloads({"client/main.py": source})


def test_clean_export_is_standalone_and_private_files_cannot_sneak_in(tmp_path):
    root = tmp_path / "source"
    for name in RUNTIME_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    for name in TEMPLATE_FILES:
        relative = "distribution/client/" + name
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    # Even future sensitive files under client/ must not be exported by glob.
    for name in ["config.yaml", ".env", ".git/config", "data/private.wav", "client/secret.py", "client/overlay_web/preview.html",
                 ".rowan-python", ".rowan-python.bak-20260920",
                 "distribution/client/.rowan-python", "distribution/client/.rowan-python.bak-20260920"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("PRIVATE_SENTINEL", encoding="utf-8")
    folder, archive_path, count = export_client(root, tmp_path / "release")
    allowed = set(RUNTIME_FILES) | set(TEMPLATE_FILES) | {"release-manifest.json"}
    assert count == len(allowed)
    with zipfile.ZipFile(archive_path) as archive:
        assert {name.split("/", 1)[1] for name in archive.namelist()} == allowed
        assert all(b"PRIVATE_SENTINEL" not in archive.read(name) for name in archive.namelist())
    code = '''
import importlib.abc, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class NoBrain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'server', 'hub'} or fullname in {'common.config', 'common.openai_models', 'common.image_models'}:
            raise RuntimeError('Standalone client imported a brain module: ' + fullname)
sys.meta_path.insert(0, NoBrain())
import client.main, common.client_config
assert Path(client.main.__file__).is_relative_to(Path(sys.argv[1]))
assert Path(common.client_config.__file__).is_relative_to(Path(sys.argv[1]))
from common.client_config import load_client_config
load_client_config(Path(sys.argv[1]) / 'config.example.yaml')
client.main.parse_args(['--help'])
'''
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(folder)],
                            cwd=folder, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    with pytest.raises(ValueError, match="already exists"):
        export_client(root, folder)
    manifest = json.loads((folder / "release-manifest.json").read_text())
    assert set(manifest["files"]) == allowed - {"release-manifest.json"}


@pytest.fixture
def client_powershell(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Client launchers require Windows")
    executable = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not executable:
        pytest.skip("PowerShell is unavailable")

    def run(code, **variables):
        env = os.environ.copy()
        env.pop("CONDA_PREFIX", None)
        env.pop("CONDA_DEFAULT_ENV", None)
        env.update({
            "ROWAN_TEST_ROOT": str(tmp_path),
            "ROWAN_TEST_RESOLVER": str(ROOT / "distribution/client/scripts/client-python.ps1"),
            **{key: str(value) for key, value in variables.items()},
        })
        script = "$ErrorActionPreference = 'Stop'\n. $env:ROWAN_TEST_RESOLVER\n" + code
        return subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )

    return run


@pytest.fixture
def supported_client_python(client_powershell):
    if sys.version_info[:2] not in {(3, 11), (3, 12)} or struct.calcsize("P") != 8:
        pytest.skip("A supported Python 3.11/3.12 64-bit interpreter is required")
    return Path(sys.executable).resolve()


@pytest.fixture
def default_client_python(tmp_path, supported_client_python):
    result = subprocess.run(
        [str(supported_client_python), "-m", "venv", "--without-pip", str(tmp_path / ".venv")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return tmp_path / ".venv/Scripts/python.exe"


def test_client_python_default_requires_local_environment_even_when_python_is_on_path(tmp_path, client_powershell):
    variables = {"PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    result = client_powershell("Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT", **variables)
    assert result.returncode != 0
    result = client_powershell(
        "ConvertTo-Json -Compress -InputObject (Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT -AllowMissingDefault)",
        **variables,
    )
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)) == tmp_path / ".venv/Scripts/python.exe"
    assert not (tmp_path / ".rowan-python").exists()


def test_client_python_resolves_existing_default(client_powershell, default_client_python):
    result = client_powershell(
        "ConvertTo-Json -Compress -InputObject (Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT)"
    )
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)) == default_client_python


def test_client_python_saved_selection_precedes_local_environment(tmp_path, client_powershell, supported_client_python):
    default = tmp_path / ".venv/Scripts/python.exe"
    default.parent.mkdir(parents=True)
    default.write_bytes(b"This is not an executable")
    (tmp_path / ".rowan-python").write_text(str(supported_client_python) + "\n", encoding="utf-8")
    result = client_powershell(
        "ConvertTo-Json -Compress -InputObject (Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT)"
    )
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)) == supported_client_python


@pytest.mark.parametrize("allow_missing", [False, True])
def test_client_python_missing_saved_selection_never_falls_back(tmp_path, client_powershell, default_client_python, allow_missing):
    selection = str(tmp_path / "removed conda environment/python.exe") + "\n"
    (tmp_path / ".rowan-python").write_text(selection, encoding="utf-8")
    result = client_powershell(
        "Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT" + (" -AllowMissingDefault" if allow_missing else "")
    )
    assert result.returncode != 0
    assert (tmp_path / ".rowan-python").read_text(encoding="utf-8") == selection


def test_client_python_conda_requires_activation_even_with_saved_and_default_python(tmp_path, client_powershell, default_client_python):
    (tmp_path / ".rowan-python").write_text(str(default_client_python), encoding="utf-8")
    result = client_powershell("Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT -Conda")
    assert result.returncode != 0


@pytest.mark.parametrize("explicit", [False, True])
def test_client_python_conda_selects_requested_environment(tmp_path, client_powershell, supported_client_python, explicit):
    (tmp_path / ".rowan-python").write_text(str(tmp_path / "stale/python.exe"), encoding="utf-8")
    variables = {"ROWAN_TEST_PYTHON": supported_client_python}
    if explicit:
        # An explicit environment wins even when activation points elsewhere.
        variables["CONDA_PREFIX"] = tmp_path / "other conda environment"
    else:
        if supported_client_python.name.lower() != "python.exe":
            pytest.skip("Activated Conda environments use python.exe on Windows")
        variables["CONDA_PREFIX"] = supported_client_python.parent
    result = client_powershell(
        "ConvertTo-Json -Compress -InputObject (Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT -Conda"
        + (" -Python $env:ROWAN_TEST_PYTHON" if explicit else "") + ")",
        **variables,
    )
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)) == supported_client_python


def test_client_python_missing_explicit_environment_never_falls_back(client_powershell, supported_client_python, tmp_path):
    result = client_powershell(
        "Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT -Conda -Python $env:ROWAN_TEST_PYTHON",
        ROWAN_TEST_PYTHON=tmp_path / "missing/python.exe", CONDA_PREFIX=supported_client_python.parent,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("selection", ["saved", "explicit", "conda"])
def test_client_python_rejects_non_python_executables(tmp_path, client_powershell, selection):
    invalid = tmp_path / "invalid environment/python.exe"
    invalid.parent.mkdir()
    invalid.write_bytes(b"This is not an executable")
    variables = {"ROWAN_TEST_PYTHON": invalid}
    arguments = ""
    if selection == "saved":
        (tmp_path / ".rowan-python").write_text(str(invalid), encoding="utf-8")
    elif selection == "explicit":
        arguments = " -Python $env:ROWAN_TEST_PYTHON"
    else:
        arguments = " -Conda"
        variables["CONDA_PREFIX"] = invalid.parent
    result = client_powershell("Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT" + arguments, **variables)
    assert result.returncode != 0


def test_client_python_save_preserves_previous_selection_bytes(tmp_path, client_powershell, supported_client_python):
    saved = tmp_path / ".rowan-python"
    previous = b"C:\\previous environment\\python.exe\r\n"
    saved.write_bytes(previous)
    result = client_powershell(
        "Save-RowanPython -Root $env:ROWAN_TEST_ROOT -Python $env:ROWAN_TEST_PYTHON",
        ROWAN_TEST_PYTHON=supported_client_python,
    )
    assert result.returncode == 0, result.stderr
    assert saved.read_text(encoding="utf-8-sig").strip() == str(supported_client_python)
    backups = list(tmp_path.glob(".rowan-python.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == previous
    result = client_powershell(
        "ConvertTo-Json -Compress -InputObject (Resolve-RowanPython -Root $env:ROWAN_TEST_ROOT)"
    )
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)) == supported_client_python
