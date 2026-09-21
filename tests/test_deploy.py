"""Autostart and the single entry points (ТЗ 4.9)."""
from __future__ import annotations

import configparser
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- the make targets -------------------------------------------------------


def test_the_makefile_offers_the_targets_the_spec_names():
    makefile = read(REPO / "Makefile")
    for target in ("hub", "client", "test", "test-regress", "migrate", "skill"):
        assert re.search(rf"^{target}:", makefile, re.MULTILINE), target
    assert "-m hub.main" in makefile and "-m client.main" in makefile
    assert "-m hub.migrations_runner" in makefile
    assert "-m hub.skill_scaffold" in makefile


def test_make_test_is_the_check_that_agents_md_describes():
    makefile = read(REPO / "Makefile")
    body = makefile.split("test:", 1)[1].split("test-regress:", 1)[0]
    assert "-m pytest" in body and "-m ruff check ." in body and "-m mypy common" in body


# --- the Linux units --------------------------------------------------------


@pytest.mark.parametrize("name,module,wanted", [
    ("rowan-hub.service", "hub.main", "multi-user.target"),
    ("rowan-client.service", "client.main", "graphical.target"),
])
def test_systemd_units_run_the_right_module_and_restart(name, module, wanted):
    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.optionxform = str  # keys like ``StandardOutput`` keep their case
    unit.read_string(read(DEPLOY / "systemd" / name))
    service = unit["Service"]
    assert f"-m {module}" in service["ExecStart"]
    assert service["Restart"] == "always"
    assert unit["Unit"]["After"] == "network-online.target"
    assert unit["Install"]["WantedBy"] == wanted
    assert "config.yaml" in service["ExecStart"], "the service starts the shipped config"
    assert "EnvironmentFile" in service, "secrets come from the environment"
    assert "_responses" not in service["ExecStart"]


def test_the_linux_installer_enables_and_starts_the_hub():
    script = read(DEPLOY / "systemd" / "install.sh")
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    for step in ("daemon-reload", "enable rowan-hub.service", "restart rowan-hub.service"):
        assert step in script
    assert "rowan-client.service" in script


# --- the Windows tasks ------------------------------------------------------


def test_the_windows_installer_makes_hidden_hub_and_visible_client_tasks():
    script = read(DEPLOY / "windows" / "install-rowan.ps1")
    assert "Register-ScheduledTask" in script
    assert "-RestartCount 999" in script and "-RestartInterval" in script
    assert "hub.main" in script and "client.main" in script
    assert "New-ScheduledTaskTrigger -AtStartup" in script, "the hub starts with the machine"
    assert "New-ScheduledTaskTrigger -AtLogOn" in script, "the client needs the desktop session"
    assert "nssm" in script.casefold(), "NSSM is used when it is present"
    assert "C:\\Users\\Anton\\anaconda3\\envs\\jarvis\\python.exe" in script


def test_the_windows_uninstaller_only_removes_the_tasks():
    script = read(DEPLOY / "windows" / "uninstall-rowan.ps1")
    assert "Unregister-ScheduledTask" in script
    assert "RowanHub" in script and "RowanClient" in script
    assert "Remove-Item" not in script, "configuration and data are kept"


def test_the_windows_script_defaults_to_this_checkout():
    script = read(DEPLOY / "windows" / "install-rowan.ps1")
    assert "Split-Path -Parent" in script and "config.yaml" in script
    assert "Choose -Hub, -Client, or both." in script


# --- the documentation ------------------------------------------------------


def test_the_deploy_readme_names_both_operating_systems_and_the_targets():
    readme = read(DEPLOY / "README.md")
    for word in ("Windows", "Linux", "systemd", "NSSM", "make hub", "make client", "make test",
                 "make migrate", "make skill", "ROWAN_ADMIN_PASSWORD"):
        assert word in readme, word


def test_no_deploy_file_carries_a_secret_by_accident():
    for path in DEPLOY.rglob("*"):
        if path.is_file():
            text = read(path)
            assert not re.search(r"(?i)\b(api[_-]?key|bot[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", text), path
    assert json.loads(read(REPO / "firmware" / "esp32_switch" / "config.example.json"))[
        "wifi_password"] == "заполнить на месте"
