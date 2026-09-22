"""ТЗ F-701: у каждого владельца дома свой чат и только его дома."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from common.config import Config
from hub.admin_backend import AdminBackend
from hub.telegram_admin import TelegramAdmin
from hub.telegram_admin_state import TelegramAdminState
from hub.telegram_homes import HomeOwners

OWNER, TENANT, STRANGER, GROUP, BOT = 8322835915, 555000111, 999000222, -10012345678, 777
HOMES = [SimpleNamespace(home_id="livingroom", name="Living room", telegram_user_id=TENANT),
         SimpleNamespace(home_id="office", name="Office", telegram_user_id=0)]


def access_state(tmp_path):
    return TelegramAdminState(tmp_path / "access.sqlite3", OWNER)


class Recorded:
    """The device store the panel asks for: it records the home it was given."""

    def __init__(self, items=()):
        self.asked: list = []
        self._items = list(items)
        self.store = self

    def switches(self, home_id=None):
        self.asked.append(home_id)
        return [row for row in self._items if not home_id or row.get("home_id") == home_id]

    def get(self, device_id):
        return None

    def scenes(self, home_id=None):
        self.asked.append(home_id)
        return []

    def resolve(self, home_id, scene_id):
        return None


def backend(tmp_path, *, owners=None, workplaces=None, switches=None):
    state = owners.access if owners is not None else access_state(tmp_path)
    places = workplaces if workplaces is not None else [
        {"id": "living", "name": "Living", "home_id": "livingroom", "connected": True},
        {"id": "office", "name": "Office", "home_id": "office", "connected": True},
        {"id": "other", "name": "Other", "home_id": "attic", "connected": False},
    ]
    cfg = SimpleNamespace(server=SimpleNamespace(telegram=SimpleNamespace(control_user_id=OWNER,
                                                                         chat_id=GROUP),
                                                permissions_enabled=True),
                          homes=HOMES)
    engine = AdminBackend(
        cfg, state, runtime=lambda: {}, get_room=lambda client_id=None: None,
        get_alerts=lambda: None, get_workplaces=lambda: places,
        get_switches=(lambda: switches) if switches is not None else (lambda: None),
        get_scope=(owners.scope if owners is not None else (lambda actor: None)),
        get_workplace_home=lambda client_id: next(
            (row["home_id"] for row in places if row["id"] == client_id), ""),
        get_home_owners=(lambda: owners) if owners is not None else (lambda: None))
    return engine, state, places


def owners_for(tmp_path, *, grants=((TENANT, "livingroom"),)):
    owners = HomeOwners(access_state(tmp_path), HOMES)
    for user_id, home_id in grants:
        owners.access.grant_home(user_id, home_id)
    return owners


# --- the ownership model -----------------------------------------------------


def test_a_home_names_its_telegram_owner_in_the_config():
    cfg = Config.model_validate({"homes": [{"home_id": "livingroom", "name": "Living",
                                            "telegram_user_id": TENANT},
                                           {"home_id": "office", "name": "Office"}]})
    assert cfg.homes[0].telegram_user_id == TENANT
    assert cfg.homes[1].telegram_user_id == 0


def test_seeding_writes_the_configured_owners_once(tmp_path):
    owners = HomeOwners(access_state(tmp_path), HOMES)
    assert owners.seed() == 1
    assert owners.seed() == 0
    assert owners.access.homes_of(TENANT) == frozenset({"livingroom"})
    assert owners.may_use_panel(TENANT) is True
    assert owners.scope(TENANT) == frozenset({"livingroom"})
    # Офис в конфиге владельца не имеет — и никому не достаётся.
    assert owners.access.homes_of(0) == frozenset()
    # Офис известен хабу (его можно выдать), но владельца у него нет.
    assert owners.owners()["office"] == ()
    assert owners.home_ids() == ("livingroom", "office")


def test_the_hub_admin_scope_is_every_home_and_nobody_else_gets_one(tmp_path):
    owners = owners_for(tmp_path)
    assert owners.scope(OWNER) is None
    assert owners.scope(STRANGER) == frozenset()
    assert owners.may_use_panel(OWNER) is True
    assert owners.may_use_panel(STRANGER) is False
    assert owners.scope("not-an-id") == frozenset()


def test_grants_are_per_home_and_survive_the_other_home(tmp_path):
    owners = owners_for(tmp_path)
    assert owners.grant(OWNER, TENANT, "office") is True
    assert owners.grant(OWNER, TENANT, "office") is False
    assert owners.access.homes_of(TENANT) == frozenset({"livingroom", "office"})
    assert owners.revoke(OWNER, TENANT, "livingroom") is True
    assert owners.access.homes_of(TENANT) == frozenset({"office"})
    assert owners.revoke(OWNER, TENANT, "livingroom") is False
    assert owners.access.owners_of("office") == (TENANT,)
    assert owners.access.owners_of("attic") == ()


def test_only_the_hub_admin_changes_owners_and_home_ids_are_checked(tmp_path):
    owners = owners_for(tmp_path)
    with pytest.raises(ValueError):
        owners.grant(TENANT, STRANGER, "office")
    with pytest.raises(ValueError):
        owners.revoke(TENANT, STRANGER, "office")
    with pytest.raises(ValueError):
        owners.access.grant_home(STRANGER, "../etc")
    with pytest.raises(ValueError):
        owners.access.grant_home(0, "office")
    assert owners.access.homes_of(STRANGER) == frozenset()


def test_removing_a_telegram_user_takes_their_homes_away(tmp_path):
    owners = owners_for(tmp_path)
    owners.access.remove_user(TENANT)
    assert owners.access.homes_of(TENANT) == frozenset()
    assert owners.access.home_owners() == {}
    assert owners.may_use_panel(TENANT) is False


def test_the_hub_admins_own_access_cannot_be_removed(tmp_path):
    with pytest.raises(ValueError):
        access_state(tmp_path).remove_user(OWNER)


# --- what a home owner may do -------------------------------------------------


def test_a_home_owner_sees_only_their_home_in_tools(tmp_path):
    owners = owners_for(tmp_path)
    engine, _, _ = backend(tmp_path, owners=owners)
    mine = asyncio.run(engine.call("workplaces.list", {}, TENANT))
    assert [row["id"] for row in mine["items"]] == ["living"]
    assert mine["selected_id"] is None
    every = asyncio.run(engine.call("workplaces.list", {}, OWNER))
    assert [row["id"] for row in every["items"]] == ["living", "office", "other"]


def test_a_home_owner_cannot_touch_a_foreign_home_or_a_hub_setting(tmp_path):
    owners = owners_for(tmp_path)
    engine, _, _ = backend(tmp_path, owners=owners)
    foreign = asyncio.run(engine.call("workplaces.photo", {"id": "office"}, TENANT))
    assert foreign["ok"] is False and "another owner" in foreign["error"]
    settings = asyncio.run(engine.call("settings.set", {"key": "server.x", "value": 1}, TENANT))
    assert settings["ok"] is False and "hub administrator" in settings["error"]
    people = asyncio.run(engine.call("profiles.list", {}, TENANT))
    assert people["ok"] is False
    homes = asyncio.run(engine.call("homes.grant", {"home_id": "office", "user_id": STRANGER}, TENANT))
    assert homes["ok"] is False
    assert owners.access.homes_of(STRANGER) == frozenset()


def test_a_single_home_owner_action_without_a_home_goes_to_their_home(tmp_path):
    owners = owners_for(tmp_path)
    switches = Recorded([{"id": "lamp", "home_id": "livingroom", "name": "Lamp"}])
    engine, _, _ = backend(tmp_path, owners=owners, switches=switches)
    result = asyncio.run(engine.call("devices.list", {}, TENANT))
    assert result["ok"] is True
    assert switches.asked == ["livingroom"]  # дом подставлен, а не угадан
    # Явно чужой дом по-прежнему отбит, даже у владельца одного дома.
    abroad = asyncio.run(engine.call("devices.list", {"home_id": "attic"}, TENANT))
    assert abroad["ok"] is False and "another owner" in abroad["error"]


def test_a_home_owner_may_not_act_without_a_home_when_they_own_several(tmp_path):
    owners = owners_for(tmp_path, grants=((TENANT, "livingroom"), (TENANT, "office")))
    switches = Recorded([])
    engine, _, _ = backend(tmp_path, owners=owners, switches=switches)
    result = asyncio.run(engine.call("devices.list", {}, TENANT))
    assert result["ok"] is False and "does not name a home" in result["error"]
    assert switches.asked == []


def test_an_account_without_homes_is_told_so(tmp_path):
    owners = owners_for(tmp_path)
    engine, _, _ = backend(tmp_path, owners=owners)
    result = asyncio.run(engine.call("status", {}, STRANGER))
    assert result["ok"] is False and "No homes are assigned" in result["error"]


def test_the_scoped_status_hides_hub_wide_numbers(tmp_path):
    owners = owners_for(tmp_path)
    engine, _, _ = backend(tmp_path, owners=owners)
    result = asyncio.run(engine.call("status", {}, TENANT))
    assert result["ok"] is True
    assert result["scoped_homes"] == ["livingroom"]
    assert "api_usage" not in result and "profiles" not in result
    assert [row["id"] for row in result["workplaces"]] == ["living"]


def test_the_hub_admin_lists_and_grants_homes(tmp_path):
    owners = owners_for(tmp_path)
    engine, state, _ = backend(tmp_path, owners=owners)
    listed = asyncio.run(engine.call("homes.list", {}, OWNER))
    assert listed["ok"] is True
    rows = {row["home_id"]: row for row in listed["items"]}
    assert rows["livingroom"]["owners"] == [TENANT]
    assert rows["livingroom"]["name"] == "Living room"
    granted = asyncio.run(engine.call("homes.grant", {"home_id": "office", "user_id": STRANGER}, OWNER))
    assert granted["ok"] is True and granted["added"] is True
    assert state.homes_of(STRANGER) == frozenset({"office"})
    revoked = asyncio.run(engine.call("homes.revoke", {"home_id": "office", "user_id": STRANGER}, OWNER))
    assert revoked["ok"] is True and revoked["removed"] is True
    assert state.homes_of(STRANGER) == frozenset()
    bad = asyncio.run(engine.call("homes.grant", {"home_id": "office", "user_id": "abc"}, OWNER))
    assert bad["ok"] is False and "numeric Telegram ID" in bad["error"]


# --- the panel itself ---------------------------------------------------------


class Provider:
    def __init__(self):
        self.sent, self.edited, self.answers = [], [], []
        self.latest = None

    async def send_text(self, text, **kwargs):
        record = {"text": text, "message_id": len(self.sent) + 100, **deepcopy(kwargs)}
        self.sent.append(record)
        self.latest = record
        return {"ok": True, "message_id": record["message_id"]}

    async def edit_text(self, text, **kwargs):
        self.latest = {"text": text, **deepcopy(kwargs)}
        self.edited.append(self.latest)
        return {"ok": True, "message_id": kwargs["message_id"]}

    async def answer_callback(self, callback_query_id, text="", show_alert=False):
        self.answers.append((callback_query_id, text, show_alert))


class PanelBackend:
    def __init__(self, places):
        self.calls, self.places = [], places

    async def call(self, action, payload, actor_id):
        self.calls.append((action, deepcopy(payload), actor_id))
        if action == "workplaces.list":
            return {"ok": True, "selected_id": None, "items": list(self.places)}
        if action == "homes.list":
            return {"ok": True, "items": [{"home_id": "livingroom", "name": "Living room",
                                           "owners": [TENANT]}]}
        return {"ok": True, "items": []}


def panel(tmp_path, *, places=(), grants=((TENANT, "livingroom"),)):
    owners = owners_for(tmp_path, grants=grants)
    provider, now = Provider(), [10.0]
    cfg = SimpleNamespace(control_user_id=OWNER, chat_id=GROUP)
    engine = TelegramAdmin(provider, cfg, owners.access,
                           PanelBackend([{"id": "living", "name": "Living", "home_id": "livingroom",
                                          "connected": True}, *places]),
                           clock=lambda: now[0], homes=owners)
    engine.set_identity(BOT, "RowanBot")
    return engine, provider, owners, now


def message(text, *, sender=TENANT, chat=None):
    chat = sender if chat is None else chat
    return {"message": {"message_id": 50, "text": text,
                        "from": {"id": sender, "is_bot": False},
                        "chat": {"id": chat, "type": "private" if chat > 0 else "supergroup"}}}


def button(provider, label):
    return next(item["callback_data"] for row in provider.latest["reply_markup"]["inline_keyboard"]
                for item in row if item["text"].startswith(label))


def click(provider, token, *, sender=TENANT, chat=None):
    chat = sender if chat is None else chat
    return {"callback_query": {"id": "query-" + token, "data": token,
            "from": {"id": sender, "is_bot": False},
            "message": {"message_id": provider.latest["message_id"],
                        "chat": {"id": chat, "type": "private" if chat > 0 else "supergroup"},
                        "from": {"id": BOT, "is_bot": True}}}}


def test_a_home_owner_opens_the_panel_in_their_own_chat(tmp_path):
    engine, provider, _, _ = panel(tmp_path)
    assert asyncio.run(engine.handle_update(message("/tools"))) is True
    assert len(provider.sent) == 1
    text = provider.sent[0]["text"]
    assert "livingroom" in text
    labels = [item["text"] for row in provider.sent[0]["reply_markup"]["inline_keyboard"] for item in row]
    assert "Computers and cameras" in labels and "Status" in labels
    # Общехабные разделы владельцу дома не показываются вовсе.
    assert not any(label in {"Settings", "Telegram users", "People profiles", "Audit log"}
                   for label in labels)


def test_the_hub_admin_still_gets_the_whole_menu(tmp_path):
    engine, provider, _, _ = panel(tmp_path)
    assert asyncio.run(engine.handle_update(message("/tools", sender=OWNER))) is True
    labels = [item["text"] for row in provider.sent[0]["reply_markup"]["inline_keyboard"] for item in row]
    assert "Settings" in labels and "Homes and owners" in labels


def test_a_stranger_and_the_group_get_no_panel(tmp_path):
    engine, provider, _, _ = panel(tmp_path)
    assert asyncio.run(engine.handle_update(message("/tools", sender=STRANGER))) is True
    assert provider.sent == []
    assert asyncio.run(engine.handle_update(message("/tools", sender=TENANT, chat=GROUP))) is True
    assert provider.sent == []


def test_the_expiring_callback_tokens_keep_working_for_a_home_owner(tmp_path):
    engine, provider, _, now = panel(tmp_path)
    asyncio.run(engine.handle_update(message("/tools")))
    token = button(provider, "Computers and cameras")
    assert asyncio.run(engine.handle_update(click(provider, token))) is True
    assert "Computers and cameras" in provider.latest["text"]
    # Кнопка живёт 15 минут: после этого она не работает и панель говорит,
    # что открыть заново можно только /tools.
    now[0] += 16 * 60
    old = button(provider, "Select:") if provider.latest.get("reply_markup") else token
    assert asyncio.run(engine.handle_update(click(provider, old))) is True
    assert provider.answers[-1][1].startswith("This button is unavailable")
