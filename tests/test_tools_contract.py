"""server/tools.py: the tool schemas and the client/server execution matrix."""
from server.tools import (
    CLIENT_TOOLS,
    MOUSE_CLICK_TOOL,
    SERVER_TOOLS,
    TOOLS,
    actions_from_tool_calls,
    is_client_tool,
    mouse_click_args,
    normalize_click_button,
)


def tool_names():
    return {t["function"]["name"] for t in TOOLS}


def test_eleven_tools_exposed():
    assert tool_names() == {
        "set_light",
        "set_switch",
        "pc_control",
        "run_command",
        "look_at_screen",
        "click_screen",
        "remember",
        "enroll_voice",
        "set_role",
        "look_at_camera",
        "enroll_face",
    }


def test_mouse_click_is_internal_only():
    assert MOUSE_CLICK_TOOL not in tool_names()
    assert MOUSE_CLICK_TOOL not in SERVER_TOOLS and MOUSE_CLICK_TOOL not in CLIENT_TOOLS


def test_matrix_split():
    assert SERVER_TOOLS == {
        "look_at_screen",
        "click_screen",
        "remember",
        "enroll_voice",
        "set_role",
        "look_at_camera",
        "enroll_face",
    }
    assert CLIENT_TOOLS == {"set_light", "set_switch", "pc_control", "run_command"}
    assert is_client_tool("pc_control") and not is_client_tool("click_screen")
    # v1.4: the camera tools run on the server, like the screen ones.
    assert not is_client_tool("look_at_camera") and not is_client_tool("enroll_face")
    assert SERVER_TOOLS.isdisjoint(CLIENT_TOOLS)
    assert set(tool_names()) == SERVER_TOOLS | CLIENT_TOOLS


def test_camera_tools_take_their_one_argument():
    camera = next(t for t in TOOLS if t["function"]["name"] == "look_at_camera")
    assert camera["function"]["parameters"]["required"] == ["query"]
    enroll = next(t for t in TOOLS if t["function"]["name"] == "enroll_face")
    assert enroll["function"]["parameters"]["required"] == ["name"]


def test_pc_control_command_enum():
    pc = next(t for t in TOOLS if t["function"]["name"] == "pc_control")
    commands = set(pc["function"]["parameters"]["properties"]["command"]["enum"])
    for required in ("minimize_app", "maximize_app", "focus_app", "type_text", "hotkey"):
        assert required in commands, required


def test_click_args_clamped():
    args = mouse_click_args(1.7, -0.3, "double click")
    assert args["x_norm"] == 1.0 and args["y_norm"] == 0.0
    assert args["button"] == "double"
    assert normalize_click_button("nonsense") == "left"


def test_server_tools_never_forwarded():
    calls = [
        {"function": {"name": "click_screen", "arguments": {"target": "x"}}},
        {"function": {"name": "look_at_camera", "arguments": {"query": "who is here"}}},
        {"function": {"name": "enroll_face", "arguments": {"name": "Anton"}}},
        {"function": {"name": "pc_control", "arguments": {"command": "mute"}}},
    ]
    items = actions_from_tool_calls(calls)
    names = [item["tool"] for item in items]
    assert "click_screen" not in names
    assert "look_at_camera" not in names and "enroll_face" not in names
    assert "pc_control" in names
