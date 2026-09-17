"""server/storage.py: dialog log and memory on a temp directory."""
import json

from server.storage import DialogLog, Memory


def test_dialog_append_creates_dated_file(tmp_path):
    log = DialogLog(data_dir=tmp_path)
    log.append({"transcript": "hi", "reply": "hello"})
    files = list((tmp_path / "dialogs").glob("*.jsonl"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert entry["transcript"] == "hi" and "ts" in entry


def test_memory_roundtrip_and_order(tmp_path):
    memory = Memory(data_dir=tmp_path)
    assert memory.facts() == []
    memory.add("first fact")
    memory.add("second fact")
    assert memory.facts() == ["first fact", "second fact"]


def test_memory_survives_garbage_lines(tmp_path):
    memory = Memory(data_dir=tmp_path)
    memory.add("good fact")
    with open(memory.path, "a", encoding="utf-8") as handle:
        handle.write("{broken json\n\n")
    assert memory.facts() == ["good fact"]
