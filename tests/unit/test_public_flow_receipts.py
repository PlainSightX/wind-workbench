"""回放命令的磁盘/HTTP边界；不连接PG、不发送HTTP或创建模型。"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def verifier():
    spec = importlib.util.spec_from_file_location("public_flow_receipts", ROOT / "tools/dev/verify_public_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("conflict", ["output", "pending", "readback"])
def test_output_conflict_is_rejected_before_any_verification(tmp_path, monkeypatch, conflict):
    module = verifier()
    output = tmp_path / "replay.json"
    existing = output if conflict == "output" else output.with_name("replay.json.pending.json") if conflict == "pending" else output.with_name("restart-readback.json")
    existing.write_text("preserved", encoding="utf-8")
    if conflict == "readback":
        output.write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(module, "verify", lambda *a, **k: calls.append((a, k)))
    args = ["--output", str(output)] + (["--readback"] if conflict == "readback" else [])
    monkeypatch.setattr(sys, "argv", ["verify_public_flow", *args])
    with pytest.raises(FileExistsError):
        module.main()
    assert calls == [] and existing.read_text() == "preserved"


def fake_success(base, output, readback=False, q1_run_id=None, *, request_key=None, delivery_prepared=None):
    assert request_key
    delivery_prepared({"request_key": request_key, "artifact_id": "fake-artifact", "history": []})
    return {"status": "passed", "replays": [], "delivery": {"id": "fake-delivery"}, "training_performed": False}


def test_failed_receipt_publication_retains_completed_recoverable_pending(tmp_path, monkeypatch):
    module = verifier()
    output = tmp_path / "replay.json"
    monkeypatch.setattr(module, "verify", fake_success)
    original_link = module.os.link
    def fail_final(source, target):
        if Path(target) == output:
            raise OSError("fake final publish failure")
        return original_link(source, target)
    monkeypatch.setattr(module.os, "link", fail_final)
    with pytest.raises(OSError, match="fake final publish failure"):
        module.main(["--output", str(output)])
    assert not output.exists()
    pending = output.with_name("replay.json.pending.json")
    saved = json.loads(pending.read_text())
    assert saved["status"] == "complete" and saved["result"]["delivery"]["id"] == "fake-delivery"
    monkeypatch.setattr(module.os, "link", original_link)
    module.main(["--output", str(output), "--recover-receipt"])
    assert json.loads(output.read_text()) == saved["result"]
    assert not pending.exists() and list(tmp_path.iterdir()) == [output]


def test_uncertain_delivery_preserves_same_key_and_blocks_blind_retry(tmp_path, monkeypatch):
    module = verifier()
    output = tmp_path / "replay.json"
    keys = []
    def unknown(base, output, readback=False, q1_run_id=None, *, request_key=None, delivery_prepared=None):
        keys.append(request_key)
        delivery_prepared({"request_key": request_key, "history": []})
        raise TimeoutError("fake remote result unknown")
    monkeypatch.setattr(module, "verify", unknown)
    with pytest.raises(TimeoutError):
        module.main(["--output", str(output)])
    saved = json.loads(output.with_name("replay.json.pending.json").read_text())
    assert saved["delivery_request"]["request_key"] == keys[0]
    assert saved["status"] == "delivery_prepared"
    with pytest.raises(FileExistsError):
        module.main(["--output", str(output)])
    assert len(keys) == 1
    with pytest.raises(ValueError, match="receipt_not_complete"):
        module.main(["--output", str(output), "--recover-receipt"])


def test_initial_receipt_disk_failure_has_no_http_side_effect(tmp_path, monkeypatch):
    module = verifier()
    calls = []
    monkeypatch.setattr(module, "verify", lambda *a, **k: calls.append(1))
    def fail(*args):
        raise OSError("fake fsync failure")
    monkeypatch.setattr(module.os, "fsync", fail)
    with pytest.raises(OSError, match="fake fsync failure"):
        module.main(["--output", str(tmp_path / "replay.json")])
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_readback_receipt_can_recover_without_repeating_http(tmp_path, monkeypatch):
    module = verifier()
    source = tmp_path / "replay.json"
    source.write_text("{}")
    destination = tmp_path / "restart-readback.json"
    calls = []
    def readback(*args, **kwargs):
        calls.append(1)
        return {"status": "passed", "replays": 15, "persisted_delivery_identical": True}
    monkeypatch.setattr(module, "verify", readback)
    original_link = module.os.link
    def fail_final(source, target):
        if Path(target) == destination:
            raise OSError("fake readback receipt publish failure")
        return original_link(source, target)
    monkeypatch.setattr(module.os, "link", fail_final)
    with pytest.raises(OSError, match="fake readback receipt publish failure"):
        module.main(["--output", str(source), "--readback"])
    monkeypatch.setattr(module.os, "link", original_link)
    module.main(["--output", str(destination), "--recover-receipt"])
    assert calls == [1]
    assert json.loads(destination.read_text())["replays"] == 15
    assert not destination.with_name(destination.name + ".pending.json").exists()


def test_readback_claims_readback_without_guessing_process_identity(tmp_path, monkeypatch):
    module = verifier()
    output = tmp_path / "replay.json"
    output.write_text(json.dumps({"replays": [{"request": {}, "response": {"value": 1}}], "delivery": {"id": "fake"}}))
    monkeypatch.setattr(module.Settings, "from_environment", lambda: None)
    monkeypatch.setattr(module, "client", lambda _: lambda method, path, body=None: {"id": "fake"} if method == "GET" else {"value": 1})
    result = module.verify("http://unused.invalid", output, readback=True)
    assert result["persisted_delivery_identical"] is True
    assert "new_process" not in result
