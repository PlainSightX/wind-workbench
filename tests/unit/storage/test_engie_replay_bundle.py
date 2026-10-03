"""文件进入可信导入目录前的边界；不加载模型、不访问网络或数据库。"""

from hashlib import sha256
import json
import zipfile

import pytest

from power_forecast_service.storage import engie_replay_bundle as bundles


@pytest.fixture
def bundle_case(tmp_path, monkeypatch):
    source = b"lawful-original-data"
    prediction = b"original-predictions"
    model = b"original-model"
    protocol = b"original-frozen-protocol"
    result = json.dumps({"source": {"sha256": sha256(source).hexdigest(), "bytes": len(source)},
        "predictions": {"sha256": sha256(prediction).hexdigest()},
        "refit": {"artifacts": {"ridge": {"sha256": sha256(model).hexdigest()}}}}).encode()
    monkeypatch.setattr(bundles, "anchors", lambda release="final-2015": {
        "result_sha256": sha256(result).hexdigest(), "protocol_sha256": sha256(protocol).hexdigest()})
    payloads = {"result.json": result, "protocol.json": protocol,
                "predictions.npz": prediction, "ridge.joblib": model}
    source_path = tmp_path / "data.zip"
    source_path.write_bytes(source)
    return payloads, source_path


def archive_at(path, payloads):
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    return path


def test_prepare_exact_bytes_repeat_without_private_paths(tmp_path, bundle_case):
    payloads, source = bundle_case
    archive = archive_at(tmp_path / "bundle.zip", payloads)
    destination = tmp_path / "ready"
    first = bundles.prepare_bundle(archive, destination, source)
    assert bundles.prepare_bundle(archive, destination, source) == first
    assert first["training_performed"] is False
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == {
        **payloads, "source.zip": source.read_bytes()}


@pytest.mark.parametrize("member", ["ridge.joblib", "protocol.json", "result.json"])
def test_tampered_member_rejected_before_any_write(tmp_path, bundle_case, member):
    payloads, source = bundle_case
    payloads[member] = b"tampered"
    archive = archive_at(tmp_path / "bad.zip", payloads)
    destination = tmp_path / "ready"
    with pytest.raises(ValueError, match="hash_mismatch"):
        bundles.prepare_bundle(archive, destination, source)
    assert not destination.exists()


def test_traversal_member_rejected(tmp_path, bundle_case):
    payloads, source = bundle_case
    payloads["../outside"] = b"untrusted"
    destination = tmp_path / "ready"
    with pytest.raises(ValueError, match="archive_invalid"):
        bundles.prepare_bundle(archive_at(tmp_path / "bad.zip", payloads), destination, source)
    assert not destination.exists()


def test_conflicting_existing_file_not_overwritten(tmp_path, bundle_case):
    payloads, source = bundle_case
    destination = tmp_path / "ready"
    destination.mkdir()
    (destination / "ridge.joblib").write_bytes(b"existing-user-model")
    with pytest.raises(ValueError, match="destination_conflict"):
        bundles.prepare_bundle(archive_at(tmp_path / "bundle.zip", payloads), destination, source)
    assert list(destination.iterdir()) == [destination / "ridge.joblib"]
    assert (destination / "ridge.joblib").read_bytes() == b"existing-user-model"


def test_wrong_source_rejected_before_writes(tmp_path, bundle_case):
    payloads, source = bundle_case
    source.write_bytes(b"x" * source.stat().st_size)
    with pytest.raises(ValueError, match="hash_mismatch"):
        bundles.prepare_bundle(archive_at(tmp_path / "bundle.zip", payloads), tmp_path / "ready", source)
    assert not (tmp_path / "ready").exists()


def test_duplicate_members_rejected(tmp_path, bundle_case):
    payloads, source = bundle_case
    archive = archive_at(tmp_path / "bundle.zip", payloads)
    with zipfile.ZipFile(archive, "a") as handle, pytest.warns(UserWarning):
        handle.writestr("ridge.joblib", payloads["ridge.joblib"])
    with pytest.raises(ValueError, match="archive_invalid"):
        bundles.prepare_bundle(archive, tmp_path / "ready", source)


def test_export_failure_does_not_publish_or_block_retry(tmp_path, bundle_case, monkeypatch):
    payloads, _ = bundle_case
    repository = tmp_path / "original"
    repository.mkdir()
    result = json.loads(payloads["result.json"])
    for name, record in bundles.asset_records(result).items():
        record["path"] = name
        (repository / name).write_bytes(payloads[name])
    result_bytes = json.dumps(result).encode()
    (repository / "result.json").write_bytes(result_bytes)
    (repository / "protocol.json").write_bytes(payloads["protocol.json"])
    monkeypatch.setattr(bundles, "anchors", lambda release="final-2015": {
        "result_path": "result.json", "result_sha256": sha256(result_bytes).hexdigest(),
        "protocol_path": "protocol.json", "protocol_sha256": sha256(payloads["protocol.json"]).hexdigest()})
    result["source"]["bytes"] = 20
    # 在ZIP已开始写入后失败，而不是仅在入口制造一个失败。
    original = zipfile.ZipFile.writestr
    def broken(self, name, data, *args, **kwargs):
        original(self, name, data, *args, **kwargs)
        raise OSError("injected_disk_failure")
    output = tmp_path / "export.zip"
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, "writestr", broken)
        with pytest.raises(OSError, match="injected_disk_failure"):
            bundles.export_bundle(repository, output)
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial"))
    receipt = bundles.export_bundle(repository, output)
    assert receipt["bundle_sha256"] == sha256(output.read_bytes()).hexdigest()


def test_prepare_interrupted_publication_resumes_exact_members(tmp_path, bundle_case, monkeypatch):
    payloads, source = bundle_case
    archive = archive_at(tmp_path / "bundle.zip", payloads)
    destination = tmp_path / "ready"
    original = bundles.os.link
    calls = []
    def interrupted(first, second):
        calls.append(second)
        if len(calls) == 2:
            raise OSError("injected_publication_failure")
        return original(first, second)
    with monkeypatch.context() as patch:
        patch.setattr(bundles.os, "link", interrupted)
        with pytest.raises(OSError, match="publication_failure"):
            bundles.prepare_bundle(archive, destination, source)
    assert len(list(destination.iterdir())) == 1
    bundles.prepare_bundle(archive, destination, source)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == {
        **payloads, "source.zip": source.read_bytes()}
