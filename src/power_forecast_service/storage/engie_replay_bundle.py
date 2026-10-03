"""原评分包的可移植获取边界；校验完成前不解包、不加载 pickle、不连接数据库。"""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

SOURCE_URL = (
    "https://raw.githubusercontent.com/NatLabRockies/OpenOA/"
    "9bfc7a3dc542b17bbfc06b692dbfb8b23c754975/examples/data/la_haute_borne.zip"
)
MAX_MEMBER_BYTES = 100_000_000


def checked(payload: bytes, expected: str) -> bytes:
    if sha256(payload).hexdigest() != expected:
        raise ValueError("engie_replay_hash_mismatch")
    return payload


def anchors(release="final-2015") -> dict:
    if release == "final-2015":
        return json.loads(Path(__file__).with_name("engie_final_source.json").read_text("utf-8"))
    if release != "development-2014":
        raise ValueError("engie_replay_release_invalid")
    trust = json.loads(Path(__file__).with_name("engie_sources.json").read_text("utf-8"))
    return {"result_path": trust["baseline_path"], "result_sha256": trust["baseline_sha256"],
        "protocol_path": trust["protocol_path"], "protocol_sha256": trust["protocol_sha256"]}


def asset_records(result: dict, release="final-2015") -> dict:
    if release == "development-2014":
        assets = {}
        for quarter, window in result["windows"].items():
            assets[f"{quarter}.npz"] = window["predictions"]
            for family, record in window["fits"].items():
                assets[f"{quarter}-{family}.joblib"] = record["artifact"]
        return assets
    return {"predictions.npz": result["predictions"], **{
        f"{family}.joblib": record for family, record in result["refit"]["artifacts"].items()
    }}


def safe_path(root: Path, relative: str) -> Path:
    root = root.absolute()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("engie_replay_path_conflict")
    target = root / relative_path
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ValueError("engie_replay_path_conflict")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("engie_replay_path_conflict")
    return target


def export_bundle(repository: Path, output: Path, release="final-2015") -> dict:
    """仅维护者从历史原件导出；公开消费者不读取 result 中的本机路径。"""
    trust = anchors(release)
    result_name = "baseline.json" if release == "development-2014" else "result.json"
    payloads = {
        result_name: checked(safe_path(repository, trust["result_path"]).read_bytes(), trust["result_sha256"]),
        "protocol.json": checked(safe_path(repository, trust["protocol_path"]).read_bytes(), trust["protocol_sha256"]),
    }
    result = json.loads(payloads[result_name])
    for name, record in asset_records(result, release).items():
        payloads[name] = checked(safe_path(repository, record["path"]).read_bytes(), record["sha256"])
    # 原始数据另从许可来源下载，不把完整数据、日志或私人目录塞进模型附件。
    output = Path(output)
    safe_path(output.parent, output.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, payload in sorted(payloads.items()):
                    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, payload)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()
    return {"bundle_sha256": sha256(output.read_bytes()).hexdigest(), "bytes": output.stat().st_size,
            "files": {name: {"sha256": sha256(value).hexdigest(), "bytes": len(value)} for name, value in payloads.items()},
            "result_sha256": trust["result_sha256"], "protocol_sha256": trust["protocol_sha256"],
            "source": {"url": SOURCE_URL, "sha256": result["source"]["sha256"], "bytes": result["source"]["bytes"]}}


def bundle_payloads(bundle: Path, release="final-2015") -> tuple[dict, dict]:
    """清单身份来自源码锚，不信任压缩包自报的 hash；所有成员校验后才允许写入。"""
    trust = anchors(release)
    result_name = "baseline.json" if release == "development-2014" else "result.json"
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(info.file_size > MAX_MEMBER_BYTES for info in infos):
            raise ValueError("engie_replay_archive_invalid")
        result_bytes = checked(archive.read(result_name), trust["result_sha256"])
        result = json.loads(result_bytes)
        expected = {result_name: {"sha256": trust["result_sha256"]},
                    "protocol.json": {"sha256": trust["protocol_sha256"]}, **asset_records(result, release)}
        if set(names) != set(expected):
            raise ValueError("engie_replay_archive_invalid")
        payloads = {name: checked(archive.read(name), record["sha256"]) for name, record in expected.items()}
    return payloads, result


def prepare_bundle(bundle: Path, destination: Path, source_zip: Path | None = None, release="final-2015") -> dict:
    """输入是 release 附件和外部数据；失败不覆盖已有导入源，重复准备要求原字节一致。"""
    payloads, result = bundle_payloads(bundle, release)
    source_record = result["source"]
    if source_zip is None:
        # 只从固定公开来源下载；hash 是原评价身份，不接受近似或更新后的数据。
        with urlopen(SOURCE_URL, timeout=90) as response:
            payload = response.read(source_record["bytes"] + 1)
    else:
        payload = Path(source_zip).read_bytes()
    if len(payload) != source_record["bytes"]:
        raise ValueError("engie_replay_source_size_mismatch")
    payloads["source.zip"] = checked(payload, source_record["sha256"])
    targets = {name: safe_path(Path(destination), name) for name in payloads}
    for name, target in targets.items():
        if target.exists() and target.read_bytes() != payloads[name]:
            raise ValueError("engie_replay_destination_conflict")
    for name, target in targets.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # 链接原子发布完整文件，拒绝覆盖；中断不会留下看似就绪的半个模型包。
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payloads[name])
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink()
    return {"files": len(payloads), "result_sha256": anchors(release)["result_sha256"],
            "source_sha256": source_record["sha256"], "training_performed": False}
