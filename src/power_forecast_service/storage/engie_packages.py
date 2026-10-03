"""ENGIE文件信任边界；数值推理可复用，不引入ORM、连接或导入事务。"""

from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

from .model_packages import PackageError


def trust():
    return json.loads(Path(__file__).with_name("engie_sources.json").read_text(encoding="utf-8"))


def checked_bytes(root, relative, expected):
    """hash来自代码锚定来源或已验证DB；拒绝越界和软链接后再读取。"""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise PackageError("engie_package_integrity_failed")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise PackageError("engie_package_integrity_failed")
    if not path.resolve().is_relative_to(root):
        raise PackageError("engie_package_integrity_failed")
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise PackageError("engie_package_missing") from exc
    if sha256(payload).hexdigest() != expected:
        raise PackageError("engie_package_integrity_failed")
    return payload


def write_verified(path, payload):
    """中断最多留下未登记临时文件；不同内容和软链接均不覆盖。"""
    path = Path(path)
    if any(parent.is_symlink() for parent in [path, *path.parents]):
        raise PackageError("engie_import_destination_conflict")
    if path.exists():
        if path.read_bytes() != payload:
            raise PackageError("engie_import_destination_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
