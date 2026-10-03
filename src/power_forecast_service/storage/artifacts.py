"""文件校验与执行来源记录；文件存在本身不代表业务成功。"""

import hashlib
import platform
from importlib.metadata import version
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FINGERPRINT_VERSION = "package-recursive-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path = PACKAGE_ROOT) -> str:
    """包内相对路径和内容共同参与；嵌套模块不遗漏，安装绝对路径不参与。"""
    source = hashlib.sha256()
    paths = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        source.update(path.relative_to(root).as_posix().encode())
        source.update(b"\0")
        source.update(bytes.fromhex(sha256_file(path)))
    return source.hexdigest()


def execution_provenance() -> dict:
    return {
        "source_tree_sha256": source_tree_sha256(),
        "source_fingerprint_version": SOURCE_FINGERPRINT_VERSION,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "celery")},
        "runtime": platform.platform(),
    }
