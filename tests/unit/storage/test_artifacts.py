"""目录变深后来源指纹必须覆盖嵌套代码，且不依赖安装位置。"""

import shutil

from power_forecast_service.storage.artifacts import source_tree_sha256


def test_source_fingerprint_tracks_nested_code_not_installation_root(tmp_path):
    root = tmp_path / "first"
    nested = root / "jobs"
    nested.mkdir(parents=True)
    source = nested / "worker.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = source_tree_sha256(root)
    copy = tmp_path / "second"
    shutil.copytree(root, copy)
    assert source_tree_sha256(copy) == before
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(root) != before
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source.rename(nested / "renamed.py")
    assert source_tree_sha256(root) != before


def test_source_fingerprint_ignores_runtime_cache(tmp_path):
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = source_tree_sha256(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"compiled")
    assert source_tree_sha256(tmp_path) == before
