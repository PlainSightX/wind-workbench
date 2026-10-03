"""维护模板只生成公开根README；链接从消费者目录解析。"""

import importlib.util
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]


def test_public_readme_navigation_and_attributes_have_one_export_target():
    spec = importlib.util.spec_from_file_location("public_export_navigation", ROOT / "tools/dev/export_public_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.public_files(ROOT)
    assert files["README.md"] == "docs/public-README.md"
    assert "docs/public-README.md" not in files
    assert files[".gitattributes"] == ".gitattributes"
    readme = (ROOT / files["README.md"]).read_text(encoding="utf-8")
    for href in re.findall(r"\[[^]]+\]\(([^)]+)\)", readme):
        if "://" not in href and not href.startswith("#"):
            assert PurePosixPath(href.split("#")[0]).as_posix() in files
