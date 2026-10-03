"""全局只放通用路径与分类；数据库/网络夹具归各自测试目录。"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line("markers", "original_replay: requires explicitly prepared original model attachments")


@pytest.fixture(scope="session")
def sample_path() -> Path:
    return ROOT / "data/sample/wind_2019_q1.csv"


def pytest_collection_modifyitems(items):
    # 分类由位置确定，移动测试时不依赖手工维护一份平行清单。
    for item in items:
        parts = item.path.relative_to(ROOT / "tests").parts
        if parts[0] in {"unit", "integration", "e2e"}:
            item.add_marker(getattr(pytest.mark, parts[0]))
        if parts[:2] == ("integration", "postgres"):
            item.add_marker(pytest.mark.postgres)
