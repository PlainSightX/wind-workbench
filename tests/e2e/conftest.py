"""只连接已启动的真实服务；不会替用户启动 Docker，也不会自动跳过。"""

import os

import httpx
import pytest


@pytest.fixture
def http_client():
    base_url = os.getenv("WIND_TEST_BASE_URL", "http://127.0.0.1:8000")
    with httpx.Client(base_url=base_url, timeout=10, trust_env=False) as client:
        yield client


@pytest.fixture
def completion_timeout():
    seconds = float(os.getenv("WIND_TEST_TIMEOUT", "180"))
    if seconds <= 0:
        raise ValueError("WIND_TEST_TIMEOUT must be positive")
    return seconds
