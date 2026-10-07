"""服务验证器使用实际 API 合同，不把自身缺陷记为模型失败。"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("inference_service", Path(__file__).resolve().parents[2] / "tools/diagnostics/inference_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def test_api_revision_and_contiguous_quote():
    documents = {"document": {"source_sha256": "a" * 64, "text": "原文第一段。原文第二段。"}}
    service.verify_citations([{"id": "document", "revision": "a" * 64, "quote": "原文第一段。"}], documents)


def test_process_ports_are_distinct_and_do_not_rebind_proxy():
    assert [service.service_port(18113, 18114, index) for index in range(4)] == [18113, 18115, 18116, 18117]


@pytest.mark.parametrize("changes", [{"revision": "b" * 64}, {"id": "other"}, {"quote": "伪造原文"}, {"quote": ""}])
def test_wrong_identity_and_quote_rejected(changes):
    documents = {"document": {"source_sha256": "a" * 64, "text": "原文第一段。"}}
    citation = {"id": "document", "revision": "a" * 64, "quote": "原文第一段。"} | changes
    with pytest.raises(ValueError):
        service.verify_citations([citation], documents)
