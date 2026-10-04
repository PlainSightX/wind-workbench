"""用注入草稿走真实助手流程，再由真实浏览器核对最终正文；不调用外部服务。"""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from langchain_core.messages import AIMessage
from power_forecast_service.assistant import workflow
from power_forecast_service.assistant.contracts import ContextRef, Question

IDENTITY = "00000000-0000-4000-8000-000000000123"


class Audit:
    def __init__(self):
        self.rows = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def begin(self):
        return self

    def add(self, value):
        self.rows.append(value)


class Provider:
    def __init__(self, drafts):
        self.drafts, self.calls = drafts, 0

    async def ainvoke(self, messages):
        value = self.drafts[self.calls]
        self.calls += 1
        return AIMessage(content=json.dumps(value))


def run(name, texts, value=12.5, stage=False):
    evidence = {
        "facts": [{"id": "c0.ridge.mae", "label": "ridge MAE", "value": value,
                   "unit": "kW", "aggregation": "离线显示fixture"}],
        "records": [{"models": {"ridge": {}}}], "scopes": [],
        "stage_requirements": [], "stage_options": [],
    }
    drafts = [{"status": "answered", "answer": text, "fact_ids": ["c0.ridge.mae"]} for text in texts]
    if stage:
        evidence["stage_requirements"] = [{"object_id": IDENTITY, "role": "selection_basis"}]
        evidence["stage_options"] = [{"object_id": IDENTITY, "role": "selection_basis",
            "label": "开发依据", "allowed_fact_ids": ["c0.ridge.mae"],
            "required_fact_ids": ["c0.ridge.mae"], "allowed_citations": [], "required_citations": []}]
        for draft in drafts:
            draft["stage_claims"] = [{"object_id": IDENTITY, "role": "selection_basis", "text": draft["answer"]}]
            draft["answer"] = ""

    async def get_results(*args):
        return evidence

    audit, provider = Audit(), Provider(drafts)
    with patch.object(workflow, "get_results", get_results), \
         patch.object(workflow, "compare_results", lambda data: data), \
         patch.object(workflow, "prepare_stage_requirements", lambda data, docs, question: (data, docs)), \
         patch.object(workflow, "corpus", lambda: {"sha256": "a" * 64, "chunks": []}):
        assistant = workflow.Assistant(lambda: audit, provider=provider)
        try:
            answer = asyncio.run(assistant.run(Question(question="MAE是多少？",
                contexts=[ContextRef(kind="engie_import", id=IDENTITY)]), direct=True))
        finally:
            assistant.close()
    assert len(audit.rows) == 1 and provider.calls == len(texts)
    return {"name": name, "answer": answer, "calls": provider.calls, "audit_rows": len(audit.rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="本次检查拥有的私有输出目录")
    parser.add_argument("--node", default=shutil.which("node"))
    parser.add_argument("--web-dependencies", type=Path, default=ROOT / "web/package.json",
                        help="已有前端依赖的package.json；复核候选时可指定原仓，不安装依赖")
    args = parser.parse_args()
    if not args.node:
        parser.error("需要已安装的Node.js及web依赖；本检查不安装依赖")
    cases = [
        run("repaired_label_sign", ["MAE=-ridge MAE：{{c0.ridge.mae}}。", "MAE为{{c0.ridge.mae}}。"]),
        run("repaired_unicode_sign", ["MAE=−ridge MAE: {{c0.ridge.mae}}。", "ridge MAE：{{c0.ridge.mae}}。"]),
        run("rejected_label_sign", ["MAE=-ridge MAE：ridge MAE：{{c0.ridge.mae}}。"] * 2),
        run("rejected_unit", ["{{c0.ridge.mae}} GW。"] * 2),
        run("negative_fact", ["偏差为{{c0.ridge.mae}}。"], -12.5),
        run("stage_repair", ["−ridge MAE：{{c0.ridge.mae}}。", "开发指标为{{c0.ridge.mae}}。"], stage=True),
        run("markdown_bullet", ["- ridge MAE：{{c0.ridge.mae}}。"]),
        run("same_unit", ["指标为{{c0.ridge.mae}} kW。"]),
        run("html_text", ['<img src="x" onerror="window.injected=true">{{c0.ridge.mae}}。']),
        run("model_name", ["ridge对应{{c0.ridge.mae}}。"]),
    ]
    expected = {
        "repaired_label_sign": "MAE为ridge MAE：12.5 kW。",
        "repaired_unicode_sign": "ridge MAE：12.5 kW。",
        "negative_fact": "偏差为ridge MAE：-12.5 kW。",
        "stage_repair": "开发依据：开发指标为ridge MAE：12.5 kW。",
        "markdown_bullet": "• ridge MAE：12.5 kW。",
        "same_unit": "指标为ridge MAE：12.5 kW。",
        "model_name": "ridge对应ridge MAE：12.5 kW。",
    }
    for case in cases:
        if case["name"].startswith("rejected_"):
            assert case["answer"]["status"] == "validation_error"
            assert case["calls"] == 2 and not case["answer"]["facts"]
        else:
            assert case["answer"]["status"] == "answered"
            if case["name"] in expected:
                assert case["answer"]["answer"] == expected[case["name"]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixture = output / "responses.json"
    fixture.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    result = subprocess.run([args.node, str(ROOT / "web/tests/assistant-offline.mjs"),
        str(ROOT), str(fixture), str(output), str(args.web_dependencies.resolve())],
        capture_output=True, text=True, encoding="utf-8", check=False, timeout=120)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
