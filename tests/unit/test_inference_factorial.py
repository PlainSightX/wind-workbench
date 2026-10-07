"""配对协议、完整消费和质量门检查；模拟数据不作为GPU性能证据。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("inference_factorial",
    Path(__file__).parents[2] / "tools/diagnostics/inference_factorial.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    protocol = {"sha256": "protocol", "qualification_identity": {"model": "fixed"},
        "engine_identity": {"version": "fixed"}, "cells": module.optimization.CELLS,
        "schedule": module.schedule(["one", "two"])}
    for batch in protocol["schedule"]:
        cell = protocol["cells"][batch["label"]]
        rows = [{"case": case, "phase": "measurement", "protocol_sha256": "protocol",
            "batch_index": batch["batch_index"], "paired_round": batch["round"], "label": batch["label"],
            "batch_position": position, "layout": cell["layout"], "actual_prefix_caching": cell["prefix_cache"],
            "qualification_identity": protocol["qualification_identity"], "engine_identity": protocol["engine_identity"],
            "machine_pass": True, "measurement_valid": True, "task_elapsed_seconds": 10,
            "cold_reset_verified": position == 0,
            "calls": [{"content": "fixed", "ttft_seconds": 2, "usage": {"completion_tokens": 20},
                "cache_delta": {"vllm:prefix_cache_hits_total": 10 if position else 0,
                    "vllm:prefix_cache_queries_total": 30}}]}
            for position, case in enumerate(batch["case_order"])]
        path = tmp_path / module.result_name(batch)
        path.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    return protocol


def review_for(protocol, summary):
    return {"protocol_sha256": protocol["sha256"], "results": summary["results"],
        "reviewer": "Independent reviewer", "method": "Inspect each answer against frozen rubric",
        "cases": [{"batch_index": batch["batch_index"], "case": case, "accepted": True, "reason": "Checked"}
            for batch in protocol["schedule"] for case in batch["case_order"]]}


def test_same_round_uses_same_rotated_questions_in_all_cells():
    ids = ["N2", "N4", "N7", "M1", "M4", "B4"]
    plan = module.schedule(ids)
    assert len(plan) == 12
    for index in range(3):
        group = plan[index*4:index*4+4]
        assert {row["label"] for row in group} == {"A", "B", "C", "D"}
        assert all(row["case_order"] == ids[index:]+ids[:index] for row in group)
    assert ids == ["N2", "N4", "N7", "M1", "M4", "B4"]


@pytest.mark.parametrize("ids", [["one"], ["one", "one"]])
def test_same_question_replays_cannot_be_the_protocol(ids):
    with pytest.raises(ValueError):
        module.schedule(ids)


def test_no_semantic_review_is_never_adoption_eligible(tmp_path):
    protocol = fixture(tmp_path)
    result = module.summarize(protocol, tmp_path)
    assert result["tasks"] == 24 and result["semantic_review"] == "pending"
    assert result["quality_eligible_cells"] == []
    assert result["cells"]["D"]["cross_question_first_call_hits_tokens"] == 30


def test_review_negative_retains_failure_and_disqualifies_cell(tmp_path):
    protocol = fixture(tmp_path)
    summary = module.summarize(protocol, tmp_path)
    review = review_for(protocol, summary)
    review["cases"][0]["accepted"] = False
    result = module.summarize(protocol, tmp_path, review)
    assert result["cells"]["A"]["machine_pass"] == 6
    assert result["cells"]["A"]["semantic_pass"] == 5
    assert result["quality_eligible_cells"] == ["B", "C", "D"]


@pytest.mark.parametrize("field,value", [("phase", "warmup"), ("protocol_sha256", "wrong"),
    ("engine_identity", {}), ("actual_prefix_caching", True), ("layout", "evidence_first")])
def test_wrong_row_identity_rejected(tmp_path, field, value):
    protocol = fixture(tmp_path)
    batch = protocol["schedule"][0]
    path = tmp_path / module.result_name(batch)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError):
        module.summarize(protocol, tmp_path)


def test_partial_measurement_cannot_be_reported_as_full_result(tmp_path):
    protocol = fixture(tmp_path)
    path = tmp_path / module.result_name(protocol["schedule"][0])
    path.write_text(path.read_text().splitlines()[0]+"\n")
    with pytest.raises(ValueError, match="missing"):
        module.summarize(protocol, tmp_path)


@pytest.mark.parametrize("change", ["missing", "duplicate", "source", "empty_reason"])
def test_review_covers_exact_complete_sources(tmp_path, change):
    protocol = fixture(tmp_path)
    summary = module.summarize(protocol, tmp_path)
    review = deepcopy(review_for(protocol, summary))
    if change == "missing":
        review["cases"].pop()
    elif change == "duplicate":
        review["cases"].append(review["cases"][0])
    elif change == "source":
        review["results"] = {}
    else:
        review["cases"][0]["reason"] = ""
    with pytest.raises(ValueError):
        module.summarize(protocol, tmp_path, review)


def test_save_never_overwrites_existing_result(tmp_path):
    path = tmp_path / "record.json"
    module.save_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        module.save_new(path, {"first": False})
    assert json.loads(path.read_bytes()) == {"first": True}
