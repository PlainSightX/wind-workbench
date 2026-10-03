"""比较的是同一真实评分问题；不能靠行数、旧汇总或碰巧同分推断可比。"""

from copy import deepcopy
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from power_forecast_service.experiments.comparison import compare_results
from power_forecast_service.experiments.comparison import selected_model_version
from power_forecast_service.forecasting.scoring import (
    ScoreRow, ScoringEvidence, metrics_for_rows, sample_fingerprint,
)


@pytest.fixture
def result():
    rows = [
        ScoreRow(
            cutoff=datetime(2020, 1, 1) + timedelta(minutes=5 * index),
            target_time=datetime(2020, 1, 1, 1) + timedelta(minutes=5 * index),
            actual=actual, predictions={"baseline": baseline, "candidate": candidate},
        )
        for index, (actual, baseline, candidate) in enumerate(((10, 8, 11), (20, 16, 21)))
    ]
    scoring = ScoringEvidence(
        version="scoring-v1", target="wind_power_single_point", unit="source_reported_unit",
        clock="source_time_timezone_unknown", evaluation_split="validation",
        metric_version="unweighted-mae-rmse-v1",
        rows=rows, samples_sha256=sample_fingerprint(rows), input_sha256="a" * 64,
        horizon_minutes=60, split_version="split-v1",
    )
    return {
        "purpose": "development", "evaluation_split": "validation", "horizon_minutes": 60,
        "horizon_steps": 12, "split_version": "split-v1",
        "input_file_sha256": "a" * 64,
        "scoring": scoring.model_dump(mode="json"),
        "metrics": {name: metrics_for_rows(rows, name) for name in ("baseline", "candidate")},
        "execution": {"runtime": "test-runtime"},
    }


def compare(left, right):
    return compare_results(uuid4(), left, "baseline", uuid4(), right, "candidate")


def test_comparison_recomputes_known_metrics_and_has_explicit_delta_direction(result):
    value = compare(result, result)
    assert value.status == "comparable"
    assert value.left.metrics["mae"] == 3
    assert value.right.metrics["mae"] == 1
    assert value.delta["mae"] == -2
    assert value.delta["rmse"] == pytest.approx(1 - (10 ** 0.5))


def test_different_training_configuration_is_allowed_but_runtime_difference_is_visible(result):
    other = deepcopy(result)
    other.update(model_version="different", determinism={"random_seed": 100})
    other["execution"] = {"runtime": "different-platform"}
    value = compare(result, other)
    assert value.status == "comparable"
    assert value.warnings[0].startswith("execution_context_differs:")
    assert value.right.context["determinism"] == {"random_seed": 100}


@pytest.mark.parametrize("change,reason", [
    ("legacy", "right:scoring_evidence_missing"),
    ("changed_label", "right:scoring_evidence_invalid"),
    ("duplicate", "right:scoring_evidence_invalid"),
    ("nan", "right:scoring_evidence_invalid"),
    ("wrong_horizon", "right:scoring_evidence_invalid"),
    ("partial_prediction", "right:scoring_evidence_invalid"),
    ("stale_summary", "right:metric_summary_mismatch"),
    ("different_input", "different:input_sha256"),
    ("different_samples", "different:samples_sha256"),
])
def test_incompatible_or_damaged_evidence_never_returns_a_delta(result, change, reason):
    other = deepcopy(result)
    scoring = other["scoring"]
    if change == "legacy":
        del other["scoring"]
    elif change == "changed_label":
        scoring["rows"][0]["actual"] = 99
    elif change == "duplicate":
        scoring["rows"][1] = deepcopy(scoring["rows"][0])
    elif change == "nan":
        scoring["rows"][0]["predictions"]["candidate"] = float("nan")
    elif change == "wrong_horizon":
        scoring["horizon_minutes"] = 30
    elif change == "partial_prediction":
        del scoring["rows"][0]["predictions"]["candidate"]
    elif change == "stale_summary":
        other["metrics"]["candidate"]["mae"] = 0
    elif change == "different_input":
        scoring["input_sha256"] = other["input_file_sha256"] = "b" * 64
    elif change == "different_samples":
        scoring["rows"][0]["actual"] = 11
        rows = [ScoreRow.model_validate(row) for row in scoring["rows"]]
        scoring["samples_sha256"] = sample_fingerprint(rows)
        other["metrics"]["candidate"] = metrics_for_rows(rows, "candidate")
    value = compare(result, other)
    assert value.status == "not_comparable"
    assert reason in value.reasons
    assert value.delta is None


def test_unknown_model_is_explained_without_key_error(result):
    value = compare_results(uuid4(), result, "unknown", uuid4(), result, "candidate")
    assert value.reasons == ["left:model_not_scored"]
    assert value.delta is None


@pytest.mark.parametrize("field,value", [
    ("purpose", "final_evaluation"), ("evaluation_split", "test"),
    ("horizon_minutes", 30), ("horizon_steps", 6), ("split_version", "another-split"),
])
def test_contradictory_outer_protocol_is_not_accepted(result, field, value):
    other = deepcopy(result)
    other[field] = value
    value = compare(result, other)
    assert "right:scoring_protocol_mismatch" in value.reasons
    assert value.delta is None


def test_missing_protocol_is_not_silently_filled_from_current_defaults(result):
    other = deepcopy(result)
    del other["scoring"]["metric_version"]
    value = compare(result, other)
    assert "right:scoring_evidence_invalid" in value.reasons
    assert value.delta is None


@pytest.mark.parametrize("broken", [None, [], 1, {"candidate": None}])
def test_damaged_summary_is_not_a_server_error(result, broken):
    other = deepcopy(result)
    other["metrics"] = broken
    value = compare(result, other)
    assert "right:metric_summary_mismatch" in value.reasons
    assert value.delta is None


@pytest.mark.parametrize("key", ["ridge_0_1", "ridge_1", "ridge_10", "hgb_delta"])
def test_candidate_version_does_not_inherit_aggregate_hgb_label(key):
    historical = {"model_version": "hist-gradient-boosting-v0.1", "frozen_spec": {
        "spec_version": "experiment-v3-fixed-q1", "candidate_key": key}}
    assert selected_model_version(historical, key) == key + "-v1"
    assert selected_model_version(historical, "persistence") == "persistence-v1"
    assert selected_model_version(historical, "hist_gradient_boosting") == historical["model_version"]
    assert selected_model_version(historical, "unknown") is None
    explicit = {**historical, "model_versions": {key: "explicit-revision"}}
    assert selected_model_version(explicit, key) == "explicit-revision"
