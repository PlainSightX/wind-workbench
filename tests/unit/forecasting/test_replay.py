"""回放的时间/输入边界；不拟合模型，不解析cutoff之后的功率。"""

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from power_forecast_service.api.workbench import comparison_series
from power_forecast_service.forecasting import replay
from power_forecast_service.forecasting.scoring import ScoreRow, metrics_for_rows, sample_fingerprint
from power_forecast_service.storage.artifacts import sha256_file
from power_forecast_service.storage.model_packages import PackageError


@pytest.fixture
def case(tmp_path, monkeypatch):
    start = datetime(2020, 1, 1)
    task, attempt, run_id, artifact_id = (uuid4() for _ in range(4))
    path = tmp_path / str(task) / str(attempt) / "wind_2019_q1.csv"
    path.parent.mkdir(parents=True)
    lines = ["Time,Wind_production,Wind_speed,Humidity,Temperature"]
    lines.extend(f"{start + timedelta(minutes=i * 5):%Y-%m-%d %H:%M},{10+i},2,50,15" for i in range(43))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = sha256_file(path)
    rows = [ScoreRow(cutoff=start + timedelta(minutes=i * 5),
                     target_time=start + timedelta(minutes=(i + 12) * 5), actual=22 + i,
                     predictions={"persistence": 10 + i, "hist_gradient_boosting": 11 + i})
            for i in (13, 14, 23)]
    scores = {"version": "scoring-v1", "input_sha256": digest, "target": "wind_power_single_point",
              "unit": "source_reported_unit", "clock": "source_time_timezone_unknown",
              "evaluation_split": "validation", "horizon_minutes": 60, "split_version": "test",
              "metric_version": "unweighted-mae-rmse-v1", "samples_sha256": sample_fingerprint(rows),
              "rows": [row.model_dump(mode="json") for row in rows]}
    result = {"scoring": scores, "purpose": "development", "evaluation_split": "validation",
              "split": {"test_start": "2020-01-01T03:00:00", "test_scored": False},
              "horizon_minutes": 60, "horizon_steps": 12, "split_version": "test",
              "input_file_sha256": digest, "frozen_spec": {"input_sha256": digest},
              "metrics": {key: metrics_for_rows(rows, key) for key in rows[0].predictions}}
    manifest = SimpleNamespace(task_id=task, attempt_id=attempt, run_id=run_id,
                               training_label_end=start + timedelta(hours=1),
                               frozen_spec={"input_sha256": digest, "dataset_id": "wind-2019-q1"},
                               model_key="persistence", horizon_minutes=60,
                               input_contract={"min_observations": 13})
    monkeypatch.setattr(replay, "checked_manifest", lambda *args: manifest)
    return tmp_path, SimpleNamespace(artifact_id=artifact_id), {
        "run_id": run_id, "task_id": task, "attempt_id": attempt, "result": result,
    }, manifest, path


@pytest.mark.parametrize("cutoff", ["01:05", "01:55"])
def test_first_last_window_excludes_future_target(case, cutoff):
    root, registration, run, manifest, _ = case
    body, actual = replay.prepare_replay(root, registration, run, datetime.fromisoformat(f"2020-01-01T{cutoff}"))
    assert len(body.observations) == 13
    assert body.observations[-1].timestamp > manifest.training_label_end
    assert body.observations[-1].wind_power != actual
    assert set(body.model_dump()) == {"artifact_id", "observations"}


@pytest.mark.parametrize("cutoff", ["01:00", "01:15", "02:00", "03:00"])
def test_not_registered_or_train_or_test_cutoff_rejected(case, cutoff):
    root, registration, run, _, _ = case
    with pytest.raises(PackageError, match="replay_cutoff_not_allowed"):
        replay.prepare_replay(root, registration, run, datetime.fromisoformat(f"2020-01-01T{cutoff}"))


@pytest.mark.parametrize("mutation,reason", [
    ("identity", "replay_identity_mismatch"), ("input", "replay_input_identity_mismatch"),
    ("test", "replay_test_boundary_violation"), ("missing", "replay_evidence_invalid"),
    ("snapshot", "replay_snapshot_changed"),
])
def test_independent_gates(case, mutation, reason):
    root, registration, run, _, path = case
    if mutation == "identity":
        run["run_id"] = uuid4()
    elif mutation == "input":
        run["result"]["input_file_sha256"] = "a" * 64
    elif mutation == "test":
        run["result"]["split"]["test_start"] = "2020-01-01T02:55:00"
    elif mutation == "missing":
        del run["result"]["scoring"]
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(PackageError, match=reason):
        replay.prepare_replay(root, registration, run, datetime(2020, 1, 1, 1, 5))


def test_parser_stops_at_cutoff_but_hash_covers_remaining_bytes(case, monkeypatch):
    root, registration, run, _, _ = case
    original = replay.csv.DictReader
    def guarded(*args, **kwargs):
        for index, row in enumerate(original(*args, **kwargs)):
            assert index <= 13, "Parsed a row beyond requested cutoff"
            yield row
    monkeypatch.setattr(replay.csv, "DictReader", guarded)
    replay.prepare_replay(root, registration, run, datetime(2020, 1, 1, 1, 5))


def test_cutoff_timezone_rejected():
    with pytest.raises(ValidationError):
        replay.ReplayRequest(artifact_id=uuid4(), cutoff="2020-01-01T01:05:00Z")


def test_formal_replay_needs_frozen_protocol_and_keeps_target_out_of_input(case):
    root, registration, run, manifest, _ = case
    result = run["result"]
    result.update(purpose="final_evaluation", evaluation_split="test", final_selection={"fixture": True})
    result["scoring"]["evaluation_split"] = "test"
    result["split"].update(test_start="2020-01-01T01:05:00", test_scored=True)
    with pytest.raises(PackageError, match="final_protocol"):
        replay.prepare_replay(root, registration, run, datetime(2020, 1, 1, 1, 55))
    manifest.frozen_spec.update(final_protocol_id="a" * 64, purpose="final_evaluation", final_selection={"fixture": True})
    manifest.input_contract["min_observations"] = 24
    body, actual = replay.prepare_replay(root, registration, run, datetime(2020, 1, 1, 1, 55))
    assert len(body.observations) == 24
    assert body.observations[-1].wind_power != actual


def test_curve_page_matches_stored_values_and_refuses_invalid_evidence(case):
    _, _, run, _, _ = case
    result = run["result"]
    page = comparison_series(run["run_id"], result, "persistence", run["run_id"], result,
                             "hist_gradient_boosting", 1, 1)
    assert page["comparison"].status == "comparable"
    assert page["total"] == 3 and len(page["rows"]) == 1
    row = result["scoring"]["rows"][1]
    assert page["rows"][0]["actual"] == row["actual"]
    assert page["rows"][0]["right_prediction"] == row["predictions"]["hist_gradient_boosting"]
    bad = deepcopy(result)
    bad["scoring"]["rows"][0]["actual"] += 1
    rejected = comparison_series(run["run_id"], result, "persistence", run["run_id"], bad,
                                 "hist_gradient_boosting", 1, 0)
    assert rejected["comparison"].status == "not_comparable"
    assert rejected["rows"] == [] and rejected["comparison"].delta is None
