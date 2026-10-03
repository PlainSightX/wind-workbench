"""真实PG及完整模型的页面接口，不拿模拟模型代替回放。"""

import shutil

from fastapi.testclient import TestClient
import pytest

from power_forecast_service.api.app import create_app
from power_forecast_service.experiments.execution import claim, publish_result
from power_forecast_service.serve import make_event_loop


def test_browse_compare_replay_and_frozen_input(config, engine, submit, publication, monkeypatch):
    task_id = submit(config).task_id
    attempt_id, spec = claim(engine, task_id, config)
    payload, packages = publication(task_id, attempt_id, spec)
    snapshot = config.artifact_root / str(task_id) / str(attempt_id) / "wind_2019_q1.csv"
    shutil.copyfile(config.data_path, snapshot)
    assert publish_result(engine, task_id, attempt_id, payload, "result.json", "a" * 64,
                          packages=packages, artifact_root=config.artifact_root)
    from sklearn.ensemble import HistGradientBoostingRegressor
    def forbidden(*args, **kwargs):
        pytest.fail("Page reads and replay must not fit")
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
    with TestClient(create_app(config), base_url="http://localhost",
                    backend_options={"loop_factory": make_event_loop}) as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        rows = client.get("/tasks").json()
        assert rows[0]["run_id"] == payload["run_id"]
        assert client.get("/tasks?offset=1").json() == []
        runs = client.get("/runs").json()
        assert runs[0]["models"] == payload["model_set"] and runs[0]["has_scoring"]
        assert "scoring" not in runs[0]
        params = {"left_run_id": payload["run_id"], "right_run_id": payload["run_id"],
                  "limit": 2, "offset": 1}
        curve = client.get("/runs/compare-series", params=params).json()
        assert curve["comparison"]["status"] == "comparable" and len(curve["rows"]) == 2
        assert curve["rows"][0]["actual"] == payload["scoring"]["rows"][1]["actual"]
        for package in packages:
            windows = client.get(f"/artifacts/{package.artifact_id}/replay-windows")
            assert windows.status_code == 200, windows.text
            for cutoff in (windows.json()["cutoffs"][0], windows.json()["cutoffs"][-1]):
                response = client.post("/replays", json={"artifact_id": str(package.artifact_id), "cutoff": cutoff})
                assert response.status_code == 200, response.text
                value = response.json()
                stored = next(r for r in payload["scoring"]["rows"] if r["cutoff"] == cutoff)
                assert value["forecast"]["prediction"] == pytest.approx(stored["predictions"][package.model_key])
                assert value["actual"] == stored["actual"]
                assert value["history"][-1]["timestamp"] == cutoff
            rejected = client.post("/replays", json={"artifact_id": str(package.artifact_id),
                                                     "cutoff": package.manifest["training_label_end"]})
            assert rejected.status_code == 409
        snapshot.write_bytes(snapshot.read_bytes() + b"\n")
        response = client.post("/replays", json={"artifact_id": str(package.artifact_id), "cutoff": cutoff})
        assert response.status_code == 409 and response.json()["detail"] == "replay_snapshot_changed"
