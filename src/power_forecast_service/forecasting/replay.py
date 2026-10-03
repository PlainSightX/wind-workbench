"""只回放原模型已登记的评分时刻；开发/冻结正式留出分别守住时间边界。"""

from collections import deque
import csv
from datetime import datetime
import hashlib
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .contracts import ForecastRequest
from .scoring import ScoringEvidence
from ..storage.model_packages import PackageError, checked_manifest, confined_path


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: UUID
    cutoff: datetime

    @field_validator("cutoff")
    @classmethod
    def source_time(cls, value):
        if value.tzinfo is not None or value.second or value.microsecond or value.minute % 5:
            raise ValueError("cutoff_must_use_naive_five_minute_source_time")
        return value


def replay_context(root, registration, run):
    """用原运行的时间边界与身份，不拿当前默认切分重新解释旧模型。"""
    manifest = checked_manifest(root, registration)
    if (str(manifest.run_id) != str(run["run_id"])
            or str(manifest.task_id) != str(run["task_id"])
            or str(manifest.attempt_id) != str(run["attempt_id"])):
        raise PackageError("replay_identity_mismatch")
    result = run["result"]
    try:
        scores = ScoringEvidence.model_validate(result["scoring"])
        split = result["split"]
        test_start = datetime.fromisoformat(split["test_start"])
        final = result.get("purpose") == "final_evaluation"
        if (test_start.tzinfo is not None or manifest.training_label_end.tzinfo is not None
                or result["purpose"] not in ("development", "final_evaluation")
                or result["evaluation_split"] != ("test" if final else "validation")
                or scores.evaluation_split != result["evaluation_split"]
                or split["test_scored"] is not final
                or scores.horizon_minutes != manifest.horizon_minutes
                or result["horizon_minutes"] != scores.horizon_minutes
                or result["split_version"] != scores.split_version
                or manifest.frozen_spec["dataset_id"] != "wind-2019-q1"
                or manifest.model_key not in scores.rows[0].predictions):
            raise PackageError("replay_evidence_invalid")
        identities = [scores.input_sha256, result["input_file_sha256"],
                      manifest.frozen_spec["input_sha256"],
                      result["frozen_spec"]["input_sha256"]]
        if len(set(identities)) != 1:
            raise PackageError("replay_input_identity_mismatch")
        if final and (not manifest.frozen_spec.get("final_protocol_id")
                      or manifest.frozen_spec.get("purpose") != "final_evaluation"
                      or result.get("final_selection") != manifest.frozen_spec.get("final_selection")
                      or any(row.cutoff < test_start for row in scores.rows)):
            raise PackageError("replay_final_protocol_invalid")
        if not final and any(row.target_time >= test_start for row in scores.rows):
            raise PackageError("replay_test_boundary_violation")
        rows = [row for row in scores.rows if row.cutoff > manifest.training_label_end]
        if not rows:
            raise PackageError("replay_no_eligible_windows")
        return manifest, rows
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PackageError):
            raise
        raise PackageError("replay_evidence_invalid") from exc


def replay_windows(root, registration, run):
    manifest, rows = replay_context(root, registration, run)
    return {"artifact_id": registration.artifact_id,
            "mode": "final_historical_replay" if run["result"]["purpose"] == "final_evaluation" else "development_historical_replay",
            "training_label_end": manifest.training_label_end,
            "cutoffs": [row.cutoff for row in rows], "count": len(rows),
            "clock": manifest.clock, "horizon_minutes": manifest.horizon_minutes}


def read_history(snapshot: Path, expected_sha256: str, artifact_id: UUID, cutoff: datetime, *, count=13):
    """单次打开：解析至cutoff便停，剩余字节仅用于完整哈希，不解析封存标签。"""
    digest = hashlib.sha256()
    observations = deque(maxlen=count)
    with snapshot.open("rb") as stream:
        def source_lines():
            for line in stream:
                digest.update(line)
                yield line.decode("utf-8-sig")

        try:
            for row in csv.DictReader(source_lines()):
                timestamp = datetime.strptime(row["Time"].replace("-T", " "), "%Y-%m-%d %H:%M")
                if timestamp > cutoff:
                    break
                observations.append({"timestamp": timestamp,
                                     "wind_power": float(row["Wind_production"]),
                                     "wind_speed": float(row["Wind_speed"]),
                                     "humidity": float(row["Humidity"]),
                                     "temperature": float(row["Temperature"])})
                if timestamp == cutoff:
                    break
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        except (KeyError, ValueError, UnicodeError, csv.Error) as exc:
            raise PackageError("replay_history_invalid") from exc
    if digest.hexdigest() != expected_sha256:
        raise PackageError("replay_snapshot_changed")
    try:
        body = ForecastRequest(artifact_id=artifact_id, observations=list(observations))
        if body.observations[-1].timestamp != cutoff:
            raise PackageError("replay_cutoff_missing")
        return body
    except ValidationError as exc:
        raise PackageError("replay_history_invalid") from exc


def prepare_replay(root, registration, run, cutoff):
    manifest, rows = replay_context(root, registration, run)
    selected = next((row for row in rows if row.cutoff == cutoff), None)
    if selected is None:
        raise PackageError("replay_cutoff_not_allowed")
    # 登记数据的固定文件名及DB关联组成路径；不接受用户路径或result里的绝对路径。
    snapshot = confined_path(root, f"{manifest.task_id}/{manifest.attempt_id}/wind_2019_q1.csv")
    if not snapshot.is_file():
        raise PackageError("replay_snapshot_missing")
    body = read_history(snapshot, manifest.frozen_spec["input_sha256"], registration.artifact_id,
                        cutoff, count=manifest.input_contract["min_observations"])
    return body, selected.actual
