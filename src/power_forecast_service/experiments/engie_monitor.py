"""先预测、后到达、再计分；PG提交位置负责恢复，而非进程内累计指标。"""

from datetime import timedelta
import math
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from ..forecasting.engie_monitor_contract import CONTRACT, LABEL_DELAY, STEP, finish_time
from ..forecasting.engie_service_contract import HORIZONS, ROSTER
from ..storage.model_packages import PackageError
from ..storage.models import (EngieMonitor, EngieMonitorIssue, EngieMonitorObservation,
                              EngieMonitorResidual, ImportedArtifact, ImportedRun)
from .engie_delivery import digest, validate_result


def model_identity(item):
    return {"artifact_id": str(item.id), "import_id": str(item.import_id), "family": item.family,
            "model_version": item.manifest["model_version"], "sha256": item.manifest["sha256"]}


def registered(session, identity):
    item = session.get(ImportedArtifact, identity)
    if item is None or item.status != "ready":
        raise PackageError("engie_monitor_artifact_not_ready")
    run = session.get(ImportedRun, item.import_id)
    return item, {"id": item.id, "import_id": run.id, "family": item.family,
                  "path": item.path, "manifest": item.manifest, "run": run.manifest}


def create_monitor(engine, body):
    from ..forecasting.engie_predictor import check_issue

    fingerprint = digest({"version": CONTRACT, **body.model_dump(mode="json", exclude={"request_key"})})
    with Session(engine) as session, session.begin():
        existing = session.scalar(select(EngieMonitor).where(EngieMonitor.request_key == body.request_key))
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise PackageError("engie_monitor_request_conflict")
            # 已完成记录仍能读回，即使模型后来停用；新的推进另查当前工件状态。
            return existing.id
        champion, c = registered(session, body.champion_id)
        shadow, s = registered(session, body.shadow_id)
        if champion.import_id != shadow.import_id or champion.family != "persistence":
            raise PackageError("engie_monitor_same_source_persistence_required")
        for item in (c, s):
            check_issue(item, body.start)
            check_issue(item, body.end - STEP)
        contract = {"version": CONTRACT, "arrival": "simulated_fixed_20_minutes",
                    "unit": "kW", "clock": "UTC", "window_issues": body.window_issues,
                    "alert": {"minimum_pairs": 12, "shadow_relative_mae_margin": 0.10,
                              "meaning": "review_only_no_automatic_adoption"},
                    "champion": model_identity(champion), "shadow": model_identity(shadow),
                    "source_sha256": c["run"]["source_sha256"],
                    "history_sha256": c["run"]["history_sha256"],
                    "predictions_sha256": c["run"]["predictions_sha256"],
                    "scope": c["run"].get("scope", "development")}
        session.execute(insert(EngieMonitor).values(
            id=uuid4(), request_key=body.request_key, fingerprint=fingerprint,
            champion_id=body.champion_id, shadow_id=body.shadow_id,
            start_time=body.start, end_time=body.end, contract=contract,
        ).on_conflict_do_nothing(index_elements=["request_key"]))
        row = session.scalar(select(EngieMonitor).where(EngieMonitor.request_key == body.request_key))
        if row.fingerprint != fingerprint or row.contract != contract:
            raise PackageError("engie_monitor_request_conflict")
        return row.id


def monitor_row(session, identity, *, lock=False):
    query = select(EngieMonitor).where(EngieMonitor.id == identity)
    row = session.scalar(query.with_for_update() if lock else query)
    if row is None:
        raise PackageError("engie_monitor_not_found")
    return row


def models_for(session, row):
    models = {}
    for role, identity in (("champion", row.champion_id), ("shadow", row.shadow_id)):
        item, registration = registered(session, identity)
        if (model_identity(item) != row.contract[role]
                or any(registration["run"][key] != row.contract[key]
                       for key in ("source_sha256", "history_sha256", "predictions_sha256"))):
            raise PackageError("engie_monitor_model_changed")
        models[role] = (item, registration)
    return models


def store_predictions(session, row, issue, source, root, models):
    from ..forecasting.engie_predictor import forecast

    results, status, reason = {}, "predicted", None
    try:
        body = source.inputs(issue, row.champion_id)
    except PackageError as exc:
        if str(exc) != "engie_history_unavailable":
            raise
        status, reason = "invalid_input", str(exc)
    else:
        for role, (artifact, registration) in models.items():
            try:
                request = body.model_copy(update={"artifact_id": artifact.id})
                result = forecast(root, registration, request)
                expected = SimpleNamespace(artifact_id=artifact.id, issue_time=issue,
                                           input_sha256=digest(result_history(body)))
                validate_result(expected, result, artifact)
                results[role] = result
            except PackageError as exc:
                status, reason = "failed", str(exc)
    session.add(EngieMonitorIssue(monitor_id=row.id, issue_time=issue, status=status,
                                 reason=reason, predictions=results or None))
    # 标签读取前实际写入本事务；后续故障使预测、评分和时钟一起回滚。
    session.flush()


def result_history(body):
    return [item.model_dump(mode="json") for item in body.history]


def store_label(session, row, target, available, turbines, clock):
    if (available < target + LABEL_DELAY or available > clock
            or target < row.start_time + STEP or target > row.end_time - STEP + timedelta(minutes=60)):
        raise PackageError("engie_monitor_label_not_arrived_or_outside")
    if set(turbines) != set(ROSTER) or any(
            value is not None and (isinstance(value, bool) or not math.isfinite(value))
            for value in turbines.values()):
        raise PackageError("engie_monitor_invalid_truth")
    observation = session.get(EngieMonitorObservation, (row.id, target))
    if observation is None:
        observation = EngieMonitorObservation(monitor_id=row.id, target_time=target,
                                              available_at=available, turbines=turbines)
        session.add(observation)
    else:
        merged = dict(observation.turbines)
        changed = False
        for name, value in turbines.items():
            if merged[name] is not None and value is not None and merged[name] != value:
                raise PackageError("engie_monitor_truth_conflict")
            if merged[name] is None and value is not None:
                merged[name], changed = value, True
        if changed:
            observation.turbines = merged
            observation.available_at = max(available, observation.available_at)
    session.flush()
    if any(value is None for value in observation.turbines.values()):
        return
    actual = sum(observation.turbines.values())
    if not math.isfinite(actual):
        raise PackageError("engie_monitor_invalid_truth")
    # 一个目标关联最多六个起报；乱序或迟到的实况仍用同一复合主键补齐。
    for index, horizon in enumerate(HORIZONS):
        issue_time = target - timedelta(minutes=horizon)
        item = session.get(EngieMonitorIssue, (row.id, issue_time))
        if item is None or item.status != "predicted":
            continue
        errors = {role: item.predictions[role]["farm_predictions"][index] - actual
                  for role in ("champion", "shadow")}
        if not all(math.isfinite(value) for value in errors.values()):
            raise PackageError("engie_monitor_invalid_residual")
        session.execute(insert(EngieMonitorResidual).values(
            monitor_id=row.id, issue_time=issue_time, horizon_minutes=horizon,
            target_time=target, champion_error_kw=errors["champion"], shadow_error_kw=errors["shadow"],
        ).on_conflict_do_nothing(index_elements=["monitor_id", "issue_time", "horizon_minutes"]))


def advance_monitor(engine, root, identity, body):
    from ..forecasting.engie_predictor import MonitorReplaySource

    with Session(engine) as session:
        row = monitor_row(session, identity)
        if body.through < row.start_time:
            raise PackageError("engie_monitor_clock_outside")
        models = models_for(session, row)
        source = MonitorReplaySource(root, models["champion"][1])
    steps = 0
    for _ in range(body.max_steps):
        with Session(engine) as session, session.begin():
            row = monitor_row(session, identity, lock=True)
            event_time = row.start_time if row.processed_until is None else row.processed_until + STEP
            if event_time > body.through:
                break
            # 并发推进在锁内重读游标；一个步骤要么全部提交，要么全部回滚。
            models = models_for(session, row)
            if event_time < row.end_time:
                store_predictions(session, row, event_time, source, root, models)
            target = event_time - LABEL_DELAY
            if row.start_time + STEP <= target <= row.end_time - STEP + timedelta(minutes=60):
                truth = source.observation(target, clock=event_time)
                store_label(session, row, target, event_time, truth, event_time)
            row.processed_until = event_time
        steps += 1
    return {"steps_committed": steps, **read_monitor(engine, identity)}


def ingest_label(engine, identity, body):
    with Session(engine) as session, session.begin():
        row = monitor_row(session, identity, lock=True)
        if row.processed_until is None:
            raise PackageError("engie_monitor_clock_not_started")
        store_label(session, row, body.target_time, body.available_at, body.turbines, row.processed_until)
    return read_monitor(engine, identity)


def summarize(issues, residuals, *, planned, clock, window, margin, minimum):
    """滚动窗口按已计划起报位置截取，不用最后N条成功样本掩盖缺测。"""
    counts = {"planned_issues": planned, "attempted_issues": len(issues),
              "valid_inputs": sum(item.status != "invalid_input" for item in issues),
              "invalid_inputs": sum(item.status == "invalid_input" for item in issues),
              "failed_issues": sum(item.status == "failed" for item in issues),
              "outputs": {role: sum(role in (item.predictions or {}) for item in issues)
                          for role in ("champion", "shadow")}}
    latest = issues[-window:]
    included = {item.issue_time for item in latest}
    rolling_invalid = sum(item.status == "invalid_input" for item in latest)
    rolling_failed = sum(item.status == "failed" for item in latest)
    rolling_predicted = sum(item.status == "predicted" for item in latest)
    points = {(item.issue_time, item.horizon_minutes): item for item in residuals}
    horizons = []
    for horizon in HORIZONS:
        pairs = [item for item in residuals if item.horizon_minutes == horizon]
        rolling = [item for item in pairs if item.issue_time in included]
        due = [item for item in issues if item.status == "predicted"
               and item.issue_time + timedelta(minutes=horizon) + LABEL_DELAY <= clock]
        rolling_due = [item for item in due if item.issue_time in included]
        rolling_pending = rolling_predicted - len(rolling_due)
        rolling_missing = sum((item.issue_time, horizon) not in points for item in rolling_due)
        # 完整性描述样本支持范围，不替代原来的相对误差判断或模型采用门。
        window_status = ("not_started" if not latest else
                         "incomplete" if rolling_invalid or rolling_failed or rolling_missing else
                         "awaiting_labels" if rolling_pending else "complete")
        metrics = {}
        for role in ("champion", "shadow"):
            errors = [getattr(item, f"{role}_error_kw") for item in rolling]
            metrics[role] = {"mae_kw": math.fsum(abs(e) / len(errors) for e in errors) if errors else None,
                             "bias_kw": math.fsum(e / len(errors) for e in errors) if errors else None}
        c, s = metrics["champion"]["mae_kw"], metrics["shadow"]["mae_kw"]
        alert = ("insufficient_pairs" if len(rolling) < minimum else
                 "shadow_worse_review" if s > c * (1 + margin) else "within_margin")
        horizons.append({"horizon_minutes": horizon, "scored_pairs": len(pairs),
                         "pending_labels": sum(item.status == "predicted" for item in issues) - len(due),
                         "due_missing_labels": sum((item.issue_time, horizon) not in points for item in due),
                         "rolling_scheduled_issues": len(latest), "rolling_pairs": len(rolling),
                         "rolling_pending_labels": rolling_pending,
                         "rolling_due_missing_labels": rolling_missing,
                         "rolling_invalid_inputs": rolling_invalid, "rolling_failed_issues": rolling_failed,
                         "window_status": window_status,
                         **metrics, "shadow_minus_champion_mae_kw": s - c if c is not None else None,
                         "alert": alert})
    return {"counts": counts, "horizons": horizons}


def read_monitor(engine, identity):
    with Session(engine) as session, session.begin():
        # 与写路径互斥，避免在READ COMMITTED的多条查询中拼出不同时点的报告。
        row = monitor_row(session, identity, lock=True)
        issues = session.scalars(select(EngieMonitorIssue).where(
            EngieMonitorIssue.monitor_id == identity).order_by(EngieMonitorIssue.issue_time)).all()
        residuals = session.scalars(select(EngieMonitorResidual).where(
            EngieMonitorResidual.monitor_id == identity)).all()
        contract = row.contract
        report = summarize(issues, residuals, planned=int((row.end_time - row.start_time) / STEP),
                           clock=row.processed_until or row.start_time, window=contract["window_issues"],
                           margin=contract["alert"]["shadow_relative_mae_margin"],
                           minimum=contract["alert"]["minimum_pairs"])
        return {"id": str(row.id), "request_key": row.request_key,
                "start": row.start_time.isoformat(), "end": row.end_time.isoformat(),
                "processed_until": row.processed_until.isoformat() if row.processed_until else None,
                "finish_time": finish_time(row.end_time).isoformat(),
                "replay_complete": row.processed_until is not None and row.processed_until >= finish_time(row.end_time),
                "contract": contract, **report}
