"""Linux Celery worker；租约心跳与训练分开，过期执行不能发布成功结果。"""

import json
import logging
import os
import shutil
import threading
import time
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from ..experiments.contracts import TaskConflict, validate_spec
from ..experiments.execution import claim, fail_attempt, heartbeat, publish_result
from ..forecasting.pipeline import train_experiment
from ..forecasting.bundles import save_packages, verify_fresh_process
from ..settings import Settings
from ..storage.artifacts import execution_provenance, sha256_file
from ..storage.database import make_sync_engine
from ..storage.model_packages import PackageError
from .celery_app import make_celery

log = logging.getLogger(__name__)
app = make_celery()


def execute_experiment(task_id: UUID, config: Settings):
    # fork 后才创建连接池，避免多个 worker 共享父进程数据库连接。
    engine = make_sync_engine(config)
    stopped = threading.Event()
    thread = None
    attempt_id = None
    try:
        accepted = claim(engine, task_id, config)
        if accepted is None:
            return "duplicate_or_ineligible"
        attempt_id, spec = accepted

        def renew():
            while not stopped.wait(config.heartbeat_seconds):
                try:
                    if not heartbeat(engine, task_id, attempt_id, config):
                        return
                except SQLAlchemyError:
                    log.warning("heartbeat unavailable task=%s attempt=%s", task_id, attempt_id)

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        execution_spec = validate_spec(spec, config)
        directory = config.artifact_root / str(task_id) / str(attempt_id)
        directory.mkdir(parents=True, exist_ok=False)
        # 每次执行使用自己的固定输入副本，原数据后续修改不改变正在训练的字节。
        snapshot = directory / config.data_path.name
        shutil.copyfile(config.data_path, snapshot)
        if sha256_file(snapshot) != spec["input_sha256"]:
            raise TaskConflict("dataset_changed_during_snapshot")
        if execution_spec.get("sequence_key"):
            from ..forecasting.sequence_pipeline import train_sequence_experiment
            product = train_sequence_experiment(snapshot, sequence_key=execution_spec["sequence_key"],
                purpose=execution_spec["purpose"], final_protocol=execution_spec.get("final_selection"))
        else:
            product = train_experiment(
            snapshot, horizon_steps=execution_spec["horizon_steps"],
            training_policy=execution_spec["training_policy"],
            model_parameters=execution_spec["hgb"], random_seed=execution_spec["random_seed"],
            candidate_key=execution_spec.get("candidate_key", "none"),
            )
        result = product.result
        if result["input_file_sha256"] != spec["input_sha256"]:
            raise TaskConflict("dataset_snapshot_changed")
        result["execution"] = execution_provenance()
        # 保留已接受的合同快照；legacy解析不回写任务，也不把旧合同伪装成v2。
        result["frozen_spec"] = spec
        packaging_started = time.perf_counter()
        packages = save_packages(product, config.artifact_root, task_id, attempt_id)
        verification_started = time.perf_counter()
        result["model_verification"] = verify_fresh_process(config.artifact_root, packages)
        result["model_delivery"] = {
            "packaging_seconds": verification_started - packaging_started,
            "independent_verification_seconds": time.perf_counter() - verification_started,
            "package_bytes": sum(path.stat().st_size for item in packages
                                 for path in (config.artifact_root / item.path).rglob("*")
                                 if path.is_file()),
        }
        result["model_artifacts"] = [
            {"artifact_id": str(item.artifact_id), "model_key": item.model_key,
             "manifest_sha256": item.manifest_sha256} for item in packages
        ]
        output = directory / "result.json"
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        relative = output.relative_to(config.artifact_root).as_posix()
        published = publish_result(
            engine, task_id, attempt_id, result, relative, sha256_file(output),
            packages=packages, artifact_root=config.artifact_root,
        )
        if not published:
            log.warning("orphan result: lease lost task=%s attempt=%s", task_id, attempt_id)
        return "succeeded" if published else "stale_attempt"
    except PackageError as exc:
        if attempt_id:
            fail_attempt(engine, task_id, attempt_id, str(exc), config, retryable=False)
        log.warning("model package failure task=%s attempt=%s reason=%s", task_id, attempt_id, exc)
        return "failed"
    except (TaskConflict, ValueError) as exc:
        if attempt_id:
            fail_attempt(
                engine, task_id, attempt_id, "input_or_contract_invalid", config, retryable=False
            )
        log.warning("contract failure task=%s error_type=%s", task_id, type(exc).__name__)
        return "failed"
    except OSError:
        if attempt_id:
            fail_attempt(
                engine, task_id, attempt_id, "artifact_io_unavailable", config, retryable=True
            )
        return "retry_wait"
    except SQLAlchemyError:
        # 无法确认 DB commit 时不假定成功/失败；租约与恢复扫描接管，孤立输出保留。
        log.warning("database outcome unconfirmed task=%s attempt=%s", task_id, attempt_id)
        return "database_outcome_unconfirmed"
    except Exception as exc:  # noqa: BLE001 - 任务边界登记未知算法失败，不让任务永久保持 running。
        if attempt_id:
            fail_attempt(
                engine, task_id, attempt_id, "algorithm_execution_failed", config, retryable=False
            )
        log.error("execution failed task=%s error_type=%s", task_id, type(exc).__name__)
        return "failed"
    finally:
        stopped.set()
        if thread:
            thread.join(timeout=6)
        engine.dispose()


@app.task(name="wind.execute")
def execute(task_id: str):
    # 凭据只在实际执行时读取；导入任务定义不能依赖开发机的密码文件。
    return execute_experiment(UUID(task_id), Settings.from_environment())
