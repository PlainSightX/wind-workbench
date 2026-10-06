"""ORM 映射与数据库约束；这里不创建连接，也不决定 HTTP 响应。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Float,
    Integer,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "experiment_tasks"
    __table_args__ = (
        Index("uq_final_evaluation_protocol", text("(spec ->> 'final_protocol_id')"), unique=True,
              postgresql_where=text("spec ->> 'purpose' = 'final_evaluation'")),
        CheckConstraint(
            "status IN ('pending_dispatch','queued','running','retry_wait','succeeded','failed')",
            name="task_status",
        ),
        CheckConstraint("attempt_count BETWEEN 0 AND 3", name="task_attempt_limit"),
        CheckConstraint(
            "(status = 'running') = (lease_until IS NOT NULL)", name="task_running_lease"
        ),
        CheckConstraint(
            "status != 'running' OR active_attempt_id IS NOT NULL", name="task_running_attempt"
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    spec: Mapped[dict] = mapped_column(JSONB)
    source_task_id: Mapped[UUID | None] = mapped_column(ForeignKey("experiment_tasks.id"))
    status: Mapped[str] = mapped_column(String(24), index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    # 与 attempt 表形成环；由迁移在两表建立后补外键。
    active_attempt_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("experiment_attempts.id", use_alter=True, name="fk_active_attempt")
    )
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Attempt(Base):
    __tablename__ = "experiment_attempts"
    __table_args__ = (
        UniqueConstraint("task_id", "number", name="attempt_number"),
        CheckConstraint(
            "status IN ('running','succeeded','failed','expired')", name="attempt_status"
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    task_id: Mapped[UUID] = mapped_column(ForeignKey("experiment_tasks.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    error_code: Mapped[str | None] = mapped_column(String(80))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Outbox(Base):
    __tablename__ = "experiment_outbox"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    task_id: Mapped[UUID] = mapped_column(ForeignKey("experiment_tasks.id"), unique=True)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    publish_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String(80))


class Run(Base):
    __tablename__ = "experiment_runs"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    task_id: Mapped[UUID] = mapped_column(ForeignKey("experiment_tasks.id"), unique=True)
    attempt_id: Mapped[UUID] = mapped_column(ForeignKey("experiment_attempts.id"), unique=True)
    result: Mapped[dict] = mapped_column(JSONB)
    artifact_path: Mapped[str] = mapped_column(String(240))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelArtifact(Base):
    """数据库决定工件是否可选；目录存在和manifest自称ready都不构成发布。"""

    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "model_key", name="artifact_run_model"),
        CheckConstraint("status IN ('ready','unavailable')", name="artifact_status"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("experiment_runs.id"))
    model_key: Mapped[str] = mapped_column(String(80))
    path: Mapped[str] = mapped_column(String(240))
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ImportedRun(Base):
    """离线来源的导入身份，不伪造由队列执行的 Task/Attempt。"""

    __tablename__ = "imported_runs"
    __table_args__ = (UniqueConstraint("source_sha256", "quarter", name="import_source_quarter"),)
    id: Mapped[UUID] = mapped_column(primary_key=True)
    source_sha256: Mapped[str] = mapped_column(String(64))
    quarter: Mapped[str] = mapped_column(String(16))
    manifest: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ImportedArtifact(Base):
    __tablename__ = "imported_artifacts"
    __table_args__ = (
        UniqueConstraint("import_id", "family", name="import_family"),
        CheckConstraint("status IN ('ready','unavailable')", name="import_artifact_status"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    import_id: Mapped[UUID] = mapped_column(ForeignKey("imported_runs.id"))
    family: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(String(240))
    manifest: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20))


class EngieDelivery(Base):
    """一次逻辑预测的不可续期发布记录，与训练任务/模型登记各自独立。"""

    __tablename__ = "engie_deliveries"
    __table_args__ = (
        CheckConstraint("status IN ('pending','published','expired','failed')", name="engie_delivery_status"),
        CheckConstraint("budget_ms BETWEEN 1 AND 120000", name="engie_delivery_budget"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    request_key: Mapped[str] = mapped_column(String(128), unique=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("imported_artifacts.id"))
    issue_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    input_sha256: Mapped[str] = mapped_column(String(64))
    budget_ms: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_sha256: Mapped[str | None] = mapped_column(String(64))
    result: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    reason: Mapped[str | None] = mapped_column(String(80))


class AnswerAudit(Base):
    """只留问题/回答hash、来源身份与消耗，不默认持久化私人聊天正文。"""

    __tablename__ = "answer_audits"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    question_sha256: Mapped[str] = mapped_column(String(64))
    answer_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    trace: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EngieMonitor(Base):
    """一次固定模型比较的身份与已提交模拟时钟；不改变发布模型。"""

    __tablename__ = "engie_monitors"
    __table_args__ = (
        CheckConstraint("end_time > start_time", name="engie_monitor_interval"),
        CheckConstraint("champion_id != shadow_id", name="engie_monitor_models"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    request_key: Mapped[str] = mapped_column(String(128), unique=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    champion_id: Mapped[UUID] = mapped_column(ForeignKey("imported_artifacts.id"))
    shadow_id: Mapped[UUID] = mapped_column(ForeignKey("imported_artifacts.id"))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    processed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    contract: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EngieMonitorIssue(Base):
    __tablename__ = "engie_monitor_issues"
    __table_args__ = (
        CheckConstraint("status IN ('predicted','invalid_input','failed')", name="engie_monitor_issue_status"),
    )
    monitor_id: Mapped[UUID] = mapped_column(ForeignKey("engie_monitors.id"), primary_key=True)
    issue_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(80))
    predictions: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))


class EngieMonitorObservation(Base):
    """四机组实况可以缺失；已收到的有限值不可被后续重投改写。"""

    __tablename__ = "engie_monitor_observations"
    monitor_id: Mapped[UUID] = mapped_column(ForeignKey("engie_monitors.id"), primary_key=True)
    target_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    turbines: Mapped[dict] = mapped_column(JSONB)


class EngieMonitorResidual(Base):
    """一个起报/时距只计分一次；两个模型始终使用同一完整全场标签。"""

    __tablename__ = "engie_monitor_residuals"
    __table_args__ = (
        ForeignKeyConstraint(["monitor_id", "issue_time"],
                             ["engie_monitor_issues.monitor_id", "engie_monitor_issues.issue_time"]),
        ForeignKeyConstraint(["monitor_id", "target_time"],
                             ["engie_monitor_observations.monitor_id", "engie_monitor_observations.target_time"]),
        CheckConstraint("horizon_minutes IN (10,20,30,40,50,60)", name="engie_monitor_horizon"),
        CheckConstraint("target_time = issue_time + horizon_minutes * INTERVAL '1 minute'",
                        name="engie_monitor_residual_target"),
    )
    monitor_id: Mapped[UUID] = mapped_column(primary_key=True)
    issue_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    champion_error_kw: Mapped[float] = mapped_column(Float)
    shadow_error_kw: Mapped[float] = mapped_column(Float)
