"""任务、执行尝试、投递通知和唯一有效结果。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_experiments"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "experiment_tasks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False, unique=True),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        sa.Column("source_task_id", sa.Uuid(), sa.ForeignKey("experiment_tasks.id")),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("active_attempt_id", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(80)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "status IN ('pending_dispatch','queued','running','retry_wait','succeeded','failed')",
            name="task_status",
        ),
        sa.CheckConstraint("attempt_count BETWEEN 0 AND 3", name="task_attempt_limit"),
        sa.CheckConstraint(
            "(status = 'running') = (lease_until IS NOT NULL)", name="task_running_lease"
        ),
        sa.CheckConstraint(
            "status != 'running' OR active_attempt_id IS NOT NULL", name="task_running_attempt"
        ),
    )
    op.create_index("ix_experiment_tasks_status", "experiment_tasks", ["status"])
    op.create_table(
        "experiment_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("experiment_tasks.id"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_code", sa.String(80)),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("task_id", "number", name="attempt_number"),
        sa.CheckConstraint(
            "status IN ('running','succeeded','failed','expired')", name="attempt_status"
        ),
    )
    op.create_index("ix_experiment_attempts_task_id", "experiment_attempts", ["task_id"])
    op.create_foreign_key(
        "fk_active_attempt",
        "experiment_tasks",
        "experiment_attempts",
        ["active_attempt_id"],
        ["id"],
    )
    op.create_table(
        "experiment_outbox",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "task_id", sa.Uuid(), sa.ForeignKey("experiment_tasks.id"), nullable=False, unique=True
        ),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("publish_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.String(80)),
    )
    op.create_index("ix_experiment_outbox_available_at", "experiment_outbox", ["available_at"])
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "task_id", sa.Uuid(), sa.ForeignKey("experiment_tasks.id"), nullable=False, unique=True
        ),
        sa.Column(
            "attempt_id",
            sa.Uuid(),
            sa.ForeignKey("experiment_attempts.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column("artifact_path", sa.String(240), nullable=False),
        sa.Column("artifact_sha256", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade():
    op.drop_table("experiment_runs")
    op.drop_table("experiment_outbox")
    op.drop_constraint("fk_active_attempt", "experiment_tasks", type_="foreignkey")
    op.drop_table("experiment_attempts")
    op.drop_table("experiment_tasks")
