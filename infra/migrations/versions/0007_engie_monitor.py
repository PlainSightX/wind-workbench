"""增加固定模型延迟评分账本，不改已有任务与模型记录。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007_engie_monitor"
down_revision = "0006_assistant"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("engie_monitors",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_key", sa.String(128), nullable=False, unique=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("champion_id", sa.Uuid(), sa.ForeignKey("imported_artifacts.id"), nullable=False),
        sa.Column("shadow_id", sa.Uuid(), sa.ForeignKey("imported_artifacts.id"), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_until", sa.DateTime(timezone=True)),
        sa.Column("contract", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("end_time > start_time", name="engie_monitor_interval"),
        sa.CheckConstraint("champion_id != shadow_id", name="engie_monitor_models"))
    op.create_table("engie_monitor_issues",
        sa.Column("monitor_id", sa.Uuid(), sa.ForeignKey("engie_monitors.id"), primary_key=True),
        sa.Column("issue_time", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("reason", sa.String(80)),
        sa.Column("predictions", postgresql.JSONB(none_as_null=True)),
        sa.CheckConstraint("status IN ('predicted','invalid_input','failed')", name="engie_monitor_issue_status"))
    op.create_table("engie_monitor_observations",
        sa.Column("monitor_id", sa.Uuid(), sa.ForeignKey("engie_monitors.id"), primary_key=True),
        sa.Column("target_time", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("turbines", postgresql.JSONB(), nullable=False))
    op.create_table("engie_monitor_residuals",
        sa.Column("monitor_id", sa.Uuid(), primary_key=True),
        sa.Column("issue_time", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("horizon_minutes", sa.Integer(), primary_key=True),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("champion_error_kw", sa.Float(), nullable=False),
        sa.Column("shadow_error_kw", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["monitor_id", "issue_time"],
                                ["engie_monitor_issues.monitor_id", "engie_monitor_issues.issue_time"]),
        sa.ForeignKeyConstraint(["monitor_id", "target_time"],
                                ["engie_monitor_observations.monitor_id", "engie_monitor_observations.target_time"]),
        sa.CheckConstraint("horizon_minutes IN (10,20,30,40,50,60)", name="engie_monitor_horizon"),
        sa.CheckConstraint("target_time = issue_time + horizon_minutes * INTERVAL '1 minute'",
                           name="engie_monitor_residual_target"))


def downgrade():
    for name in ("engie_monitor_residuals", "engie_monitor_observations", "engie_monitor_issues", "engie_monitors"):
        op.drop_table(name)
