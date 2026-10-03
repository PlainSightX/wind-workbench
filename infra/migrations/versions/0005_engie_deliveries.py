"""持久化预测发布身份和截止时间，保留旧训练/离线导入行。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_engie_deliveries"
down_revision = "0004_engie_imports"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "engie_deliveries",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("request_key", sa.String(128), nullable=False, unique=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), sa.ForeignKey("imported_artifacts.id"), nullable=False),
        sa.Column("issue_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("input_sha256", sa.String(64), nullable=False),
        sa.Column("budget_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("finalized_at", sa.DateTime(timezone=True)),
        sa.Column("result_sha256", sa.String(64)),
        sa.Column("result", postgresql.JSONB(none_as_null=True)),
        sa.Column("reason", sa.String(80)),
        sa.CheckConstraint("status IN ('pending','published','expired','failed')", name="engie_delivery_status"),
        sa.CheckConstraint("budget_ms BETWEEN 1 AND 120000", name="engie_delivery_budget"),
    )


def downgrade():
    op.drop_table("engie_deliveries")
