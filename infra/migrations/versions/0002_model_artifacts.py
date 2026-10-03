"""新增完整模型登记；历史结果文件保持原语义，不伪造历史可用模型。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_model_artifacts"
down_revision = "0001_experiments"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_artifacts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("experiment_runs.id"), nullable=False),
        sa.Column("model_key", sa.String(80), nullable=False),
        sa.Column("path", sa.String(240), nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("manifest", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "model_key", name="artifact_run_model"),
        sa.CheckConstraint("status IN ('ready','unavailable')", name="artifact_status"),
    )


def downgrade():
    op.drop_table("model_artifacts")
