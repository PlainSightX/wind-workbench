"""登记经过验证的离线运行，不修改旧训练运行或工件行。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_engie_imports"
down_revision = "0003_final_protocol"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "imported_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("quarter", sa.String(16), nullable=False),
        sa.Column("manifest", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("source_sha256", "quarter", name="import_source_quarter"),
    )
    op.create_table(
        "imported_artifacts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("import_id", sa.Uuid(), sa.ForeignKey("imported_runs.id"), nullable=False),
        sa.Column("family", sa.String(32), nullable=False),
        sa.Column("path", sa.String(240), nullable=False),
        sa.Column("manifest", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.UniqueConstraint("import_id", "family", name="import_family"),
        sa.CheckConstraint("status IN ('ready','unavailable')", name="import_artifact_status"),
    )


def downgrade():
    op.drop_table("imported_artifacts")
    op.drop_table("imported_runs")
