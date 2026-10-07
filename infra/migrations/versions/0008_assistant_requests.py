"""可选推理服务的调用前登记；不修改预测、训练或原回答审计。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008_assistant_requests"
down_revision = "0007_engie_monitor"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("assistant_requests",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("calls", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finalized_at", sa.DateTime(timezone=True)),
        sa.Column("outcome", sa.String(32)),
        sa.Column("error", sa.String(80)),
        sa.Column("answer_sha256", sa.String(64)),
        sa.CheckConstraint("status IN ('pending','completed','unknown')", name="assistant_request_status"))


def downgrade():
    op.drop_table("assistant_requests")
