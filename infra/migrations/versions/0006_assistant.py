"""添加版本文档向量索引和最小回答审计，不更新训练或预测对象。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0006_assistant"
down_revision = "0005_engie_deliveries"
branch_labels = None
depends_on = None


def upgrade():
    # vector由管理员在建库时安装；应用迁移不索要superuser权限。
    op.create_table("assistant_chunks",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("corpus_sha256", sa.String(64), nullable=False),
        sa.Column("text_sha256", sa.String(64), nullable=False),
        sa.Column("embedding_revision", sa.String(40), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False))
    op.create_index("ix_assistant_chunks_corpus_sha256", "assistant_chunks", ["corpus_sha256"])
    op.create_table("answer_audits",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("question_sha256", sa.String(64), nullable=False),
        sa.Column("answer_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("trace", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()))


def downgrade():
    op.drop_table("answer_audits")
    op.drop_table("assistant_chunks")
