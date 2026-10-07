"""向量类型仅在助手运行/迁移时导入，普通API定义不加载数值库。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import String, CheckConstraint, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from ..storage.models import Base


class AssistantChunk(Base):
    __tablename__ = "assistant_chunks"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    corpus_sha256: Mapped[str] = mapped_column(String(64), index=True)
    text_sha256: Mapped[str] = mapped_column(String(64))
    embedding_revision: Mapped[str] = mapped_column(String(40))
    content: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(512))


class AssistantRequest(Base):
    """调用前登记身份；仅存 hash/调用状态，不存私人问题和回答正文。"""

    __tablename__ = "assistant_requests"
    __table_args__ = (CheckConstraint("status IN ('pending','completed','unknown')",
                                     name="assistant_request_status"),)
    id: Mapped[UUID] = mapped_column(primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    calls: Mapped[list] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(String(80))
    answer_sha256: Mapped[str | None] = mapped_column(String(64))
