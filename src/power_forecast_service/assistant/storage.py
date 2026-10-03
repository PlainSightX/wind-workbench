"""向量类型仅在助手运行/迁移时导入，普通API定义不加载数值库。"""

from sqlalchemy import String
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
