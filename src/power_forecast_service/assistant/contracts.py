"""助手的输入、事实和回答合同；对象范围由HTTP请求固定，模型不能扩大。"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MAX_ANSWER_FACTS = 15
MAX_ANSWER_CITATIONS = 8


class ContextRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["q1_run", "engie_import"]
    id: UUID
    model: str | None = Field(default=None, max_length=80)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=1200)
    contexts: list[ContextRef] = Field(min_length=1, max_length=2)


class Fact(BaseModel):
    id: str
    label: str
    value: float | str | bool
    unit: str = ""
    object_id: str
    artifact_id: str | None = None
    aggregation: str
    source_sha256: str
    pointer: str
    derived: bool = False
    stage: str = "selected_result"
    source_path: str | None = None


class StageClaim(BaseModel):
    """把正文的连续片段绑定到决策阶段；不允许仅在附件声称已回答。"""

    model_config = ConfigDict(extra="forbid")
    object_id: UUID
    role: Literal["selection_basis", "development_result", "adoption_decision", "final_holdout_result"]
    text: str = Field(min_length=1, max_length=1500)
    citations: list[str] = Field(default_factory=list, max_length=4)


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(default="", max_length=3500)
    fact_ids: list[str] = Field(default_factory=list, max_length=MAX_ANSWER_FACTS)
    citations: list[str] = Field(default_factory=list, max_length=MAX_ANSWER_CITATIONS)
    quotes: dict[str, str] = Field(default_factory=dict, max_length=MAX_ANSWER_CITATIONS)
    stage_claims: list[StageClaim] = Field(default_factory=list, max_length=8)


class CitationSelection(BaseModel):
    """模型只选本次请求内的原文块，版本由可信文档映射持有。"""

    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1, max_length=160)


class PlainReferenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["plain"]
    text: str = Field(min_length=1, max_length=3500)
    citations: list[CitationSelection] = Field(default_factory=list, max_length=MAX_ANSWER_CITATIONS)


class ReferenceStageClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object_id: UUID
    role: Literal["selection_basis", "development_result", "adoption_decision", "final_holdout_result"]
    text: str = Field(min_length=1, max_length=1500)
    citations: list[CitationSelection] = Field(default_factory=list, max_length=4)


class StagedReferenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["staged"]
    claims: list[ReferenceStageClaim] = Field(min_length=1, max_length=8)


class ReferenceAnswer(BaseModel):
    """可选生成合同；正文只能选一种表示，事实列表和原文由程序解析。"""

    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence"]
    body: Annotated[PlainReferenceBody | StagedReferenceBody, Field(discriminator="kind")]


class AssistantError(Exception):
    """公开稳定错误码，不携带provider响应、凭据或数据库连接文本。"""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(code)
