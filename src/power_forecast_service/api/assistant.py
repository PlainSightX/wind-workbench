"""助手独立端点；浏览器不能指定SQL、检索文件、provider或无限调用预算。"""

import asyncio
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Request

from ..assistant.contracts import Question
from ..assistant.retrieval import corpus
from ..assistant.workflow import TIMEOUT_SECONDS

router = APIRouter(prefix="/assistant")


@router.post("/answers")
async def answer(request: Request, body: Question, x_request_id: UUID | None = Header(default=None)):
    assistant = request.app.state.assistant
    try:
        await asyncio.wait_for(assistant.gate.acquire(), timeout=0.01)
    except TimeoutError:
        raise HTTPException(429, "assistant_busy") from None
    request_id = x_request_id or uuid4()
    if assistant.journal:
        # 请求表的ORM会加载向量类型；留到生命周期/实际请求，保持API定义导入轻量。
        from ..assistant.requests import DuplicateRequest
        try:
            await assistant.journal.reserve(request_id, body, seconds=TIMEOUT_SECONDS)
        except DuplicateRequest as exc:
            assistant.gate.release()
            raise HTTPException(409, {"code": "request_identity_conflict" if exc.conflict
                else "request_already_registered", **exc.snapshot}) from None
        except BaseException:
            assistant.gate.release()
            raise
    async def execute():
        try:
            result = await assistant.run(body, **({"request_id": str(request_id)} if assistant.journal else {}))
            if assistant.journal and not await assistant.journal.finish(request_id, result):
                result.update(status="dependency_error", error="provider_result_unknown",
                    answer="请求结果尚未确认，不会自动重发。", facts=[], citations=[])
            return result
        except BaseException:
            if assistant.journal:
                # 留住发送前身份；写库失败也不能转换成自动重发或假成功。
                await assistant.journal.finish(request_id)
            raise
        finally:
            assistant.gate.release()
    # 断开客户端不释放仍在工作的容量槽；统一超时由工作流负责。
    task = asyncio.create_task(execute())
    request.app.state.assistant_tasks.add(task)
    def finished(task):
        request.app.state.assistant_tasks.discard(task)
        if not task.cancelled():
            task.exception()
    task.add_done_callback(finished)
    return await asyncio.shield(task)


@router.get("/requests/{request_id}")
async def request_status(request_id: UUID, request: Request):
    journal = request.app.state.assistant.journal
    result = await journal.lookup(request_id) if journal else None
    if result is None:
        raise HTTPException(404, "assistant_request_not_found")
    return result


@router.get("/documents/{document_id}")
async def document(document_id: str):
    for chunk in corpus()["chunks"]:
        if chunk["id"] == document_id:
            # 仅白名单快照，不能用URL传入文件路径。
            return {k:chunk[k] for k in ("id", "title", "source_sha256", "text_sha256", "text")}
    raise HTTPException(404, "document_not_found")
