"""助手独立端点；浏览器不能指定SQL、检索文件、provider或无限调用预算。"""

import asyncio

from fastapi import APIRouter, HTTPException, Request

from ..assistant.contracts import Question
from ..assistant.retrieval import corpus

router = APIRouter(prefix="/assistant")


@router.post("/answers")
async def answer(request: Request, body: Question):
    assistant = request.app.state.assistant
    try:
        await asyncio.wait_for(assistant.gate.acquire(), timeout=0.01)
    except TimeoutError:
        raise HTTPException(429, "assistant_busy") from None
    async def execute():
        try:
            return await assistant.run(body)
        finally:
            assistant.gate.release()
    # 断开客户端不释放仍在工作的容量槽；统一超时由工作流负责。
    task = asyncio.create_task(execute())
    request.app.state.assistant_tasks.add(task)
    task.add_done_callback(request.app.state.assistant_tasks.discard)
    return await asyncio.shield(task)


@router.get("/documents/{document_id}")
async def document(document_id: str):
    for chunk in corpus()["chunks"]:
        if chunk["id"] == document_id:
            # 仅白名单快照，不能用URL传入文件路径。
            return {k:chunk[k] for k in ("id", "title", "source_sha256", "text_sha256", "text")}
    raise HTTPException(404, "document_not_found")
