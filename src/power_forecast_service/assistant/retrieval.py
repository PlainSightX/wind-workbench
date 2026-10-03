"""固定中文embedding与精确pgvector；关键词对照消费相同版本和相同语料。"""

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import BoundedSemaphore

from sqlalchemy import select

from ..settings import ROOT
from .contracts import AssistantError

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
CORPUS_PATH = Path(__file__).with_name("corpus.json")


class ReadExecutor(ThreadPoolExecutor):
    """取消await不能终止本地计算；占用真实线程直到完成，期间拒绝排队。"""

    def __init__(self):
        super().__init__(max_workers=1, thread_name_prefix="assistant-read")
        self.slot = BoundedSemaphore(1)

    def submit(self, fn, /, *args, **kwargs):
        if not self.slot.acquire(blocking=False):
            raise AssistantError("assistant_busy")
        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            self.slot.release()
            raise
        future.add_done_callback(lambda _: self.slot.release())
        return future


def runtime_root():
    return Path(os.getenv("WIND_ASSISTANT_RUNTIME", str(ROOT / ".local/runtime/assistant")))


def corpus():
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if hashlib.sha256(json.dumps(payload["chunks"], ensure_ascii=False, sort_keys=True).encode()).hexdigest() != payload["sha256"]:
        raise AssistantError("corpus_integrity_error")
    for chunk in payload["chunks"]:
        if hashlib.sha256(chunk["text"].encode()).hexdigest() != chunk["text_sha256"]:
            raise AssistantError("corpus_integrity_error")
    return payload


@lru_cache(maxsize=1)
def embedding_model():
    from transformers import AutoModel, AutoTokenizer
    import torch

    root = runtime_root()
    try:
        manifest = json.loads((root / "embedding-manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != EMBEDDING_MODEL or manifest["revision"] != EMBEDDING_REVISION:
            raise ValueError("embedding identity mismatch")
        for name, sha in manifest["files"].items():
            if Path(name).name != name or hashlib.sha256((root / "embedding" / name).read_bytes()).hexdigest() != sha:
                raise ValueError("embedding byte mismatch")
        torch.set_num_threads(2)
        return (AutoTokenizer.from_pretrained(root / "embedding", local_files_only=True),
                AutoModel.from_pretrained(root / "embedding", local_files_only=True).to("cpu").eval())
    except (OSError, ValueError, KeyError) as exc:
        raise AssistantError("embedding_unavailable") from exc


def embed(texts, *, query=False):
    import torch

    tokenizer, model = embedding_model()
    if query:
        texts = ["为这个句子生成表示以用于检索相关文章：" + t for t in texts]
    vectors = []
    for start in range(0, len(texts), 8):
        inputs = tokenizer(texts[start:start + 8], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.inference_mode():
            encoded = model(**inputs).last_hidden_state[:, 0]
            vectors.extend(torch.nn.functional.normalize(encoded, p=2, dim=1).tolist())
    return vectors


def terms(value):
    words = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", value.lower())
    return [part for word in words for part in ([word] if word.isascii() else [word[i:i+2] for i in range(max(1, len(word)-1))])]


def keyword_rank(query, chunks, limit=4):
    """简单词项/中文双字匹配；IDF降低常见词影响，不伪称语义检索。"""
    q = set(terms(query))
    docs = [(c, set(terms(c["title"] + " " + c["text"]))) for c in chunks]
    def score(item):
        return sum(math.log(1 + len(docs)/(1 + sum(term in d[1] for d in docs))) for term in q & item[1])
    ranked = sorted(docs, key=lambda x: (-score(x), x[0]["id"]))
    return [{**c, "score": score((c, t))} for c, t in ranked[:limit] if score((c, t)) > 0]


def expand_sections(ranked, allowed):
    """小段向量负责命中，完整同节提供上下文，避免把表头/时间边界切掉。"""
    output, seen = [], set()
    # 同版本概览用于解释阶段，不让旧开发叙述覆盖最终评价。
    anchors = [next((c for c in allowed if c["id"].startswith(prefix)), None) for prefix in ("a3-", "q1-")]
    for chunk in [c for c in anchors if c] + [c for hit in ranked for c in allowed if c["title"] == hit["title"]]:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            output.append(chunk)
    return output[:16]


async def retrieve(sessions, scopes, query, strategy, executor):
    import asyncio
    from .storage import AssistantChunk

    pack = corpus()
    allowed = [c for c in pack["chunks"] if set(c["scopes"]) & set(scopes)]
    if not allowed:
        return []
    if strategy == "keyword":
        return expand_sections(keyword_rank(query, allowed), allowed)
    if strategy != "vector":
        raise AssistantError("retrieval_strategy_invalid")
    loop = asyncio.get_running_loop()
    vector = (await loop.run_in_executor(executor, lambda: embed([query], query=True)))[0]
    async with sessions() as session:
        from sqlalchemy import text
        await session.execute(text("SET TRANSACTION READ ONLY"))
        distance = AssistantChunk.embedding.cosine_distance(vector)
        rows = (await session.execute(select(AssistantChunk, distance.label("distance")).where(
            AssistantChunk.corpus_sha256 == pack["sha256"],
            AssistantChunk.embedding_revision == EMBEDDING_REVISION,
            AssistantChunk.id.in_([c["id"] for c in allowed]),
        ).order_by(distance, AssistantChunk.id).limit(4))).all()
    if len(rows) != min(4, len(allowed)):
        raise AssistantError("document_index_unavailable")
    by_id = {c["id"]: c for c in allowed}
    results = []
    for row, distance in rows:
        original = by_id[row.id]
        if row.text_sha256 != original["text_sha256"] or row.content != original["text"]:
            raise AssistantError("document_index_integrity_error")
        results.append({**original, "score": 1 - distance})
    return expand_sections(results, allowed)
