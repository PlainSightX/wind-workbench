"""DeepSeek 对照：复用服务的真实调用方式（同 messages、json_object、temperature=0、max_tokens=1700）。

输出格式与 gen_vllm.py 相同，便于 score.py 统一评分。key 只在内存中读取，不写入任何输出。
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "src/power_forecast_service").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from power_forecast_service.assistant.configuration import provider_key  # noqa: E402
from score import PANELS, load  # noqa: E402


def call(client, model, row, extra):
    messages = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    for attempt in range(4):
        t = time.time()
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=1700,
                                               response_format={"type": "json_object"}, timeout=120, **extra)
            u = r.usage
            return {"id": row["id"], "text": r.choices[0].message.content or "", "output_tokens": u.completion_tokens,
                    "prompt_tokens": u.prompt_tokens,
                    "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None),
                    "finish": r.choices[0].finish_reason, "seconds": time.time() - t, "model": r.model}
        except Exception as exc:  # 429/超时：退避重试，记录最后错误
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(2 ** attempt * 3)
    return {"id": row["id"], "text": "", "output_tokens": 0, "prompt_tokens": 0, "finish": "error", "error": err,
            "seconds": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--panels", nargs="+", default=["test300", "dev32"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--thinking", choices=["default", "disabled"], default="default")
    args = ap.parse_args()
    client = OpenAI(api_key=provider_key(), base_url="https://api.deepseek.com")
    extra = {"extra_body": {"thinking": {"type": "disabled"}}} if args.thinking == "disabled" else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for panel in args.panels:
        rows = load(PANELS[panel])[: args.limit]
        t0 = time.time()
        with cf.ThreadPoolExecutor(args.workers) as pool:
            records = list(pool.map(lambda r: call(client, args.model, r, extra), rows))
        wall = time.time() - t0
        (out / f"{panel}.json").write_text(json.dumps({"wall_seconds": wall, "records": records}, ensure_ascii=False),
                                           encoding="utf-8")
        errs = sum(r["finish"] == "error" for r in records)
        print(panel, len(records), "errors", errs, "wall", round(wall, 1),
              "models", sorted({r.get("model") for r in records if r.get("model")}),
              "in", sum(r["prompt_tokens"] for r in records), "cached", sum(r.get("cached_tokens") or 0 for r in records),
              "out", sum(r["output_tokens"] for r in records), flush=True)


if __name__ == "__main__":
    main()
