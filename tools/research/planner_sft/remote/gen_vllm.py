"""vLLM 约束解码批量生成：贪心、非思考模式、JSON schema 约束，记录逐题耗时与 token。"""
import argparse
import gzip
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams


SORT = False


def relax(schema, key=None):
    """去掉 xgrammar 不支持的 uniqueItems；评分阶段仍按完整 schema 校验。
    SORT=True 时把 properties 按字母序排列，与训练目标的 sort_keys 序列化一致
    （约束解码按 properties 顺序生成字段；消费者不依赖键顺序）。"""
    if isinstance(schema, dict):
        items = [(k, relax(v, k)) for k, v in schema.items() if k != "uniqueItems"]
        if SORT and key == "properties":
            items.sort()
        return dict(items)
    if isinstance(schema, list):
        return [relax(v) for v in schema]
    return schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--sort-schema", action="store_true")
    args = ap.parse_args()
    global SORT
    SORT = args.sort_schema
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True,
              enable_lora=bool(args.lora), max_lora_rank=64, seed=0)
    tok = llm.get_tokenizer()
    lora = LoRARequest("a", 1, args.lora) if args.lora else None
    for panel in args.panels:
        rows = json.load(gzip.open(panel, "rt", encoding="utf-8"))
        prompts, params = [], []
        for r in rows:
            text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            prompts.append(text)
            params.append(SamplingParams(temperature=0.0, max_tokens=1024,
                                         structured_outputs=StructuredOutputsParams(json=relax(r["schema"]))))
        t0 = time.time()
        res = llm.generate(prompts, params, lora_request=lora)
        wall = time.time() - t0
        records = []
        for r, o in zip(rows, res):
            c = o.outputs[0]
            records.append({"id": r["id"], "text": c.text, "output_tokens": len(c.token_ids),
                            "prompt_tokens": len(o.prompt_token_ids), "finish": c.finish_reason})
        name = Path(panel).name.split(".")[0]
        (out / f"{name}.json").write_text(json.dumps({"wall_seconds": wall, "records": records},
                                                     ensure_ascii=False), encoding="utf-8")
        print(name, len(records), round(wall, 1), flush=True)


if __name__ == "__main__":
    main()
