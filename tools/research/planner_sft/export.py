"""导出远端所需的精简数据：训练用 messages+completion；评测用 messages+schema。"""
import gzip
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 评测/监督工具链（原 SFT 工作包）；公开仓库未包含，可用 WIND_EVAL_HARNESS 指向本地副本。
SRC = Path(os.environ.get("WIND_EVAL_HARNESS", Path(__file__).resolve().parent.parent / "inference-i1/sft-task-redesign-v1"))


def load(p):
    d = json.loads(Path(p).read_bytes())
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def slim(rows, train=False):
    out = []
    for r in rows:
        x = {"id": r["id"], "messages": r["messages"]}
        if train:
            x["completion"] = r["completion"]
        else:
            x["schema"] = r["schema"]
        out.append(x)
    return out


def main():
    up = HERE / "upload"
    up.mkdir(exist_ok=True)
    panels = {
        "train": slim(load(HERE / "data/train.json"), train=True),
        "test300": slim(load(HERE / "data/test.json")),
        "dev32": slim(load(SRC / "composition_dev32-v3.json")),
        "current40": slim(load(SRC / "current40_rows.json")),
    }
    for name, rows in panels.items():
        with gzip.open(up / f"{name}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        print(name, len(rows))


if __name__ == "__main__":
    main()
