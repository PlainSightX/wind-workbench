"""导出远端所需的精简数据：训练用 messages+completion；评测用 messages+schema。"""
import gzip
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 监督检查器未随公开源码分发；用 WIND_EVAL_HARNESS 指向本地副本。
if "WIND_EVAL_HARNESS" not in os.environ:
    raise SystemExit("set WIND_EVAL_HARNESS to the supervision checker directory")
SRC = Path(os.environ["WIND_EVAL_HARNESS"])


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
