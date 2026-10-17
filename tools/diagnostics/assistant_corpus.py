"""从已交付文档生成只读语料快照；只维护选择规则，不另写一套问答真相。"""

import hashlib
import json
from pathlib import Path
import re

from power_forecast_service.settings import ROOT


def build():
    # 保留源文档版本与原行号；基线中的旧未来计划不进入最终结果的默认检索。
    selections = [
        ("a3", "docs/results/wind-engie-a3-20260923/README.md", ["engie_final"], lambda h: not h.startswith("## 4.")),
        ("a2", "docs/results/wind-engie-a2-20260923/README.md", ["engie_final"], lambda h: h.startswith(("## 问题", "## 采用"))),
        ("engie", "docs/results/wind-engie-baseline-20260923/README.md", ["engie_final", "engie_development"], lambda h: h.startswith(("## 1.", "## 2.", "## 3."))),
        ("q1", "docs/WIND-SEQUENCE-RESULTS.md", ["q1"], lambda h: True),
    ]
    chunks = []
    for name, relative, scopes, wanted in selections:
        path = ROOT / relative
        raw = path.read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("utf-8-sig").splitlines()
        title, heading, group, first = lines[0].lstrip("# "), "", [], 1
        def flush():
            nonlocal group
            body = "\n".join(group).strip()
            if not body:
                return
            sha = hashlib.sha256(body.encode()).hexdigest()
            chunks.append({"id": f"{name}-{sha[:16]}", "title": title + " / " + heading.lstrip("# "),
                "source": relative, "source_sha256": source_sha, "start_line": first,
                "text_sha256": sha, "scopes": scopes, "text":body})
            group = []
        for number, line in enumerate(lines, 1):
            if line.startswith("## "):
                flush()
                heading = line
            if not wanted(heading) or line.startswith("#"):
                continue
            # 源基线的未来保留期是当时计划，不用它否定后来完成的最终评价。
            if name == "engie" and (line.startswith("| 保留期") or (not heading)):
                continue
            if len("\n".join(group)) + len(line) > 420:
                flush()
            if not group:
                first = number
            group.append(line)
        flush()
    serialized = json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode()
    package = {"schema_version":1, "source_commit":"477afa08ca22bcd671ecaf949c4b00b70bd68147",
               "sha256":hashlib.sha256(serialized).hexdigest(), "chunks":chunks}
    path = ROOT / "src/power_forecast_service/assistant/corpus.json"
    path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chunks":len(chunks), "sha256":package["sha256"]}))


if __name__ == "__main__":
    build()
