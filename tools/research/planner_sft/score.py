"""本地评分：复用原 supervision_evaluator_v2 的严格自动判定（schema、真实消费者、义务、顺序、范围）。

pass = automatic_screen_passed（无 failures）。pending 项（分组/措辞差异）不算失败——
在此前的逐题复核中，自动通过但有 pending 的题绝大多数被判通过。
另报 strict（无 pending）作为下界。所有比例附 Wilson 95% 区间与配对 McNemar 检验。
"""
import argparse
import json
import os
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 监督检查器未随公开源码分发；用 WIND_EVAL_HARNESS 指向本地副本。
if "WIND_EVAL_HARNESS" not in os.environ:
    raise SystemExit("set WIND_EVAL_HARNESS to the supervision checker directory")
SRC = Path(os.environ["WIND_EVAL_HARNESS"])
sys.path.insert(0, str(SRC))
import supervision_evaluator_v2 as e  # noqa: E402

PANELS = {"test300": HERE / "data/test.json", "dev32": SRC / "composition_dev32-v3.json",
          "current40": SRC / "current40_rows.json"}


def load(p):
    d = json.loads(Path(p).read_bytes())
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def mcnemar(a, b):
    """a,b: 同题配对布尔列表；返回(仅a对, 仅b对, 精确双侧p)。"""
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    if n == 0:
        return x, y, 1.0
    k = min(x, y)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return x, y, min(1.0, p)


def fine_atoms(operations):
    """语义原子序列：values按事实、coverage按计数拆开，与分组方式无关。"""
    seq = []
    for op in operations:
        op = e.v1.normalize_operation(op)
        if op["kind"] == "values":
            seq += [("values", k) for k in op["fact_ids"]]
        elif op["kind"] == "coverage" and op.get("counts"):
            seq += [("coverage", k) for k in op["counts"]]
        else:
            seq.append(e.canonical(op))
    return seq


def plan_sequence(items, requirement=False):
    seq = []
    for item in items:
        if requirement:
            seq += [("gap",)] if "missing" in item else fine_atoms(item["operations"])
        elif item["resolution"]["kind"] == "unavailable":
            seq.append(("gap",))
        else:
            seq += fine_atoms(item["resolution"]["operations"])
    return seq


def lenient(row, plan, res):
    """去掉只由分组造成的两类失败：义务集合与顺序改为按语义原子序列比较。"""
    fails = res.get("plan", {}).get("failures", []) + res.get("delivery", {}).get("failures", [])
    expected = row.get("plan_requirements", e.v1.requirements_from_plan(row["gold"]))
    want, got = plan_sequence(expected, True), plan_sequence(plan["requests"])
    from collections import Counter
    same_set = Counter(map(str, want)) == Counter(map(str, got))
    def first_order(seq):
        seen = []
        for x in map(str, seq):
            if x not in seen:
                seen.append(x)
        return seen
    same_order = first_order(want) == first_order(got)
    keep = []
    for f in fails:
        if f["kind"] == "operation_obligations_differ" and same_set:
            continue
        if f["kind"] == "explicit_request_order_mismatch" and same_order and same_set:
            continue
        keep.append(f["kind"])
    return not keep


def score_panel(panel, out_dir):
    rows = {r["id"]: r for r in load(PANELS[panel])}
    f = Path(out_dir) / f"{panel}.json"
    if not f.exists():
        return None
    gen = json.loads(f.read_bytes())
    results = {}
    for rec in gen["records"]:
        row = rows[rec["id"]]
        try:
            plan = json.loads(rec["text"])
            res = e.evaluate_plan(row, plan)
            ok = bool(res["automatic_screen_passed"] and not res.get("error"))
            strict = bool(res["passed"])
            soft = bool(not res.get("error") and res.get("consumer_accepted") and lenient(row, plan, res))
            why = [x["kind"] for x in res.get("plan", {}).get("failures", []) + res.get("delivery", {}).get("failures", [])]
            if res.get("error"):
                why.append("consumer_rejected:" + str(res["error"]["code"]))
        except Exception as exc:  # 非法 JSON 等
            ok, strict, soft, why = False, False, False, ["invalid:" + type(exc).__name__]
        results[rec["id"]] = {"pass": ok, "strict": strict, "soft": soft, "why": why, "out_tokens": rec["output_tokens"]}
    return {"wall": gen["wall_seconds"], "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=dir")
    ap.add_argument("--panels", nargs="+", default=["test300", "dev32"])
    ap.add_argument("--save")
    args = ap.parse_args()
    runs = dict(x.split("=", 1) for x in args.runs)
    summary = {}
    for panel in args.panels:
        test_rows = {r["id"]: r for r in load(PANELS[panel])}
        scored = {name: score_panel(panel, d) for name, d in runs.items()}
        scored = {k: v for k, v in scored.items() if v}
        print(f"\n## {panel}")
        ids = sorted(test_rows)
        for name, s in scored.items():
            r = s["results"]
            k = sum(r[i]["pass"] for i in ids if i in r); n = len(r)
            ks = sum(r[i]["strict"] for i in ids if i in r)
            kf = sum(r[i]["soft"] for i in ids if i in r)
            line = f"{name:12s} pass {k}/{n} = {100*k/n:.1f}% CI{wilson(k,n)}  soft {kf}/{n} CI{wilson(kf,n)}  strict {ks}/{n}  wall {s['wall']:.0f}s"
            if panel == "test300":
                for flag in (False, True):
                    sub = [i for i in ids if i in r and test_rows[i].get("test_unseen_combo") == flag]
                    kk = sum(r[i]["pass"] for i in sub)
                    line += f"  | {'unseen' if flag else 'seen'}-combo {kk}/{len(sub)}"
                for na in (1, 2, 3):
                    sub = [i for i in ids if i in r and len(test_rows[i]["kinds"]) == na]
                    kk = sum(r[i]["pass"] for i in sub)
                    line += f"  n{na}:{kk}/{len(sub)}"
            print(line)
            summary.setdefault(panel, {})[name] = {"pass": k, "soft": kf, "n": n, "strict": ks, "wall": s["wall"]}
        names = list(scored)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ra, rb = scored[names[a]]["results"], scored[names[b]]["results"]
                common = [i for i in ids if i in ra and i in rb]
                x, y, p = mcnemar([ra[i]["pass"] for i in common], [rb[i]["pass"] for i in common])
                x2, y2, p2 = mcnemar([ra[i]["soft"] for i in common], [rb[i]["soft"] for i in common])
                print(f"  {names[a]} vs {names[b]}: pass only-a {x} only-b {y} p={p:.4f} | soft only-a {x2} only-b {y2} p={p2:.4f}")
        for name, s in scored.items():
            from collections import Counter
            c = Counter(w.split(":")[0] for v in s["results"].values() for w in v["why"])
            print(f"  {name} failure kinds: {dict(c.most_common(8))}")
    if args.save:
        Path(args.save).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
