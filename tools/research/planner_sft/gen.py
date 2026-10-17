"""组合义务数据生成：原子义务 × 表述模板，训练/测试按模板与组合双重切分。

每个原子义务给出一段中文问法与一个 request（gold 规划片段）；问题由 1~3 个原子
按模板拼接。gold 按问题中的出现顺序排列，并用真实消费者自检（gold 必须通过）。
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

# 监督检查器未随公开源码分发；用 WIND_EVAL_HARNESS 指向本地副本。
if "WIND_EVAL_HARNESS" not in os.environ:
    raise SystemExit("set WIND_EVAL_HARNESS to the supervision checker directory")
SRC = Path(os.environ["WIND_EVAL_HARNESS"])
sys.path.insert(0, str(SRC))
import build_supervision as v  # noqa: E402
import supervision_evaluator_v2 as e  # noqa: E402
from build_C import excluded_compositions  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
SEEDS = v.load(v.WIND_SOURCE)
B9 = ["lightgbm", "lightgbm_l1", "lightgbm_l1_shrink", "persistence", "ridge"]
D4 = ["lightgbm", "persistence", "ridge"]
COUNT = {"planned": "计划", "input_valid": "合法输入", "output_count": "实际输出", "scoreable": "可评分"}
GAPS = {  # 标签与 missing 固定，避免同一缺口出现多种监督写法
    "p95": ("实际生产延迟P95", "实际生产延迟P95测量记录", ["实际生产延迟P95", "生产环境实测延迟的P95", "真实到达延迟P95"]),
    "p99": ("实际生产延迟P99", "实际生产延迟P99测量记录", ["实际生产延迟P99", "生产环境实测延迟的P99", "真实到达延迟P99"]),
    "revenue": ("实际售电收入", "实际售电收款凭证", ["实际售电收入", "部署后真实多收的电费", "真实电力交易收入"]),
    "fault": ("确诊传感器故障数量", "确诊传感器故障数量记录", ["确诊由传感器故障造成的拒绝次数", "已确认的传感器硬件故障数量"]),
}
CLOCKS = ["00:05", "00:15", "00:40", "01:30", "02:00", "05:50", "06:00", "08:20", "12:10",
          "15:00", "18:30", "21:10", "23:20", "23:45", "23:50", "23:55"]
UP = {"mae": "MAE", "rmse": "RMSE", "bias": "BIAS"}


def atom(kind, rng, models):
    """返回 (原子id, 需求标签, [问法变体], request项, 是否需要stage文档, 是否需要policy文档)。"""
    m1, m2, m3 = rng.sample(models, 3)
    k = rng.choice(["mae", "rmse"])
    if kind == "read":
        k = rng.choice(["mae", "rmse", "bias"])
        return (kind, [f"{m1}的{UP[k]}原值", f"{m1}的{UP[k]}是多少", f"读取{m1}的{UP[k]}"],
                v.known(f"{UP[k]}原值", v.val(f"c0.{m1}.{k}")), set())
    if kind == "read2":
        return (kind, [f"{m1}的MAE和RMSE原值", f"{m1}的两项误差MAE、RMSE", f"列出{m1}的MAE与RMSE"],
                v.known("两项原值", v.val(f"c0.{m1}.mae", f"c0.{m1}.rmse")), set())
    if kind == "compare":
        return (kind, [f"{m1}与{m2}的{UP[k]}谁更小", f"比较{m1}和{m2}的{UP[k]}大小", f"{m1}的{UP[k]}是否低于{m2}"],
                v.known(f"{UP[k]}关系", v.compare(f"c0.{m1}.{k}", f"c0.{m2}.{k}")), set())
    if kind == "conj":
        return (kind, [f"{m1}的MAE和RMSE是否都严格低于{m2}", f"{m1}相比{m2}是否两项误差同时严格更小"],
                v.known("两项是否同时严格更低", v.compare(f"c0.{m1}.mae", f"c0.{m2}.mae"),
                        v.compare(f"c0.{m1}.rmse", f"c0.{m2}.rmse"), summary=True), set())
    if kind == "sort":
        group = [m1, m2, m3]
        d = rng.choice(["ascending", "descending"])
        word = "从小到大" if d == "ascending" else "从大到小"
        names = "、".join(group)
        return (kind, [f"将{names}按{UP[k]}{word}排序", f"把{names}的{UP[k]}{word}排列", f"{names}按{UP[k]}{word}的完整排序"],
                v.known(f"{UP[k]}排序", v.sort([f"c0.{x}.{k}" for x in group], d)), set())
    if kind == "gain":
        m = rng.choice([x for x in models if x != "persistence"])
        s = rng.choice(["gain", "delta"])
        word = "改善率" if s == "gain" else "差值"
        return (kind, [f"{m}相对persistence的{UP[k]}{word}", f"{m}比持久性基线的{UP[k]}{word}"],
                v.known(f"{UP[k]}{word}", v.val(f"c0.{m}.{k}_{s}")), set())
    if kind == "bias_vanish":
        key = f"c0.{m1}.bias"
        return (kind, [f"{m1}的BIAS是否已消失", f"根据BIAS原值判断{m1}的偏差是否消失"],
                v.known("偏差是否消失", v.val(key), bias=[key]), set())
    if kind == "coverage":
        pair = rng.sample(list(COUNT), 2)
        pair = sorted(pair, key=list(COUNT).index)
        names = "、".join(COUNT[x] for x in pair)
        return (kind, [f"{names}数量及集合定义", f"{names}两个次数及其统计含义", f"核对{names}的数量与定义"],
                v.known("指定覆盖", v.coverage(*pair)), set())
    if kind == "period":
        return (kind, ["所选对象的评价期间", "这批结果属于哪个评价季度"],
                v.known("评价期间", {"kind": "period", "context": "c0"}), set())
    if kind == "time":
        c = rng.choice(CLOCKS)
        return (kind, [f"{c}起报的输入截止和全部目标时刻", f"{c}起报时最晚可用输入与六个目标时刻", f"假设{c}起报，列输入截止及全部预测目标"],
                v.known("起报情景", {"kind": "time", "context": "c0"}), {"policy"})
    if kind == "policy":
        return (kind, ["当前模拟到达延迟规则及输入可用边界", "本次实验如何模拟到达延迟、是否为实测", "模拟延迟如何限制可用输入"],
                v.known("模拟到达规则", {"kind": "delay_policy", "context": "c0"}), {"policy"})
    if kind == "dev":
        return (kind, ["开发阶段共享收缩的MAE改善率", "作为采用依据的开发MAE改善率"],
                v.known("开发改善", v.stage("development_result", "c0.development.lightgbm_l1_shrink.mae_gain")), {"stage"})
    if kind == "adopt":
        return (kind, ["采用登记和开发MAE门槛", "实际采用状态及事前开发门槛"],
                v.known("采用登记与门槛", v.stage("adoption_decision", "c0.adopted", "c0.development.mae_gain_gate")), {"stage"})
    if kind == "boundary":
        return (kind, ["正式留出核验与事前采用决定的边界", "正式核验能否改写事前采用"],
                v.known("正式边界", v.stage("final_holdout_result")), {"stage"})
    if kind in GAPS:
        label, missing, phr = GAPS[kind]
        return (kind, list(phr), v.gap(label, missing), set())
    raise ValueError(kind)


KINDS = ["read", "read2", "compare", "conj", "sort", "gain", "bias_vanish", "coverage", "period",
         "time", "policy", "dev", "adopt", "boundary", "p95", "p99", "revenue", "fault"]
# 互斥：同一问题内不重复同类时间/阶段原子，避免gold内部冲突
EXCLUSIVE = [{"time"}, {"dev", "adopt"}, {"p95", "p99"}]

JOIN_TRAIN = [
    ("按顺序分别回答：先{0}；再{1}。只回答这两项，不添其他。", True),
    ("请依次回答{0}，然后{1}，不附加其他指标。", True),
    ("只回答{0}和{1}。", False),
    ("我想知道{0}，另外{1}。不要额外内容。", False),
    ("分别给出：一、{0}；二、{1}。", True),
]
JOIN3_TRAIN = [
    ("按顺序回答三项：先{0}；再{1}；最后{2}。不添其他。", True),
    ("只分别回答{0}、{1}和{2}。", False),
]
JOIN1_TRAIN = [("只回答：{0}。不附其他内容。", False), ("问题：{0}。", False)]
JOIN_TEST = [
    ("第一，{0}；第二，{1}。其余不要。", True),
    ("帮我查一下{0}，顺带说明{1}，别加无关指标。", False),
]
JOIN3_TEST = [("依次回答：{0}；{1}；{2}。其余不要。", True)]
JOIN1_TEST = [("{0}？只要这一项。", False)]


def choose_seed(needs, models_needed):
    order = [67, 68, 23] if "policy" in needs else [68, 23, 67] if "stage" in needs else [23, 67, 68, 0]
    return order


def build_row(rng, n, joins, phrase_pick, rid, partition):
    for _ in range(50):
        kinds = rng.sample(KINDS, n)
        if any(len(set(kinds) & ex) > 1 for ex in EXCLUSIVE):
            continue
        atoms = [atom(k, rng, B9) for k in kinds]
        needs = set().union(*(a[3] for a in atoms))
        if needs == {"policy", "stage"} and any(k in ("dev", "adopt") for k in kinds):
            continue  # 开发阶段文档与延迟文档不在同一来源
        template, ordered = rng.choice(joins)
        phrases = [phrase_pick(rng, a[1]) for a in atoms]
        question = template.format(*phrases)
        gold = v.plan(*[deepcopy(a[2]) for a in atoms])
        if len(gold["requests"]) != len({json.dumps(r, ensure_ascii=False, sort_keys=True) for r in gold["requests"]}):
            continue
        for si in choose_seed(needs, None):
            seed = SEEDS[si]
            try:
                row = v.make_row(seed, "gen", rid, 1, question, gold, partition=partition,
                                 composition="+".join(kinds), ordered=ordered)
            except Exception:
                continue
            res = e.evaluate_plan(row, row["gold"])
            if res["automatic_screen_passed"] and not res.get("error"):
                row["kinds"] = kinds
                return row
    return None


def combo_key(row):
    return "+".join(sorted(row["kinds"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=900)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)
    # 测试专用组合：随机保留约15%的两原子组合不进训练
    pairs = sorted({"+".join(sorted(p)) for p in
                    [(a, b) for i, a in enumerate(KINDS) for b in KINDS[i + 1:]]})
    held = set(random.Random(99).sample(pairs, len(pairs) * 15 // 100))
    train_pick = lambda r, ph: r.choice(ph[:-1]) if len(ph) > 2 else r.choice(ph)  # noqa: E731
    test_pick = lambda r, ph: ph[-1] if len(ph) > 2 else r.choice(ph)  # noqa: E731
    train, test, seen = [], [], set()
    stats = Counter()
    i = 0
    while len(train) < args.train:
        i += 1
        n = rng.choices([1, 2, 3], weights=[2, 5, 3])[0]
        joins = {1: JOIN1_TRAIN, 2: JOIN_TRAIN, 3: JOIN3_TRAIN}[n]
        row = build_row(rng, n, joins, train_pick, f"tr-{i:05d}", "train")
        if row is None:
            stats["fail"] += 1; continue
        ks = row["kinds"]
        sub = {"+".join(sorted(p)) for p in [(a, b) for x, a in enumerate(ks) for b in ks[x + 1:]]}
        if sub & held or excluded_compositions(row["gold"]):
            stats["skip_held"] += 1; continue
        q = row["payload"]["question"]
        if q in seen: continue
        seen.add(q); train.append(row)
    j = 0
    while len(test) < args.test:
        j += 1
        n = rng.choices([1, 2, 3], weights=[1, 5, 4])[0]
        joins = {1: JOIN1_TEST, 2: JOIN_TEST, 3: JOIN3_TEST}[n]
        row = build_row(rng, n, joins, test_pick, f"te-{j:05d}", "test")
        if row is None: continue
        q = row["payload"]["question"]
        if q in seen: continue
        ks = row["kinds"]
        sub = {"+".join(sorted(p)) for p in [(a, b) for x, a in enumerate(ks) for b in ks[x + 1:]]}
        row["test_unseen_combo"] = bool(sub & held)
        seen.add(q); test.append(row)
    for name, rows in [("train", train), ("test", test)]:
        (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"train": len(train), "test": len(test), "stats": stats,
                      "test_unseen_combo": sum(r["test_unseen_combo"] for r in test),
                      "n_atoms_train": Counter(len(r["kinds"]) for r in train),
                      "kinds_train": Counter(k for r in train for k in r["kinds"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
