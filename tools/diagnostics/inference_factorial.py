"""合格候选的配对性能对照；复用冻结回答入口，不改变模型或业务验收。"""

import argparse
import asyncio
from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
from statistics import median
import time


HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("factorial_optimization", HERE.with_name("inference_optimization.py"))
optimization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimization)
baseline = optimization.baseline

# 三轮调换条件先后；同轮四格使用完全相同的题目顺序。
ORDERS = (("A", "B", "D", "C"), ("C", "D", "B", "A"), ("B", "A", "C", "D"))


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def schedule(case_ids):
    if len(case_ids) < 2 or len(case_ids) != len(set(case_ids)):
        raise ValueError("Unique different questions are required")
    return [{"batch_index": round_index * 4 + index, "round": round_index + 1, "label": label,
             "case_order": case_ids[round_index:] + case_ids[:round_index]}
            for round_index, order in enumerate(ORDERS) for index, label in enumerate(order)]


def result_name(batch, phase="measurement"):
    return f"{batch['batch_index']:02d}-r{batch['round']}-{batch['label']}-{phase}.jsonl"


def make_protocol(workload_path, manifest_path, qualification_path, output):
    workload, manifest = baseline.load_workload(workload_path), baseline.load_workload(manifest_path)
    if manifest["workload_sha256"] != workload["sha256"]:
        raise ValueError("Workload and manifest differ")
    identity = optimization.qualification_identity(workload, manifest, "json_schema", "xgrammar")
    qualification = optimization.load_qualification(qualification_path, identity)
    for name, expected in manifest["source_files_sha256"].items():
        if baseline.file_digest(optimization.ROOT / name) != expected:
            raise ValueError("Frozen source differs: " + name)
    value = {"schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "qualification_identity": identity, "qualification_sha256": qualification["sha256"],
        "engine_identity": qualification["engine_identity"], "runner_sha256": baseline.file_digest(HERE),
        "schedule": schedule([item["id"] for item in workload["cases"]]),
        "cells": optimization.CELLS, "warmup": "All original-layout questions before each batch; retain separately; then reset APC",
        "cache_reset": "One reset per different-question batch; verify first actual call hits zero, including APC-off",
        "quality": "Original machine gates plus independent final-answer review for every measured task; keep all failures",
        "adoption": "All measured tasks accepted; report paired round effects and actual hits; no benefit claim from warmup, repair/output changes or same-question reuse alone",
        "scope": "Sequential same-host loopback actual Assistant.answer; not HTTP/PG concurrency or production throughput"}
    value["sha256"] = baseline.digest(value)
    save_new(output, value)
    return value


def check_protocol(protocol, workload, manifest, qualification):
    identity = optimization.qualification_identity(workload, manifest, "json_schema", "xgrammar")
    if (manifest["workload_sha256"] != workload["sha256"]
            or protocol["qualification_identity"] != identity
            or protocol["qualification_sha256"] != qualification["sha256"]
            or protocol["engine_identity"] != qualification["engine_identity"]
            or protocol["runner_sha256"] != baseline.file_digest(HERE)
            or protocol["cells"] != optimization.CELLS
            or protocol["schedule"] != schedule([item["id"] for item in workload["cases"]])):
        raise ValueError("Protocol differs from the qualified frozen configuration")


def read_batch(path, batch, protocol):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if [row["case"] for row in rows] != batch["case_order"]:
        raise ValueError("Measured batch is missing, duplicated or out of order")
    for position, row in enumerate(rows):
        if (row.get("phase") != "measurement" or row.get("protocol_sha256") != protocol["sha256"]
                or row.get("batch_index") != batch["batch_index"] or row.get("paired_round") != batch["round"]
                or row.get("label") != batch["label"] or row.get("batch_position") != position
                or row.get("layout") != protocol["cells"][batch["label"]]["layout"]
                or row.get("actual_prefix_caching") is not protocol["cells"][batch["label"]]["prefix_cache"]
                or row.get("qualification_identity") != protocol["qualification_identity"]
                or row.get("engine_identity") != protocol["engine_identity"]):
            raise ValueError("Measured row identity differs")
    return rows


async def collect_batch(args):
    import httpx
    from transformers import AutoTokenizer
    protocol = baseline.load_workload(args.protocol)
    workload, manifest = baseline.load_workload(args.workload), baseline.load_workload(args.manifest)
    qualification = optimization.load_qualification(args.qualification, protocol["qualification_identity"])
    check_protocol(protocol, workload, manifest, qualification)
    for name, expected in manifest["source_files_sha256"].items():
        if baseline.file_digest(optimization.ROOT / name) != expected:
            raise ValueError("Frozen source differs: " + name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory, local_files_only=True, trust_remote_code=False)
    for name, expected in manifest["tokenizer_files_sha256"].items():
        if baseline.file_digest(args.model_directory / name) != expected:
            raise ValueError("Tokenizer differs: " + name)
    batch = protocol["schedule"][args.batch_index]
    cell = protocol["cells"][batch["label"]]
    for layout in ("original", "evidence_first"):
        expected = {row["id"]: row for row in manifest["layouts"][layout]["requests"]}
        for item in workload["cases"]:
            tokens = baseline.chat_token_ids(tokenizer, optimization.prepared_messages(item, layout, workload["response_mode"]),
                **workload.get("chat_template_kwargs", {}))
            if len(tokens) != expected[item["id"]]["input_tokens"] or baseline.digest(tokens) != expected[item["id"]]["token_ids_sha256"]:
                raise ValueError("Complete request token identity differs")
    # 已完成的批次可恢复消费，但部分写入不自动重跑，以免重复远端副作用。
    for previous in protocol["schedule"][:args.batch_index]:
        read_batch(args.output_dir / result_name(previous), previous, protocol)
    outputs = [args.output_dir / result_name(batch, phase) for phase in ("warmup", "measurement")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Never overwrite or silently repeat an existing batch")
    async with httpx.AsyncClient(timeout=35, trust_env=False) as client:
        info = await client.get(args.base_url + "/server_info", params={"config_format": "json"})
        info.raise_for_status()
        version = await client.get(args.base_url + "/version")
        version.raise_for_status()
        engine = optimization.engine_measurement_identity(info.json(), version.json().get("version"), "xgrammar")
        actual_prefix = optimization.engine_prefix_setting(info.json())
        if engine != qualification["engine_identity"] or actual_prefix is not cell["prefix_cache"]:
            raise ValueError("Actual engine differs from this qualified cell")
        items = {item["id"]: item for item in workload["cases"]}
        for phase, output_path in zip(("warmup", "measurement"), outputs):
            reset = await client.post(args.base_url + "/reset_prefix_cache")
            reset.raise_for_status()
            order = [item["id"] for item in workload["cases"]] if phase == "warmup" else batch["case_order"]
            layout = "original" if phase == "warmup" else cell["layout"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            batch_started = time.perf_counter()
            with output_path.open("x", encoding="utf-8") as out:
                for position, case_id in enumerate(order):
                    provider = optimization.StreamingProvider(client, tokenizer, workload, items[case_id],
                        base_url=args.base_url, model=args.model, layout=layout, structured=True, backend="xgrammar")
                    row = await optimization.replay_case(provider, items[case_id], layout, response_mode=workload["response_mode"])
                    row.update(phase=phase, protocol_sha256=protocol["sha256"], batch_index=batch["batch_index"],
                        paired_round=batch["round"], label=batch["label"], batch_position=position,
                        qualification_identity=protocol["qualification_identity"], engine_identity=engine,
                        qualification_sha256=qualification["sha256"], actual_prefix_caching=actual_prefix,
                        cache_state="reset_before_different_question_batch", reset_http_status=reset.status_code,
                        batch_elapsed_seconds=time.perf_counter()-batch_started,
                        transport="same-host loopback", collected_at=datetime.now(UTC).isoformat())
                    try:
                        row["cold_reset_verified"] = optimization.verify_cache_observation(row, actual_prefix,
                            first_after_reset=position == 0)
                        row["measurement_valid"] = bool(row["calls"]) and all(
                            call["status"] == "returned" and call["cache_delta"]["vllm:num_preemptions_total"] == 0
                            for call in row["calls"])
                    except optimization.AssistantError as error:
                        row.update(measurement_valid=False, measurement_error=error.code)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(json.dumps({"phase": phase, "batch_index": batch["batch_index"], "case": case_id,
                        "cell": batch["label"], "machine_pass": row["machine_pass"],
                        "measurement_valid": row["measurement_valid"], "calls": row["model_calls"],
                        "task_seconds": row["task_elapsed_seconds"]}), flush=True)
                    if not row["measurement_valid"]:
                        raise RuntimeError("Measurement invalid; raw partial batch retained, no automatic replay")


def summarize(protocol, output_dir, review=None):
    rows, files = [], {}
    for batch in protocol["schedule"]:
        path = output_dir / result_name(batch)
        rows.extend(read_batch(path, batch, protocol))
        files[path.name] = baseline.file_digest(path)
    decisions = {}
    if review is not None:
        if (review.get("results") != files or review.get("protocol_sha256") != protocol["sha256"]
                or not str(review.get("reviewer", "")).strip() or not str(review.get("method", "")).strip()):
            raise ValueError("Independent review source identity is missing or differs")
        for entry in review["cases"]:
            key = (entry["batch_index"], entry["case"])
            if key in decisions or type(entry.get("accepted")) is not bool or not str(entry.get("reason", "")).strip():
                raise ValueError("Independent review decision is incomplete or duplicated")
            decisions[key] = entry["accepted"]
        if set(decisions) != {(row["batch_index"], row["case"]) for row in rows}:
            raise ValueError("Independent review does not cover every measured task")
    cells = {}
    for label in optimization.CELLS:
        group = [row for row in rows if row["label"] == label]
        calls = [call for row in group for call in row["calls"]]
        first = [row["calls"][0] for row in group]
        # 区分不同问题的首调用命中和同题纠错复用，不能以总命中掩盖来源。
        cross = [row["calls"][0] for row in group if row["batch_position"] > 0]
        hit = lambda selected: sum(call.get("cache_delta", {}).get("vllm:prefix_cache_hits_total", 0) for call in selected)
        query = lambda selected: sum(call.get("cache_delta", {}).get("vllm:prefix_cache_queries_total", 0) for call in selected)
        cells[label] = {"tasks": len(group), "machine_pass": sum(row["machine_pass"] for row in group),
            "semantic_pass": sum(decisions[(row["batch_index"], row["case"])] for row in group) if review else None,
            "measurement_valid": sum(row["measurement_valid"] for row in group), "calls": len(calls),
            "output_tokens": sum((call.get("usage") or {}).get("completion_tokens", 0) for call in calls),
            "task_median_seconds": median(row["task_elapsed_seconds"] for row in group),
            "first_call_ttft_median_seconds": median(call["ttft_seconds"] for call in first),
            "all_call_cache_hits_tokens": hit(calls), "cross_question_first_call_hits_tokens": hit(cross),
            "cross_question_first_call_queries_tokens": query(cross),
            "cold_resets_verified": sum(row.get("cold_reset_verified") is True for row in group)}
    paired = []
    for left, right in (("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D")):
        for round_number in range(1, len(ORDERS)+1):
            a = {row["case"]: row for row in rows if row["label"] == left and row["paired_round"] == round_number}
            b = {row["case"]: row for row in rows if row["label"] == right and row["paired_round"] == round_number}
            def gain(key):
                av, bv = median(key(row) for row in a.values()), median(key(row) for row in b.values())
                return {"left_seconds": av, "right_seconds": bv, "relative_reduction": (av-bv)/av if av else None}
            paired.append({"left": left, "right": right, "round": round_number,
                "task": gain(lambda row: row["task_elapsed_seconds"]),
                "first_call_ttft": gain(lambda row: row["calls"][0]["ttft_seconds"]),
                "identical_raw_generation_tasks": sum(
                    [call.get("content") for call in a[case]["calls"]] == [call.get("content") for call in b[case]["calls"]]
                    for case in a),
                "same_call_and_output_token_tasks": sum(
                    [(call.get("usage") or {}).get("completion_tokens") for call in a[case]["calls"]]
                    == [(call.get("usage") or {}).get("completion_tokens") for call in b[case]["calls"]]
                    for case in a)})
    return {"protocol_sha256": protocol["sha256"], "results": files, "tasks": len(rows),
        "semantic_review": "complete" if review else "pending", "cells": cells, "paired_rounds": paired,
        "quality_eligible_cells": [label for label, value in cells.items() if review
            and value["tasks"] == value["machine_pass"] == value["semantic_pass"] == value["measurement_valid"]
            and value["cold_resets_verified"] == len(ORDERS)],
        "adoption_decision": "Requires explicit evidence review; no automatic latency-only adoption",
        "limits": "Three paired rounds of known development questions, one model/seed; warmup excluded; not unseen generalization, concurrency or HTTP/PG reliability"}


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    plan = actions.add_parser("plan")
    for name in ("workload", "manifest", "qualification", "output"):
        plan.add_argument("--"+name, type=Path, required=True)
    collect = actions.add_parser("batch")
    for name in ("workload", "manifest", "qualification", "protocol", "model-directory", "output-dir"):
        collect.add_argument("--"+name, type=Path, required=True)
    collect.add_argument("--batch-index", type=int, choices=range(len(ORDERS)*4), required=True)
    collect.add_argument("--base-url", default="http://127.0.0.1:18110")
    collect.add_argument("--model", required=True)
    report = actions.add_parser("summarize")
    for name in ("protocol", "output-dir", "output"):
        report.add_argument("--"+name, type=Path, required=True)
    report.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        value = make_protocol(args.workload, args.manifest, args.qualification, args.output)
        print(json.dumps({"protocol_sha256": value["sha256"], "batches": len(value["schedule"]), "model_requests": 0}))
    elif args.action == "batch":
        import fcntl
        with (args.model_directory.resolve().parent / "engine-replay.lock").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another measurement owns this engine; no request sent")
            asyncio.run(collect_batch(args))
    else:
        protocol = baseline.load_workload(args.protocol)
        review = json.loads(args.review.read_bytes()) if args.review else None
        save_new(args.output, summarize(protocol, args.output_dir, review))


if __name__ == "__main__":
    main()
