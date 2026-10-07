"""复用实际回答/纠错代码的完整输入对照；引擎与业务语义分别验收。"""

import argparse
import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("inference_baseline", Path(__file__).with_name("inference_baseline.py"))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer, Question
from power_forecast_service.assistant.prompting import answer_messages
from power_forecast_service.assistant.workflow import Assistant, prepare_answer_context, response_contract, TIMEOUT_SECONDS
from power_forecast_service.assistant.references import reference_schema
from power_forecast_service.assistant.coverage import coverage_requirements

CELLS = {
    "A": {"layout": "original", "prefix_cache": False},
    "B": {"layout": "original", "prefix_cache": True},
    "C": {"layout": "evidence_first", "prefix_cache": False},
    "D": {"layout": "evidence_first", "prefix_cache": True},
}


def qualification_identity(workload, manifest, decoding, backend):
    """模型、输入、代码及解码共同定义资格；布局/APC是后续实验因素。"""
    if backend not in ("xgrammar", "guidance"):
        raise ValueError("An explicit structured-output backend is required")
    if decoding == "json_schema" and backend == "xgrammar" and workload.get("response_mode", "legacy") == "legacy":
        # 当前完整schema的quotes.maxProperties在冻结版本不受支持，不交给auto隐式回退。
        raise ValueError("Full contract schema requires guidance on vLLM0.11.2")
    mode = workload.get("response_mode", "legacy")
    _, _, prompt_version = response_contract(mode)
    if manifest.get("response_mode", "legacy") != mode:
        raise ValueError("Response mode differs from the frozen manifest")
    return {"workload_sha256": workload["sha256"], "manifest_sha256": manifest["sha256"],
        "model": workload["model_candidate"], "revision": workload["model_revision"],
        "prompt_version": prompt_version, "response_mode": mode, "decoding": decoding, "structured_backend": backend,
        "max_model_calls": 2, "max_tokens": workload["max_tokens"], "task_deadline_seconds": TIMEOUT_SECONDS,
        "read_timeout_seconds": 35, "request_deadline_seconds": None,
        "decoder_schemas_sha256": baseline.digest([contract_schema(item, mode) for item in workload["cases"]])
            if decoding == "json_schema" else None}


def qualification_evidence(record, directory):
    """资格摘要必须仍能定位原结果及其独立语义审阅，哈希不能替代审阅。"""
    engine = record.get("engine_identity", {})
    if (not isinstance(engine, dict) or engine_measurement_identity({"vllm_config": engine},
            engine.get("version"), record["identity"]["structured_backend"]) != engine):
        raise ValueError("Qualified engine identity is incomplete")
    sources = []
    for key in ("results", "review"):
        path = (directory / record[key]["file"]).resolve()
        if not path.is_relative_to(directory.resolve()) or baseline.file_digest(path) != record[key]["sha256"]:
            raise ValueError("Qualification evidence identity differs")
        sources.append(path)
    rows = [json.loads(line) for line in sources[0].read_text(encoding="utf-8").splitlines()]
    review = baseline.load_workload(sources[1])
    cases = record["cases"]
    if (len(rows) != len(cases) or {row["case"] for row in rows} != set(cases)
            or len(set(cases)) != len(cases) or not cases):
        raise ValueError("Qualification must cover every case exactly once")
    reviews = review.get("cases", [])
    if (review.get("results_sha256") != record["results"]["sha256"]
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
            or not isinstance(review.get("method"), str) or not review["method"].strip()
            or len(reviews) != len(cases) or {row["case"] for row in reviews} != set(cases)
            or any(row.get("accepted") is not True or not isinstance(row.get("reason"), str)
                   or not row["reason"].strip() for row in reviews)):
        raise ValueError("Independent semantic qualification is missing or failed")
    for row in rows:
        calls = row.get("calls", [])
        if (row.get("label") not in ("qualification", "schema") or row.get("repetition") != 1
                or row.get("layout") != "original" or row.get("machine_pass") is not True
                or row.get("measurement_valid") is not True or row.get("missing_required_facts")
                or row.get("result", {}).get("status") != "answered" or not 1 <= len(calls) <= 2
                or row.get("model_calls") != len(calls)
                or any(call.get("status") != "returned" or not call.get("complete_input_verified")
                       or call.get("finish_reason") != "stop" for call in calls)
                or row.get("qualification_identity") != record["identity"]
                or row.get("engine_identity") != record["engine_identity"]):
            raise ValueError("Machine qualification is failed, incomplete or configuration-mismatched")


def load_qualification(path, identity):
    if path is None:
        raise ValueError("Factorial measurement requires a passed qualification")
    record = baseline.load_workload(path)
    if record.get("status") != "passed" or record.get("identity") != identity:
        raise ValueError("Qualification configuration differs or did not pass")
    qualification_evidence(record, path.parent)
    return record


def seal_qualification(workload_path, manifest_path, results_path, review_path, output, decoding, backend):
    workload, manifest = baseline.load_workload(workload_path), baseline.load_workload(manifest_path)
    if manifest["workload_sha256"] != workload["sha256"]:
        raise ValueError("Qualification workload differs from manifest")
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Qualification results are empty")
    root = output.parent.resolve()
    if not all(path.resolve().is_relative_to(root) for path in (results_path, review_path)):
        raise ValueError("Qualification evidence must remain beside its certificate")
    record = {"schema_version": 1, "status": "passed", "created_at": datetime.now(UTC).isoformat(),
        "identity": qualification_identity(workload, manifest, decoding, backend),
        "engine_identity": rows[0].get("engine_identity"), "cases": [item["id"] for item in workload["cases"]],
        "results": {"file": results_path.resolve().relative_to(root).as_posix(), "sha256": baseline.file_digest(results_path)},
        "review": {"file": review_path.resolve().relative_to(root).as_posix(), "sha256": baseline.file_digest(review_path)}}
    if not record["engine_identity"]:
        raise ValueError("Actual qualification engine identity is missing")
    qualification_evidence(record, root)
    record["sha256"] = baseline.digest(record)
    baseline.save(output, record)
    print(json.dumps({"qualification": "passed", "cases": len(record["cases"]), "sha256": record["sha256"]}))


def decoding_mode(label, requested):
    """资格检查固定解码模式；四格对照必须显式复用同一已选择模式。"""
    expected = {"qualification": "json_object", "schema": "json_schema"}.get(label)
    if expected:
        if requested is not None and requested != expected:
            raise ValueError("Qualification label and decoding mode differ")
        return expected
    if label not in CELLS or requested not in ("json_object", "json_schema"):
        raise ValueError("Factorial cells require an explicit common decoding mode")
    return requested


def generation_options(workload):
    """独立候选只可改变已声明生成参数，不能借配置覆盖完整消息或调用预算。"""
    sampling = workload.get("sampling", {})
    template = workload.get("chat_template_kwargs", {})
    if set(sampling) - {"top_p", "top_k", "min_p"} or set(template) - {"enable_thinking"}:
        raise ValueError("Unsupported generation configuration")
    if "enable_thinking" in template and type(template["enable_thinking"]) is not bool:
        raise ValueError("Thinking mode must be explicit boolean")
    return sampling, template


def json_messages(messages):
    return [{"role": role, "content": content} for role, content in messages]


def question_for(item):
    payload = json.loads(item["messages"][1]["content"])
    return Question(question=item["question"], contexts=payload["contexts"])


def prepared_messages(item, layout, response_mode="legacy"):
    schema, instructions, _ = response_contract(response_mode)
    data, docs = prepare_answer_context(deepcopy(item["evidence"]), deepcopy(item["documents"]), item["question"], response_mode)
    actual_schema = reference_schema(data, docs, coverage_requirements(data, item["question"])) if response_mode == "references" else schema.model_json_schema()
    return json_messages(answer_messages(question_for(item), data, docs,
        instructions, actual_schema, layout=layout,
        schema_placement="user_tail" if response_mode == "references" else "system"))


def rebind_reference_workload(source, output):
    """显式冻结新输出请求；保留原任务和外部验收，不覆盖旧请求或伪称字节相同。"""
    value = baseline.load_workload(source)
    if value.get("response_mode", "legacy") != "legacy":
        raise ValueError("Reference rebind requires the original legacy workload")
    original = deepcopy(value)
    for item in value["cases"]:
        messages = prepared_messages(item, "original", "references")
        payload = json.loads(messages[1]["content"])
        # schema迁入请求尾部是显式表示变化；任何原问题、事实、文档或约束变化仍必须走refresh。
        payload.pop("output_schema")
        if payload != json.loads(item["messages"][1]["content"]):
            raise ValueError("Reference rebind changed full user evidence: " + item["id"])
        baseline.public_payload(messages)
        item["messages"] = messages
    value.pop("sha256")
    value["parent_workload_sha256"] = original["sha256"]
    value["original_capture_metadata"] = {key: original.get(key) for key in (
        "created_at", "source_commit", "workflow_sha256", "prompt_version", "assembly")}
    value.update(response_mode="references", prompt_version=response_contract("references")[2],
                 created_at=datetime.now(UTC).isoformat(),
                 workflow_sha256=baseline.file_digest(ROOT / "src/power_forecast_service/assistant/workflow.py"),
                 assembly="Offline response-contract rebind of the complete original capture; no new PG read, model generation or acceptance",
                 representation_change="Instructions/schema changed and schema moved to user tail; all original user evidence fields, questions, cases, required_facts and semantic_rubric preserved")
    value["sha256"] = baseline.digest(value)
    baseline.save(output, value)
    print(json.dumps({"cases": len(value["cases"]), "response_mode": "references",
                      "workload_sha256": value["sha256"], "parent_workload_sha256": original["sha256"],
                      "original_user_evidence_equal": True, "complete_user_payload_equal": False, "original_message_bytes_equal": False,
                      "model_calls": 0}))


def refresh_derived_workload(source, output):
    """独立的新输入版本：只增加来源事实并重算角色选项，原任务和验收不回灌运行时。"""
    original = baseline.load_workload(source)
    value = deepcopy(original)
    changes = []
    for item in value["cases"]:
        old_facts, old_docs = deepcopy(item["evidence"]["facts"]), deepcopy(item["documents"])
        data, docs = prepare_answer_context(item["evidence"], item["documents"], item["question"], "references")
        if any(f not in data["facts"] for f in old_facts) or any(d not in docs for d in old_docs):
            raise ValueError("Derived refresh removed or altered original evidence: " + item["id"])
        item.update(evidence=data, documents=docs)
        item["messages"] = prepared_messages(item, "original", "references")
        baseline.public_payload(item["messages"])
        changes.append({"id": item["id"], "added_fact_ids": [f["id"] for f in data["facts"] if f not in old_facts],
                        "added_document_ids": [d["id"] for d in docs if d not in old_docs]})
    value.pop("sha256")
    value.update(parent_workload_sha256=original["sha256"], response_mode="references",
        prompt_version=response_contract("references")[2], created_at=datetime.now(UTC).isoformat(),
        workflow_sha256=baseline.file_digest(ROOT / "src/power_forecast_service/assistant/workflow.py"),
        assembly="Shared actual consumer preparation, source-derived evidence and revised role options; no new PG read or model calls",
        representation_change="Explicit evidence revision; original questions/facts/documents, model/sampling and independent acceptance preserved",
        derived_evidence_changes=changes)
    value["sha256"] = baseline.digest(value)
    baseline.save(output, value)
    return {"workload_sha256": value["sha256"], "parent_workload_sha256": original["sha256"], "changes": changes}


def same_complete_input(left, right):
    """比较全部值，不把相同长度或相同 token 数误当成内容相同。"""
    return (len(left) == len(right) == 2 and left[0] == right[0]
            and left[1]["role"] == right[1]["role"] == "user"
            and json.loads(left[1]["content"]) == json.loads(right[1]["content"]))


def cache_metrics(text):
    from prometheus_client.parser import text_string_to_metric_families
    wanted = {"vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
              "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"}
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in wanted:
                values[sample.name] = values.get(sample.name, 0) + sample.value
    return values


def cache_delta(before, after):
    """累计计数缺失或倒退不能当成零命中；冷缓存由首个实际请求核验。"""
    counters = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                "vllm:num_preemptions_total")
    if any(name not in before or name not in after for name in counters):
        raise AssistantError("provider_metrics_incomplete")
    result = {name: after[name] - before[name] for name in counters}
    if any(value < 0 for value in result.values()):
        raise AssistantError("provider_metrics_reset_during_request")
    if result[counters[1]] > result[counters[0]]:
        raise AssistantError("provider_metrics_inconsistent")
    return result


def verify_cache_observation(row, prefix_enabled, *, first_after_reset):
    """不能把 HTTP 200 或未观测到的计数变化记成有效对照。"""
    for index, call in enumerate(row["calls"]):
        if call["status"] != "returned":
            continue
        hits = call["cache_delta"]["vllm:prefix_cache_hits_total"]
        queries = call["cache_delta"]["vllm:prefix_cache_queries_total"]
        if prefix_enabled and queries != call["input_tokens"]:
            raise AssistantError("provider_cache_query_mismatch")
        if (not prefix_enabled or first_after_reset and index == 0) and hits != 0:
            raise AssistantError("provider_cache_reset_unverified")
    return bool(row["calls"] and row["calls"][0]["status"] == "returned"
                and first_after_reset)


def contract_schema(item, response_mode="legacy"):
    """限定现有合法 ID 的取值，不预填答案、不解除正文与阶段的外部验收。"""
    response_schema, _, _ = response_contract(response_mode)
    schema = deepcopy(response_schema.model_json_schema())
    citations = [document["id"] for document in item["documents"]] or ["__no_citation__"]
    if response_mode == "references":
        data, docs = prepare_answer_context(deepcopy(item["evidence"]), deepcopy(item["documents"]), item["question"], response_mode)
        return reference_schema(data, docs, coverage_requirements(data, item["question"]))
    properties = schema["properties"]
    facts = [fact["id"] for fact in item["evidence"]["facts"]]
    citations = [document["id"] for document in item["documents"]]
    if facts:
        properties["fact_ids"]["items"]["enum"] = sorted(set(facts))
    if citations:
        properties["citations"]["items"]["enum"] = sorted(set(citations))
        schema["$defs"]["StageClaim"]["properties"]["citations"]["items"]["enum"] = sorted(set(citations))
    objects = [str(option["object_id"]) for option in item["evidence"].get("stage_options", [])]
    if objects:
        schema["$defs"]["StageClaim"]["properties"]["object_id"]["enum"] = sorted(set(objects))
    return schema


def prepare(workload_path, model_directory, output):
    from transformers import AutoTokenizer
    value = baseline.load_workload(workload_path)
    generation_options(value)
    mode = value.get("response_mode", "legacy")
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
    layouts = {}
    for layout in ("original", "evidence_first"):
        requests, ids = [], []
        for item in value["cases"]:
            messages = prepared_messages(item, layout, mode)
            if not same_complete_input(item["messages"], messages):
                raise ValueError("Full evidence changed: " + item["id"])
            if layout == "original" and messages != item["messages"]:
                raise ValueError("Original first-call bytes changed: " + item["id"])
            tokens = baseline.chat_token_ids(tokenizer, messages, **value.get("chat_template_kwargs", {}))
            if len(tokens) + value["max_tokens"] > 32768:
                raise ValueError("Complete first call exceeds the frozen context reservation")
            ids.append(tokens)
            requests.append({"id": item["id"], "input_tokens": len(tokens),
                "message_sha256": baseline.digest(messages), "token_ids_sha256": baseline.digest(tokens),
                "full_value_equivalence": True, "original_bytes_equal": messages == item["messages"]})
        layouts[layout] = {"requests": requests,
            "common_prefix_tokens": [[baseline.common_prefix(a, b) for b in ids] for a in ids]}
    result = {"schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "response_mode": mode, "prompt_version": response_contract(mode)[2],
        "workload_sha256": value["sha256"], "model": value["model_candidate"], "revision": value["model_revision"],
        "max_model_len": 32768, "max_tokens": value["max_tokens"], "temperature": value["temperature"], "seed": value["seed"],
        "chat_template_kwargs": value.get("chat_template_kwargs", {}),
        "sampling": value.get("sampling", {}),
        "layouts": layouts, "cells": CELLS, "repetitions": 3,
        "quality_scope": "Original Assistant.answer, maximum two calls, machine acceptance and required facts; independent semantic review mandatory",
        "measurement_scope": "Same-host loopback, complete selected snapshots; no public HTTP/PG audit or production arrival process",
        "adoption_rule": "Preserve original acceptance and semantic correctness; cache benefit must survive different questions and repeated paired batches, not same-question replay only",
        "qualification_protocol": "Run existing original-layout repair once per case, then bounded legal-ID schema comparison. Select only a configuration with all six machine and independent semantic gates passed. Otherwise allow one separately frozen model candidate, never unlimited prompt rescue.",
        "factorial_decoding_rule": "After qualification, keep the selected decoding mode identical in A/B/C/D; never mix grammar gains into cache or layout gains",
        "later_preserved": "Independent task qualification, mixed real result snapshots, actual optional provider, admission/cancellation/recovery and I4 delivery",
        "source_files_sha256": {name: baseline.file_digest(ROOT / name) for name in (
            "src/power_forecast_service/assistant/workflow.py", "src/power_forecast_service/assistant/prompting.py",
            "src/power_forecast_service/assistant/coverage.py", "tools/diagnostics/inference_optimization.py",
            "src/power_forecast_service/assistant/contracts.py", "src/power_forecast_service/assistant/validation.py",
            "src/power_forecast_service/assistant/references.py",
            "src/power_forecast_service/assistant/temporal.py", "src/power_forecast_service/assistant/stage_sources.json",
            "src/power_forecast_service/assistant/stages.py", "src/power_forecast_service/assistant/corpus.json")},
        "tokenizer_files_sha256": {name: baseline.file_digest(model_directory / name)
                                   for name in ("tokenizer.json", "tokenizer_config.json")}, "truncated": False}
    result["sha256"] = baseline.digest(result)
    baseline.save(output, result)
    print(json.dumps({"cases": len(value["cases"]), "layouts": 2,
        "original_bytes_equal": all(row["original_bytes_equal"] for row in layouts["original"]["requests"]),
        "shared_prefix_original": layouts["original"]["common_prefix_tokens"][0][1],
        "shared_prefix_evidence_first": layouts["evidence_first"]["common_prefix_tokens"][0][1],
        "manifest_sha256": result["sha256"], "model_calls": 0}, ensure_ascii=False))


class StreamingProvider:
    """适配现有 ainvoke 接口；请求 token 身份与 usage 必须逐调用一致。"""

    def __init__(self, client, tokenizer, workload, item, *, base_url, model, layout, structured=False, backend=None):
        self.client, self.tokenizer = client, tokenizer
        self.workload, self.item = workload, item
        self.base_url, self.model = base_url, model
        self.layout, self.structured = layout, structured
        self.backend = backend
        self.calls = []

    async def ainvoke(self, messages):
        sampling, template = generation_options(self.workload)
        payload = json_messages(messages)
        if not same_complete_input(self.item["messages"], payload[:2]):
            raise AssistantError("provider_input_mismatch")
        mode = self.workload.get("response_mode", "legacy")
        if not self.calls and payload != prepared_messages(self.item, self.layout, mode):
            raise AssistantError("provider_input_mismatch")
        if self.calls and len(payload) != 4:
            raise AssistantError("provider_repair_input_mismatch")
        tokens = baseline.chat_token_ids(self.tokenizer, payload, **template)
        if len(tokens) + self.workload["max_tokens"] > 32768:
            raise AssistantError("provider_context_reservation_exceeded")
        body = {"model": self.model, "messages": payload, "max_tokens": self.workload["max_tokens"],
            "temperature": self.workload["temperature"], "seed": self.workload["seed"],
            "response_format": {"type": "json_object"}, "stream": True, "stream_options": {"include_usage": True}}
        if template:
            body["chat_template_kwargs"] = template
        body.update(sampling)
        if self.structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "wind_answer", "schema": contract_schema(self.item, mode), "strict": True}}
        record = {"attempt": len(self.calls)+1, "input_tokens": len(tokens),
            "token_ids_sha256": baseline.digest(tokens), "message_sha256": baseline.digest(payload),
            "full_evidence_verified": True, "request_dispatched": False, "status": "started",
            "structured_backend": self.backend, "decoder_schema_sha256": baseline.digest(body["response_format"]),
            "read_timeout_seconds": 35, "request_deadline_seconds": None,
            "task_deadline_seconds": TIMEOUT_SECONDS, "remote_outcome": "not_sent"}
        self.calls.append(record)
        started, first, event_times = time.perf_counter(), None, []
        content, usage, finish = "", None, None
        try:
            before = await self.client.get(self.base_url + "/metrics")
            before.raise_for_status()
            record["metrics_before"] = cache_metrics(before.text)
            # 引擎时延不包含客户端计数采样；任务总时延仍包含整个适配器。
            started = time.perf_counter()
            record["request_dispatched"] = True
            record["remote_outcome"] = "unknown"
            async with self.client.stream("POST", self.base_url + "/v1/chat/completions", json=body, timeout=35) as reply:
                reply.raise_for_status()
                async for line in reply.aiter_lines():
                    chunk = baseline.decode_sse(line)
                    if chunk is None or chunk == "done":
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            elapsed = time.perf_counter()-started
                            first = elapsed if first is None else first
                            event_times.append(elapsed)
                            content += delta
            record.update(status="returned", ttft_seconds=first, engine_e2e_seconds=time.perf_counter()-started,
                remote_outcome="returned",
                usage=usage, finish_reason=finish, content=content,
                content_event_gaps_seconds=[b-a for a, b in zip(event_times, event_times[1:])],
                event_gap_not_token_itl=True, complete_input_verified=(usage or {}).get("prompt_tokens") == len(tokens))
            after = await self.client.get(self.base_url + "/metrics")
            after.raise_for_status()
            record["metrics_after"] = cache_metrics(after.text)
            record["cache_delta"] = cache_delta(record["metrics_before"], record["metrics_after"])
            if not record["complete_input_verified"]:
                raise AssistantError("provider_input_mismatch")
            if finish != "stop":
                raise AssistantError("provider_incomplete_generation")
            return SimpleNamespace(content=content, usage_metadata={
                "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]}, response_metadata={"model_name": self.model})
        except BaseException as error:
            record.update(status="failed", error=getattr(error, "code", type(error).__name__),
                          provider_elapsed_seconds=time.perf_counter()-started,
                          content=content, usage=usage, finish_reason=finish, ttft_seconds=first)
            raise


async def replay_case(provider, item, layout, *, response_mode="legacy"):
    captures = []
    assistant = Assistant(None, provider=provider, capture_draft=captures.append, prompt_layout=layout,
                          response_mode=response_mode)
    state = {"question": question_for(item), "evidence": deepcopy(item["evidence"]),
             "documents": deepcopy(item["documents"]), "trace": {"model_calls": [], "tools": []}}
    started = time.perf_counter()
    row = {"case": item["id"], "layout": layout, "response_mode": response_mode, "semantic_review": "pending",
           "consumer": "Actual Assistant.answer; frozen real-PG evidence, not current PG audit or public HTTP"}
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            result = (await assistant.answer(state))["result"]
        missing = sorted(set(item["required_facts"]) - {fact["id"] for fact in result["facts"]})
        row.update(status="returned", result=result, missing_required_facts=missing,
                   machine_pass=result["status"] == "answered" and not missing)
    except (AssistantError, TimeoutError) as error:
        row.update(status="rejected", error=getattr(error, "code", "assistant_timeout"), machine_pass=False)
    finally:
        assistant.close()
        row.update(task_elapsed_seconds=time.perf_counter()-started, trace=state["trace"],
                   drafts=captures, calls=provider.calls, model_calls=len(provider.calls),
                   network_requests_attempted=sum(call.get("request_dispatched", False) for call in provider.calls))
    return row


def engine_configuration(payload):
    config = payload.get("vllm_config")
    if isinstance(config, str):
        config = json.loads(config)
    if not isinstance(config, dict):
        raise ValueError("Engine configuration is unavailable")
    return config


def engine_prefix_setting(payload):
    config = engine_configuration(payload)
    cache = config.get("cache_config")
    if not isinstance(cache, dict) or type(cache.get("enable_prefix_caching")) is not bool:
        raise ValueError("Actual prefix cache setting is unavailable")
    return cache["enable_prefix_caching"]


def engine_measurement_identity(payload, version, backend):
    """只排除APC开关；模型/KV/调度与后端不得在四格间偷偷变化。"""
    config = engine_configuration(payload)
    structured = config.get("structured_outputs_config", {})
    if version != "0.11.2" or structured.get("backend") != backend:
        raise ValueError("Actual engine version or pinned backend differs")
    model, cache, scheduler = (config.get(key, {}) for key in ("model_config", "cache_config", "scheduler_config"))
    if (not all(key in model for key in ("dtype", "quantization", "max_model_len"))
            or not all(key in cache for key in ("cache_dtype", "block_size"))
            or not all(key in scheduler for key in ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill"))):
        raise ValueError("Actual engine measurement configuration is incomplete")
    return {"version": version, "structured_outputs_config": structured,
        "model_config": {key: model.get(key) for key in ("dtype", "quantization", "max_model_len", "revision",
            "tokenizer_revision", "trust_remote_code", "served_model_name")},
        "cache_config": {key: cache.get(key) for key in ("cache_dtype", "block_size", "gpu_memory_utilization", "kv_cache_memory_bytes")},
        "scheduler_config": scheduler,
        "parallel_config": {key: config.get("parallel_config", {}).get(key) for key in ("tensor_parallel_size",
            "pipeline_parallel_size", "data_parallel_size", "distributed_executor_backend", "enable_expert_parallel")},
        "compilation_config": {key: config.get("compilation_config", {}).get(key) for key in ("mode", "backend",
            "cudagraph_mode", "cudagraph_capture_sizes", "custom_ops", "splitting_ops", "compile_sizes", "pass_config")},
        "speculative_config": config.get("speculative_config")}


async def benchmark(args):
    import httpx
    from transformers import AutoTokenizer
    value = baseline.load_workload(args.workload)
    decoding = decoding_mode(args.label, args.decoding)
    manifest = baseline.load_workload(args.manifest)
    if manifest["workload_sha256"] != value["sha256"]:
        raise ValueError("Workload does not match the frozen I2 plan")
    identity = qualification_identity(value, manifest, decoding, args.structured_backend)
    qualification = load_qualification(args.qualification, identity) if args.label in CELLS else None
    for name, expected in manifest["source_files_sha256"].items():
        if baseline.file_digest(ROOT / name) != expected:
            raise ValueError("Source identity differs: " + name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory, local_files_only=True, trust_remote_code=False)
    for name, expected in manifest["tokenizer_files_sha256"].items():
        if baseline.file_digest(args.model_directory / name) != expected:
            raise ValueError("Tokenizer identity differs: " + name)
    if args.output.exists():
        raise FileExistsError("Never overwrite a previous measurement")
    cell = CELLS.get(args.label, {"layout": "original", "prefix_cache": True})
    expected_tokens = {row["id"]: row for row in manifest["layouts"][cell["layout"]]["requests"]}
    for item in value["cases"]:
        tokens = baseline.chat_token_ids(tokenizer, prepared_messages(item, cell["layout"], value.get("response_mode", "legacy")),
                                         **value.get("chat_template_kwargs", {}))
        expected = expected_tokens[item["id"]]
        if len(tokens) != expected["input_tokens"] or baseline.digest(tokens) != expected["token_ids_sha256"]:
            raise ValueError("Full token identity differs before measurement: " + item["id"])
    async with httpx.AsyncClient(timeout=35, trust_env=False) as client:
        response = await client.get(args.base_url + "/health")
        response.raise_for_status()
        response = await client.get(args.base_url + "/server_info", params={"config_format": "json"})
        response.raise_for_status()
        actual_prefix = engine_prefix_setting(response.json())
        config_payload = response.json()
        version = await client.get(args.base_url + "/version")
        version.raise_for_status()
        engine = engine_measurement_identity(config_payload, version.json().get("version"), args.structured_backend)
        if qualification and engine != qualification["engine_identity"]:
            raise ValueError("Actual engine differs from the qualified configuration")
        if actual_prefix is not cell["prefix_cache"]:
            raise ValueError("Actual engine prefix cache setting differs from the selected cell")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            for repeat in range(args.repetitions):
                response = await client.post(args.base_url + "/reset_prefix_cache")
                response.raise_for_status()
                items = value["cases"][repeat % len(value["cases"]):] + value["cases"][:repeat % len(value["cases"])]
                batch_started = time.perf_counter()
                for index, item in enumerate(items):
                    before = await client.get(args.base_url + "/metrics")
                    before.raise_for_status()
                    provider = StreamingProvider(client, tokenizer, value, item, base_url=args.base_url,
                        model=args.model, layout=cell["layout"], structured=decoding == "json_schema", backend=args.structured_backend)
                    row = await replay_case(provider, item, cell["layout"], response_mode=value.get("response_mode", "legacy"))
                    after = await client.get(args.base_url + "/metrics")
                    after.raise_for_status()
                    row.update(label=args.label, decoding=decoding, repetition=repeat+1, batch_position=index,
                        qualification_identity=identity, engine_identity=engine,
                        qualification_sha256=qualification["sha256"] if qualification else None,
                        actual_prefix_caching=actual_prefix, workload_sha256=value["sha256"],
                        manifest_sha256=manifest["sha256"], metrics_before=before.text, metrics_after=after.text,
                        batch_elapsed_seconds=time.perf_counter()-batch_started,
                        cache_state="reset_before_different_question_batch", transport="same-host loopback",
                        collected_at=datetime.now(UTC).isoformat())
                    cache_error = None
                    try:
                        row["cold_reset_verified"] = verify_cache_observation(row, actual_prefix,
                            first_after_reset=index == 0)
                        row["measurement_valid"] = bool(row["calls"]) and all(
                            call["status"] == "returned" for call in row["calls"])
                    except AssistantError as error:
                        row.update(measurement_valid=False, measurement_error=error.code)
                        cache_error = error
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    print(json.dumps({"case": row["case"], "cell": args.label, "repetition": repeat+1,
                        "calls": row["model_calls"], "machine_pass": row["machine_pass"],
                        "error": row.get("error"), "task_seconds": row["task_elapsed_seconds"]}), flush=True)
                    if cache_error:
                        raise cache_error


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action", required=True)
    rebind = actions.add_parser("rebind-reference-workload")
    rebind.add_argument("--workload", type=Path, required=True)
    rebind.add_argument("--output", type=Path, required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--workload", type=Path, required=True)
    prep.add_argument("--model-directory", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    seal = actions.add_parser("seal-qualification")
    for name in ("workload", "manifest", "results", "review", "output"):
        seal.add_argument("--" + name, type=Path, required=True)
    seal.add_argument("--decoding", choices=["json_object", "json_schema"], required=True)
    seal.add_argument("--structured-backend", choices=["xgrammar", "guidance"], required=True)
    bench = actions.add_parser("benchmark")
    bench.add_argument("--workload", type=Path, required=True)
    bench.add_argument("--manifest", type=Path, required=True)
    bench.add_argument("--model-directory", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--base-url", default="http://127.0.0.1:18110")
    bench.add_argument("--model", default="inference-i1")
    bench.add_argument("--label", choices=[*CELLS, "qualification", "schema"], required=True)
    bench.add_argument("--decoding", choices=["json_object", "json_schema"])
    bench.add_argument("--structured-backend", choices=["xgrammar", "guidance"], required=True)
    bench.add_argument("--qualification", type=Path)
    bench.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.action == "rebind-reference-workload":
        rebind_reference_workload(args.workload, args.output)
        return
    if args.action == "prepare":
        prepare(args.workload, args.model_directory, args.output)
        return
    if args.action == "seal-qualification":
        seal_qualification(args.workload, args.manifest, args.results, args.review, args.output,
                           args.decoding, args.structured_backend)
        return
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    try:
        decoding_mode(args.label, args.decoding)
    except ValueError as error:
        parser.error(str(error))
    # 同一引擎的质量和性能批次互斥；文件锁在进程退出时由操作系统释放。
    import fcntl
    # 与 I1 测量器使用模型目录的同一父目录，不能因结果子目录不同而绕开互斥。
    with (args.model_directory.resolve().parent / "engine-replay.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another measurement owns this engine; no request sent")
        asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
