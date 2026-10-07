"""验收对照工具的真实消费和防错边界；不把模拟 provider 当成模型成绩。"""

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import tomllib
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from power_forecast_service.assistant.contracts import AssistantError, DraftAnswer, Question
from power_forecast_service.assistant import workflow

spec = importlib.util.spec_from_file_location("inference_optimization",
    Path(__file__).parents[2] / "tools/diagnostics/inference_optimization.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_metrics_tool_dependency_matches_the_checked_in_lock():
    """工具依赖取自已有锁，测试容器显式消费；生产镜像不借 Jupyter 安装它。"""
    root = Path(__file__).resolve().parents[2]
    packages = tomllib.loads((root / "uv.lock").read_text("utf-8"))["package"]
    package = next(row for row in packages if row["name"] == "prometheus-client")
    requirements = (root / "infra/inference-tools-requirements.txt").read_text("utf-8")
    assert f'prometheus-client=={package["version"]} \\' in requirements
    assert set(re.findall(r"sha256:[a-f0-9]{64}", requirements)) == {
        package["sdist"]["hash"], *[wheel["hash"] for wheel in package["wheels"]]}
    check = (root / "tools/dev/check-service.ps1").read_text("utf-8")
    assert "--require-hashes -r .local/runtime/test-requirements.txt -r infra/inference-tools-requirements.txt" in check


def item():
    question = Question(question="MAE是多少？", contexts=[{"kind": "engie_import", "id": UUID(int=1)}])
    evidence = {"facts": [{"id": "c0.ridge.mae", "label": "MAE", "value": 12.5, "unit": "kW"}],
        "records": [{"models": {"ridge": {}}}], "scopes": [], "stage_requirements": [], "stage_options": []}
    case = {"id": "N", "question": question.question, "evidence": evidence, "documents": [],
            "required_facts": ["c0.ridge.mae"]}
    from power_forecast_service.assistant.prompting import answer_messages
    case["messages"] = module.json_messages(answer_messages(question, evidence, [],
        workflow.INSTRUCTIONS, DraftAnswer.model_json_schema()))
    return case


class Provider:
    def __init__(self, drafts):
        self.drafts, self.calls = drafts, []

    async def ainvoke(self, messages):
        self.calls.append({"messages": deepcopy(messages)})
        return SimpleNamespace(content=json.dumps(self.drafts[len(self.calls)-1]),
                               usage_metadata={}, response_metadata={})


@pytest.mark.parametrize("layout", ["original", "evidence_first"])
def test_actual_answer_reuses_single_existing_repair(monkeypatch, layout):
    monkeypatch.setattr(workflow, "prepare_stage_requirements", lambda evidence, docs, question: (evidence, docs))
    provider = Provider([
        {"status": "answered", "answer": "MAE为12.5。", "fact_ids": []},
        {"status": "answered", "answer": "MAE为{{c0.ridge.mae}}。", "fact_ids": ["c0.ridge.mae"]}])
    result = asyncio.run(module.replay_case(provider, item(), layout))
    assert result["machine_pass"] is True and result["model_calls"] == 2
    assert result["trace"]["repair_reason"] == "answer_number_unbound"
    assert result["result"]["answer"] == "MAE为MAE：12.5 kW。"
    messages = provider.calls[1]["messages"]
    assert len(messages) == 4 and messages[2][0] == "assistant"
    assert "answer_number_unbound" in messages[3][1]
    assert result["semantic_review"] == "pending"


def test_schema_failure_uses_same_single_repair_budget(monkeypatch):
    monkeypatch.setattr(workflow, "prepare_stage_requirements", lambda evidence, docs, question: (evidence, docs))
    provider = Provider([{"status": "answered", "answer": 123}] * 2)
    result = asyncio.run(module.replay_case(provider, item(), "original"))
    assert result["machine_pass"] is False
    assert result["error"] == "answer_schema_invalid" and result["model_calls"] == 2


def test_required_fact_gate_cannot_be_bypassed_by_machine_answer(monkeypatch):
    monkeypatch.setattr(workflow, "prepare_stage_requirements", lambda evidence, docs, question: (evidence, docs))
    case = item()
    case["required_facts"] += ["c0.other.mae"]
    provider = Provider([{"status": "answered", "answer": "{{c0.ridge.mae}}", "fact_ids": ["c0.ridge.mae"]}])
    result = asyncio.run(module.replay_case(provider, case, "original"))
    assert result["status"] == "returned"
    assert result["machine_pass"] is False
    assert result["missing_required_facts"] == ["c0.other.mae"]


def test_structured_schema_binds_ids_without_writing_a_semantic_answer():
    case = item()
    schema = module.contract_schema(case)
    assert schema["properties"]["fact_ids"]["items"]["enum"] == ["c0.ridge.mae"]
    assert schema["properties"]["answer"] == DraftAnswer.model_json_schema()["properties"]["answer"]
    assert DraftAnswer.model_json_schema()["properties"]["fact_ids"]["items"] == {"type": "string"}
    assert case == item()


@pytest.mark.parametrize("label,requested,expected", [
    ("qualification", None, "json_object"), ("schema", None, "json_schema"),
    ("A", "json_object", "json_object"), ("D", "json_schema", "json_schema"),
])
def test_qualification_and_factorial_decoding_are_separate(label, requested, expected):
    assert module.decoding_mode(label, requested) == expected


@pytest.mark.parametrize("label,requested", [("A", None), ("schema", "json_object"),
    ("qualification", "json_schema"), ("D", "unknown")])
def test_decoding_cannot_silently_change_between_qualification_and_cells(label, requested):
    with pytest.raises(ValueError):
        module.decoding_mode(label, requested)


def test_candidate_configuration_cannot_override_evidence_or_budget():
    assert module.generation_options({}) == ({}, {})
    with pytest.raises(ValueError, match="Unsupported"):
        module.generation_options({"sampling": {"messages": []}})
    with pytest.raises(ValueError, match="Unsupported"):
        module.generation_options({"sampling": {"max_tokens": 200}})
    with pytest.raises(ValueError, match="boolean"):
        module.generation_options({"chat_template_kwargs": {"enable_thinking": "false"}})


def test_explicit_candidate_template_flag_reaches_tokenizer_without_changing_messages():
    received = []
    class CandidateTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            received.append((deepcopy(messages), kwargs))
            return [1, 2, 3]
    messages = item()["messages"]
    assert module.baseline.chat_token_ids(CandidateTokenizer(), messages, enable_thinking=False) == [1, 2, 3]
    assert received[0][0] == messages
    assert received[0][1] == {"tokenize": True, "add_generation_prompt": True,
                             "return_dict": False, "enable_thinking": False}


def test_full_input_equivalence_is_by_value_not_length():
    case = item()
    first = module.prepared_messages(case, "original")
    ordered = module.prepared_messages(case, "evidence_first")
    assert first == case["messages"]
    assert module.same_complete_input(first, ordered)
    payload = json.loads(ordered[1]["content"])
    payload["evidence"]["facts"][0]["value"] = 12.6
    ordered[1]["content"] = json.dumps(payload, ensure_ascii=False)
    assert not module.same_complete_input(first, ordered)


@pytest.mark.parametrize("payload,expected", [
    ({"vllm_config": {"cache_config": {"enable_prefix_caching": True}}}, True),
    ({"vllm_config": json.dumps({"cache_config": {"enable_prefix_caching": False}})}, False),
])
def test_actual_engine_cache_setting(payload, expected):
    assert module.engine_prefix_setting(payload) is expected


@pytest.mark.parametrize("payload", [{}, {"vllm_config": {}},
    {"vllm_config": {"cache_config": {"enable_prefix_caching": "False"}}}])
def test_missing_engine_configuration_is_not_assumed(payload):
    with pytest.raises(ValueError, match="unavailable"):
        module.engine_prefix_setting(payload)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
        return list(range(30))


@pytest.mark.parametrize("tokens,finish,error", [
    (29, "stop", "provider_input_mismatch"),
    (30, "length", "provider_incomplete_generation"),
    (30, "stop", None),
])
def test_streaming_provider_requires_complete_real_usage(tokens, finish, error):
    case = item()
    received = []
    content = json.dumps({"status": "answered", "answer": "中文完整回答"}, ensure_ascii=False)
    def transport(request):
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:prefix_cache_hits_total 0\nvllm:prefix_cache_queries_total 0\nvllm:num_preemptions_total 0\n")
        received.append(json.loads(request.content))
        chunks = [{"choices": [{"delta": {"content": content}, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": finish}], "usage": {
                      "prompt_tokens": tokens, "completion_tokens": 10, "total_tokens": tokens+10}}]
        text = "\n\n".join("data: " + json.dumps(chunk, ensure_ascii=False) for chunk in chunks) + "\n\ndata: [DONE]\n\n"
        return httpx.Response(200, content=text.encode(), headers={"content-type": "text/event-stream"})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            provider = module.StreamingProvider(client, Tokenizer(), {"max_tokens": 1700,
                "temperature": 0, "seed": 42}, case, base_url="http://test.invalid", model="test",
                layout="original")
            messages = [(row["role"], row["content"]) for row in case["messages"]]
            if error:
                with pytest.raises(AssistantError) as raised:
                    await provider.ainvoke(messages)
                assert raised.value.code == error
            else:
                reply = await provider.ainvoke(messages)
                assert json.loads(reply.content)["answer"] == "中文完整回答"
            assert provider.calls[0]["complete_input_verified"] is (tokens == 30)
            assert received[0]["messages"] == case["messages"]
            assert received[0]["max_tokens"] == 1700
    asyncio.run(run())


def test_cache_counters_sum_engine_labels_and_detect_missing_or_reset_metrics():
    before = module.cache_metrics('vllm:prefix_cache_hits_total{engine="0"} 10\n'
        'vllm:prefix_cache_queries_total{engine="0"} 30\nvllm:num_preemptions_total 0\n')
    after = module.cache_metrics('vllm:prefix_cache_hits_total{engine="0"} 12\n'
        'vllm:prefix_cache_hits_total{engine="1"} 3\n'
        'vllm:prefix_cache_queries_total{engine="0"} 38\nvllm:num_preemptions_total 0\n')
    assert module.cache_delta(before, after)["vllm:prefix_cache_hits_total"] == 5
    with pytest.raises(AssistantError) as missing:
        module.cache_delta({}, after)
    assert missing.value.code == "provider_metrics_incomplete"
    with pytest.raises(AssistantError) as reset:
        module.cache_delta(after, before)
    assert reset.value.code == "provider_metrics_reset_during_request"


@pytest.mark.parametrize("enabled,first,hits,queries,error", [
    (True, True, 0, 30, None), (True, False, 24, 30, None),
    (False, True, 0, 0, None), (True, True, 4, 30, "provider_cache_reset_unverified"),
    (False, False, 4, 30, "provider_cache_reset_unverified"),
    (True, False, 4, 29, "provider_cache_query_mismatch"),
])
def test_cache_state_is_proven_by_calls_not_reset_http(enabled, first, hits, queries, error):
    row = {"calls": [{"status": "returned", "input_tokens": 30, "cache_delta": {
        "vllm:prefix_cache_hits_total": hits, "vllm:prefix_cache_queries_total": queries}}]}
    if error:
        with pytest.raises(AssistantError) as raised:
            module.verify_cache_observation(row, enabled, first_after_reset=first)
        assert raised.value.code == error
    else:
        assert module.verify_cache_observation(row, enabled, first_after_reset=first) is first


def test_repair_reuse_is_not_misclassified_as_failed_cold_reset():
    row = {"calls": [{"status": "returned", "input_tokens": 30, "cache_delta": {
        "vllm:prefix_cache_hits_total": 0, "vllm:prefix_cache_queries_total": 30}},
        {"status": "returned", "input_tokens": 45, "cache_delta": {
        "vllm:prefix_cache_hits_total": 24, "vllm:prefix_cache_queries_total": 45}}]}
    assert module.verify_cache_observation(row, True, first_after_reset=True)


def test_modified_complete_input_is_rejected_before_network():
    case = item()
    calls = []
    def transport(request):
        calls.append(request)
        raise AssertionError("Network must not be reached")
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            provider = module.StreamingProvider(client, Tokenizer(), {"max_tokens": 1700}, case,
                base_url="http://test.invalid", model="test", layout="original")
            bad = deepcopy(case["messages"])
            bad[1]["content"] = json.dumps({"question": "删掉证据的错误输入"})
            with pytest.raises(AssistantError) as raised:
                await provider.ainvoke([(m["role"], m["content"]) for m in bad])
            assert raised.value.code == "provider_input_mismatch"
            assert calls == [] and provider.calls == []
    asyncio.run(run())


def test_failed_metric_probe_is_not_counted_as_a_dispatched_model_request():
    case = item()
    received = []
    def transport(request):
        received.append(request.url.path)
        assert request.url.path == "/metrics"
        return httpx.Response(503)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            provider = module.StreamingProvider(client, Tokenizer(), {"max_tokens": 1700,
                "temperature": 0, "seed": 42}, case, base_url="http://test.invalid",
                model="test", layout="original")
            messages = [(m["role"], m["content"]) for m in case["messages"]]
            with pytest.raises(httpx.HTTPStatusError):
                await provider.ainvoke(messages)
            assert provider.calls[0]["status"] == "failed"
            assert provider.calls[0]["request_dispatched"] is False
            assert received == ["/metrics"]
    asyncio.run(run())


def test_stream_interruption_keeps_partial_content_and_never_passes():
    case = item()
    class Interrupted(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("controlled stream interruption")
    def transport(request):
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:prefix_cache_hits_total 0\nvllm:prefix_cache_queries_total 0\nvllm:num_preemptions_total 0\n")
        return httpx.Response(200, stream=Interrupted())
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            provider = module.StreamingProvider(client, Tokenizer(), {"max_tokens": 1700,
                "temperature": 0, "seed": 42}, case, base_url="http://test.invalid",
                model="test", layout="original")
            with pytest.raises(httpx.ReadError):
                await provider.ainvoke([(m["role"], m["content"]) for m in case["messages"]])
            assert provider.calls[0]["status"] == "failed"
            assert provider.calls[0]["content"] == "partial"
            assert provider.calls[0]["request_dispatched"] is True
            assert provider.calls[0]["remote_outcome"] == "unknown"
            assert provider.calls[0]["read_timeout_seconds"] == 35
            assert provider.calls[0]["request_deadline_seconds"] is None
            assert provider.calls[0]["task_deadline_seconds"] == 75
            assert not provider.calls[0].get("complete_input_verified", False)
    asyncio.run(run())


def qualification_files(tmp_path, *, accepted=True):
    """模拟独立审阅文件只验证准入机制；不能生成真实模型资格。"""
    def freeze(path, value):
        value["sha256"] = module.baseline.digest(value)
        module.baseline.save(path, value)
        return value
    workload = freeze(tmp_path / "workload.json", {"cases": [item()], "model_candidate": "test-model",
        "model_revision": "fixed-revision", "max_tokens": 1700})
    manifest = freeze(tmp_path / "manifest.json", {"workload_sha256": workload["sha256"], "source_files_sha256": {}})
    identity = module.qualification_identity(workload, manifest, "json_object", "xgrammar")
    engine = module.engine_measurement_identity(engine_payload(), "0.11.2", "xgrammar")
    rows = [{"case": "N", "label": "qualification", "layout": "original", "repetition": 1,
        "machine_pass": accepted, "measurement_valid": True, "missing_required_facts": [],
        "result": {"status": "answered"}, "model_calls": 1, "qualification_identity": identity,
        "engine_identity": engine, "calls": [{"status": "returned", "complete_input_verified": True, "finish_reason": "stop"}]}]
    results = tmp_path / "results.jsonl"
    results.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    freeze(tmp_path / "review.json", {"results_sha256": module.baseline.file_digest(results),
        "reviewer": "unit fixture", "method": "controlled gate fixture, not actual semantic evaluation",
        "cases": [{"case": "N", "accepted": accepted, "reason": "controlled test result"}]})
    return workload, manifest, identity, engine


def test_factorial_requires_qualification_before_tokenizer_or_network(tmp_path, monkeypatch):
    qualification_files(tmp_path)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
        lambda *args, **kwargs: pytest.fail("Tokenizer must not be loaded without qualification"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: pytest.fail("No network without qualification"))
    args = SimpleNamespace(workload=tmp_path / "workload.json", manifest=tmp_path / "manifest.json",
        label="D", decoding="json_object", structured_backend="xgrammar", qualification=None)
    with pytest.raises(ValueError, match="requires a passed qualification"):
        asyncio.run(module.benchmark(args))


@pytest.mark.parametrize("mode", ["legacy", "references"])
def test_benchmark_token_precheck_consumes_frozen_response_mode(tmp_path, monkeypatch, mode):
    import transformers
    case = item()
    expected_messages = module.prepared_messages(case, "original", mode)
    workload = {"cases": [case], "model_candidate": "test-model", "model_revision": "fixed",
                "max_tokens": 1700, "response_mode": mode}
    workload["sha256"] = module.baseline.digest(workload)
    manifest = {"workload_sha256": workload["sha256"], "response_mode": mode,
                "source_files_sha256": {}, "tokenizer_files_sha256": {},
                "layouts": {"original": {"requests": [{"id": "N", "input_tokens": 2,
                                  "token_ids_sha256": module.baseline.digest([1, 2])}]}}}
    manifest["sha256"] = module.baseline.digest(manifest)
    module.baseline.save(tmp_path / "workload.json", workload)
    module.baseline.save(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())

    def tokenize(tokenizer, messages, **kwargs):
        assert messages == expected_messages
        return [1, 2]

    def stop_before_network(*args, **kwargs):
        raise RuntimeError("token_precheck_passed_no_network")

    monkeypatch.setattr(module.baseline, "chat_token_ids", tokenize)
    monkeypatch.setattr(httpx, "AsyncClient", stop_before_network)
    args = SimpleNamespace(workload=tmp_path / "workload.json", manifest=tmp_path / "manifest.json",
        model_directory=tmp_path, output=tmp_path / "never-created.jsonl", label="qualification",
        decoding="json_object", structured_backend="xgrammar")
    with pytest.raises(RuntimeError, match="token_precheck_passed_no_network"):
        asyncio.run(module.benchmark(args))
    assert not args.output.exists()


def test_passed_qualification_consumes_original_results_and_review(tmp_path):
    workload, manifest, identity, engine = qualification_files(tmp_path)
    output = tmp_path / "qualification.json"
    module.seal_qualification(tmp_path / "workload.json", tmp_path / "manifest.json",
        tmp_path / "results.jsonl", tmp_path / "review.json", output, "json_object", "xgrammar")
    record = module.load_qualification(output, identity)
    assert record["engine_identity"] == engine and record["cases"] == ["N"]
    (tmp_path / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence identity"):
        module.load_qualification(output, identity)


def test_failed_qualification_cannot_be_sealed(tmp_path):
    qualification_files(tmp_path, accepted=False)
    output = tmp_path / "qualification.json"
    with pytest.raises(ValueError, match="qualification.*failed"):
        module.seal_qualification(tmp_path / "workload.json", tmp_path / "manifest.json",
            tmp_path / "results.jsonl", tmp_path / "review.json", output, "json_object", "xgrammar")
    assert not output.exists()


def test_semantic_pass_does_not_override_machine_failure(tmp_path):
    qualification_files(tmp_path)
    path = tmp_path / "results.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["machine_pass"] = False
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    review_path = tmp_path / "review.json"
    review = module.baseline.load_workload(review_path)
    del review["sha256"]
    review["results_sha256"] = module.baseline.file_digest(path)
    review["sha256"] = module.baseline.digest(review)
    review_path.write_text(json.dumps(review), encoding="utf-8")
    output = tmp_path / "qualification.json"
    with pytest.raises(ValueError, match="Machine qualification"):
        module.seal_qualification(tmp_path / "workload.json", tmp_path / "manifest.json",
            path, review_path, output, "json_object", "xgrammar")
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("model", "another-model"), ("revision", "new-revision"),
    ("manifest_sha256", "new-code-or-template"), ("structured_backend", "guidance"),
    ("decoding", "json_schema"), ("prompt_version", "old-workflow"), ("response_mode", "references")])
def test_qualification_cannot_be_reused_under_changed_configuration(tmp_path, field, value):
    _, _, identity, _ = qualification_files(tmp_path)
    output = tmp_path / "qualification.json"
    module.seal_qualification(tmp_path / "workload.json", tmp_path / "manifest.json",
        tmp_path / "results.jsonl", tmp_path / "review.json", output, "json_object", "xgrammar")
    with pytest.raises(ValueError, match="configuration differs"):
        module.load_qualification(output, {**identity, field: value})


def engine_payload(backend="xgrammar", cache=True):
    return {"vllm_config": {"structured_outputs_config": {"backend": backend},
        "model_config": {"dtype": "bfloat16", "quantization": "fp8", "max_model_len": 32768},
        "cache_config": {"cache_dtype": "auto", "block_size": 16, "enable_prefix_caching": cache},
        "scheduler_config": {"max_num_seqs": 4, "max_num_batched_tokens": 8192, "enable_chunked_prefill": True}}}


def test_engine_backend_is_observed_not_inferred_from_auto():
    with pytest.raises(ValueError, match="pinned backend differs"):
        module.engine_measurement_identity(engine_payload("auto"), "0.11.2", "xgrammar")
    first = module.engine_measurement_identity(engine_payload(cache=True), "0.11.2", "xgrammar")
    second = module.engine_measurement_identity(engine_payload(cache=False), "0.11.2", "xgrammar")
    assert first == second
    changed = engine_payload()
    changed["vllm_config"]["scheduler_config"]["max_num_seqs"] = 2
    assert first != module.engine_measurement_identity(changed, "0.11.2", "xgrammar")


def test_incompatible_schema_is_rejected_before_auto_fallback():
    with pytest.raises(ValueError, match="requires guidance"):
        module.qualification_identity({"cases": [item()]}, {}, "json_schema", "xgrammar")


@pytest.mark.parametrize("layout", ["original", "evidence_first"])
def test_reference_replay_consumes_real_workflow_and_keeps_external_fact_gate(monkeypatch, layout):
    monkeypatch.setattr(workflow, "prepare_stage_requirements", lambda evidence, docs, question: (evidence, docs))
    case = item()
    case["required_facts"] += ["c0.not_in_runtime.mae"]
    provider = Provider([
        {"status": "answered", "body": {"kind": "plain", "text": "MAE为12.5。"}},
        {"status": "answered", "body": {"kind": "plain", "text": "MAE为{{c0.ridge.mae}}。"}}])
    row = asyncio.run(module.replay_case(provider, case, layout, response_mode="references"))
    assert row["result"]["status"] == "answered" and row["model_calls"] == 2
    assert row["machine_pass"] is False and row["missing_required_facts"] == ["c0.not_in_runtime.mae"]
    assert row["semantic_review"] == "pending" and row["response_mode"] == "references"
    payload = json.loads(provider.calls[0]["messages"][1][1])
    assert "required_facts" not in payload and "c0.not_in_runtime.mae" not in json.dumps(provider.calls)
    assert module.json_messages(provider.calls[0]["messages"]) == module.prepared_messages(case, layout, "references")


def test_reference_freeze_preserves_all_evidence_and_original_evaluation(tmp_path):
    case = item()
    case["semantic_rubric"] = "independent review only"
    value = {"cases": [case], "model_candidate": "original-model", "model_revision": "original-revision",
             "prompt_version": "historical-capture", "sampling": {"top_p": 0.8}, "max_tokens": 1700}
    value["sha256"] = module.baseline.digest(value)
    source, output = tmp_path / "original.json", tmp_path / "references.json"
    module.baseline.save(source, value)
    before = source.read_bytes()
    module.rebind_reference_workload(source, output)
    frozen = module.baseline.load_workload(output)
    current = frozen["cases"][0]
    assert source.read_bytes() == before and frozen["parent_workload_sha256"] == value["sha256"]
    assert frozen["sha256"] != value["sha256"] and frozen["response_mode"] == "references"
    assert frozen["original_capture_metadata"]["prompt_version"] == "historical-capture"
    for key in ("question", "evidence", "documents", "required_facts", "semantic_rubric"):
        assert current[key] == case[key]
    rebound_payload = json.loads(current["messages"][1]["content"])
    rebound_payload.pop("output_schema")
    assert rebound_payload == json.loads(case["messages"][1]["content"])
    assert current["messages"][0] != case["messages"][0]
    assert frozen["sampling"] == value["sampling"] and frozen["max_tokens"] == value["max_tokens"]
    with pytest.raises(FileExistsError):
        module.rebind_reference_workload(source, output)


def test_reference_freeze_rejects_changed_payload_instead_of_silent_reassembly(tmp_path):
    case = item()
    case["documents"] = [{"id": "changed-evidence", "text": "not the original input"}]
    value = {"cases": [case]}
    value["sha256"] = module.baseline.digest(value)
    source, output = tmp_path / "original.json", tmp_path / "references.json"
    module.baseline.save(source, value)
    with pytest.raises(ValueError, match="changed full user evidence"):
        module.rebind_reference_workload(source, output)
    assert not output.exists()


def test_reference_identity_is_distinct_and_cannot_reuse_legacy_manifest():
    workload = {"sha256": "w", "model_candidate": "model", "model_revision": "revision",
                "max_tokens": 1700, "response_mode": "references"}
    with pytest.raises(ValueError, match="Response mode differs"):
        module.qualification_identity(workload, {"sha256": "m"}, "json_object", "xgrammar")
    identity = module.qualification_identity(workload, {"sha256": "m", "response_mode": "references"},
                                             "json_object", "xgrammar")
    assert identity["response_mode"] == "references" and identity["prompt_version"] == workflow.REFERENCE_PROMPT_VERSION


def test_reference_schema_only_constrains_sources_not_answers():
    case = item()
    case["documents"] = [{"id": "doc", "source_sha256": "a" * 64}]
    schema = module.contract_schema(case, "references")
    assert set(schema["properties"]) == {"status", "body"}
    assert schema["$defs"]["CitationSelection"]["properties"]["document_id"]["enum"] == ["doc"]
    assert set(schema["$defs"]["CitationSelection"]["properties"]) == {"document_id"}
    assert schema["$defs"]["CitationSelection"]["additionalProperties"] is False
    assert "enum" not in schema["$defs"]["PlainReferenceBody"]["properties"]["text"]


def time_case():
    from power_forecast_service.assistant.stages import stage_sources
    case = item()
    case["question"] = "13:20起报用到多晚的输入？预测哪些时距？延迟是实测吗？"
    source = stage_sources()["engie"]
    case["evidence"]["records"][0].update(id=str(UUID(int=1)), scope="engie_final",
        result_sha256=source["result_sha256"], protocol_sha256=source["protocol_sha256"])
    case["semantic_rubric"] = "EXTERNAL_RUBRIC_SENTINEL"
    return case


def test_explicit_derived_refresh_preserves_original_facts_and_external_acceptance(tmp_path):
    case = time_case()
    value = {"cases": [case], "response_mode": "references", "model_candidate": "same-model",
             "model_revision": "same-revision", "max_tokens": 1700, "sampling": {"top_p": 0.8}}
    value["sha256"] = module.baseline.digest(value)
    source, target = tmp_path / "previous.json", tmp_path / "current.json"
    module.baseline.save(source, value)
    before = source.read_bytes()
    receipt = module.refresh_derived_workload(source, target)
    current = module.baseline.load_workload(target)
    assert source.read_bytes() == before and receipt["parent_workload_sha256"] == value["sha256"]
    updated = current["cases"][0]
    for key in ("question", "required_facts", "semantic_rubric"):
        assert updated[key] == case[key]
    assert all(f in updated["evidence"]["facts"] for f in case["evidence"]["facts"])
    assert "EXTERNAL_RUBRIC_SENTINEL" not in json.dumps(updated["messages"])
    assert len(receipt["changes"][0]["added_fact_ids"]) == 5
    assert module.prepared_messages(updated, "original", "references") == updated["messages"]
    with pytest.raises(FileExistsError):
        module.refresh_derived_workload(source, target)


@pytest.mark.parametrize("layout", ["original", "evidence_first"])
def test_temporal_repair_consumes_same_derived_input_as_measurement(layout):
    case = time_case()
    data, _ = workflow.prepare_answer_context(case["evidence"], [], case["question"], "references")
    text = "{{c0.ridge.mae}}；" + "；".join("{{" + f + "}}" for f in data["temporal_requirements"])
    provider = Provider([{"status": "answered", "body": {"kind": "plain", "text": "{{c0.ridge.mae}}"}},
                         {"status": "answered", "body": {"kind": "plain", "text": text}}])
    row = asyncio.run(module.replay_case(provider, case, layout, response_mode="references"))
    assert row["machine_pass"] and row["model_calls"] == 2
    assert row["trace"]["repair_reason"] == "answer_time_fact_missing"
    assert "13:00" in row["result"]["answer"] and "14:20" in row["result"]["answer"]
    assert module.json_messages(provider.calls[0]["messages"]) == module.prepared_messages(case, layout, "references")
    assert "EXTERNAL_RUBRIC_SENTINEL" not in json.dumps(provider.calls)


def test_reference_xgrammar_is_separate_from_unsupported_legacy_quotes_schema():
    workload = {"sha256": "w", "model_candidate": "m", "model_revision": "r", "max_tokens": 1700,
                "response_mode": "references", "cases": [item()]}
    identity = module.qualification_identity(workload, {"sha256": "p", "response_mode": "references"}, "json_schema", "xgrammar")
    assert identity["decoder_schemas_sha256"] == module.baseline.digest([module.contract_schema(item(), "references")])
    with pytest.raises(ValueError, match="requires guidance"):
        module.qualification_identity({**workload, "response_mode": "legacy"}, {"sha256": "p"}, "json_schema", "xgrammar")


def test_request_specific_schema_does_not_change_system_prefix():
    first, second = item(), time_case()
    a = module.prepared_messages(first, "evidence_first", "references")
    b = module.prepared_messages(second, "evidence_first", "references")
    assert a[0] == b[0]
    assert list(json.loads(a[1]["content"]))[-1] == "output_schema"
    assert json.loads(a[1]["content"])["output_schema"] == module.contract_schema(first, "references")
