# 查询规划微调接入服务回答链路：8B LoRA 端到端 78% vs 原始 8B 19%

2026-10-17，承接[首答规划评测](../sft-obligation-8b-20261017/README.md)。评测脚本、逐题记录与原始返回
保存在未公开的工作包中；本目录公开汇总 `summary.json`。

## 结论

把 Qwen3-8B + LoRA（与首答评测同一适配器）接到结果助手的真实回答节点，与未微调的 Qwen3-8B
在同一 vLLM 服务、同一批题上对照。每题都走
schema → HTTP/SSE vLLM → 解析 → `execute_program` → 全部服务校验 → 至多一次修复调用。

| 指标 | LoRA（字母序 schema） | 原始 8B（默认序） | LoRA（默认序） |
|---|---:|---:|---:|
| test300 服务交付（校验通过） | **300/300** | 277/300 | 296/300 |
| test300 交付且评测器通过（auto） | **233/300 = 77.7%** CI 72.6–82.0 | 57/300 = 19.0% CI 15.0–23.8 | 150/300 = 50.0% |
| test300 交付且 strict 通过 | 213（71.0%） | 30（10.0%） | 135（45.0%） |
| test300 交付且宽松口径通过 | 295（98.3%） | 103（34.3%） | 197（65.7%） |
| dev32 交付 | 32/32 | 32/32 | 32/32 |
| dev32 交付且 auto 通过 | **30/32** CI 79.9–98.3 | 9/32 | 16/32 |
| 每题模型调用（test300 均值） | 1.003 | 1.077 | 1.02 |
| 修复调用后交付（test300） | 1/1 | 5/23 | 5/6 |
| 端到端延迟 p50 / p95（test300，并发 8） | 15.9 / 24.3 s | 16.6 / 32.0 s | 14.2 / 39.9 s |

- 配对 McNemar（交付且 auto 通过）：LoRA vs 原始，test300 177:1，dev32 21:0（p<1e-5）；
  字母序 vs 默认序（同一适配器），test300 83:0，dev32 14:0。
- 服务内首答与离线批量评测一致：LoRA 首答 auto 232/300，离线记录也是 232/300；
  300 题评测判定逐题相同，297 题首答 JSON 完全相同（3 题不同，原因未定位；离线为单进程大批量，
  服务为并发 8 的在线批处理）。服务链路没有让规划质量变差。LoRA 的修复调用在 test300、dev32 各救回 1 题；
  原始 8B 有 23 题触发修复、5 题修复后交付，但这 5 题都没通过评测器（auto 首答与最终都是 57）。
- 原始 8B 的服务内首答与离线记录差得更多：test300 auto 57 对 55，dev32 9 对 12，
  首答 JSON 只有 231/300、26/32 相同。原始模型输出对批处理差异更敏感，比较时以同一服务内的配对结果为准。
- 服务校验器不等于评测器：LoRA 有 67 题已交付但 auto 不通过（原始 8B 为 220 题）。
  没有出现评测器判通过、服务却拒绝的情况。
- 测试集里没有题走确定性编译器路由；4 次运行共 1328 题次全部是模型路由，表中都是模型调用的结果。

## LoRA 没通过的题

LoRA 的 67 题失败里有 62 题只有 `explicit_request_order_mismatch` 一种失败。查下来不是顺序错了，
而是缺口措辞：题目写"真实到达延迟P95/P99"，gold 写"实际生产延迟…"，模型写"实际到达延迟…"
（另有"电力交易收入"对"售电收入"）。严格检查器把缺口文本当作顺序签名的一部分，
所以记为顺序不符。宽松口径（只比语义原子序列）把这类都算通过（295/300）。
真正的语义失败约 4 题：漏掉"两项是否同时更低"的比较汇总（2 题），
以及把比较写成读数（2 题）。这些计划都通过了服务校验并已交付，说明当前服务校验
不检查"是否回答了所问的关系"，只检查绑定、覆盖与格式。

原始 8B 的失败大多是义务不全或多余（`operation_obligations_differ` 140、`presentation_obligations_differ` 67）、
阶段角色和引用不符；另有 18 题被服务拒绝（9 题 `answer_program_stage`），
5 题在 60 s provider 预算内没有返回（`provider_result_unknown`，按合同不重发）。

## 做法

1. **代码改动**（默认关闭）：[`structured_output.vllm_schema(schema, sort_properties=False)`](../../../src/power_forecast_service/assistant/structured_output.py)，
   只把 schema 位置上的 `properties` 按键名排序，`enum`/`default` 等数据不动；
   [`VLLMProvider(..., sort_schema_properties=False)`](../../../src/power_forecast_service/assistant/provider.py) 透传，开启时在 `identity` 和每次返回的
   `response_metadata` 记 `schema_property_order: sorted-properties-v1`，默认身份逐字不变；
   `make_assistant` 读 `inference_sort_schema_properties`；[`tools/dev/serve_inference.py`](../../../tools/dev/serve_inference.py) 新增
   `--sort-schema-properties`（用法见[可选 vLLM 后端](../../inference-backend.md)）。业务验收仍用原 schema（provider 内 `Draft202012Validator(schema)`）。
   新增单元测试：默认字节不变、只重排 schema 位置、语言等价、与有限数组展开组合、
   provider 发送顺序与身份记录、CLI 透传（`tests/unit/assistant/test_structured_output.py`、
   `test_provider.py`、`tests/unit/test_inference_entrypoint.py`）。
2. **评测**（未公开的评测脚本）：按评测器 `builder.service_input` 的方式取
   payload 的 evidence/documents/question，构造
   `Assistant(None, provider=VLLMProvider(url, model, generation_profile="offline-greedy-v1", sort_schema_properties=...), response_mode="program_requested")`，
   直接调用 `answer` 节点。答复节点本身不用数据库；`sessions=None` 只跳过 `evidence` 节点（取证据、检索）
   和 `run()` 末尾的审计写入。用服务的 `capture_draft` 钩子记录每稿计划，用一个只旁路记录的 provider 包装
   保存实际发出的消息/schema 摘要和原始返回文本。每题外层套服务同样的 75 s 总预算。
   全部 332 题的首个请求与训练/离线评测行的 messages 和 schema 逐字相同。
3. **评分**：最终计划与首答原文都用 `supervision_evaluator_v2.evaluate_plan`（研究用监督检查器，未随公开源码分发）；
   比例附 Wilson 95% 区间，配对 McNemar 精确检验（复用 [`score.py`](../../../tools/research/planner_sft/score.py)）。
4. **HTTP 全链路**：另起一次性 PostgreSQL 容器（tmpfs，端口 5440），`alembic upgrade head`，
   用 `tools/dev/import_engie.py --release final-2015` 导入 ENGIE 正式结果（import id 与测试行相同），
   再用 `serve_inference.py --sort-schema-properties` 起服务问 3 题（dev32 第一题、test300 第一个二义务题、
   第一个三义务题，按文件顺序取）。这一步走数据库取证据、关键词检索、请求登记、审计写入。
   3/3 `answered`，交付事实 ID 与 gold 执行结果相同，请求登记 `completed/answered`，每题 1 次调用。
   用完删除容器与临时密钥、导入副本；现有数据库和容器未触碰。

## 身份

- 底座：Qwen3-8B（`config.json` sha256 `f7c4eadf…`）；适配器：`adapter_model.safetensors` sha256 `e6c50124…`
  （182 MB，未上传）。
- 引擎：vLLM 0.11.2、torch 2.9.0、xgrammar 0.1.25，RTX PRO 6000 Blackwell 96 GB，bf16，
  `--max-model-len 32768 --max-num-seqs 8 --enable-prefix-caching --enable-lora --max-lora-rank 64`，
  一个服务同时提供 `qwen3-8b` 与 `wind-planner`。
- 采样：`offline-greedy-v1`（temperature 0、top_p 1、top_k −1、max_tokens 1024、seed 42、非思考模式）。
  客户端并发 8，provider 超时 60 s。
- 评测时开发仓库有未提交改动；本次公开的助手源码与评测所用版本相同。完整摘要见 `summary.json`。

## 局限

- **语料版本**：助手的 `corpus.json` 与 `stage_sources.json` 在测试集生成后被改写过
  （文档 ID 变化，例如 `a3-8f352bd1dee97182` → `a3-d818e5a8b0e2945c`），按当前文件构造的阶段证据会报
  `corpus_stage_evidence_missing`。离线评测因此把这两份数据固定为生成测试集时的版本
  （sha256 见 `summary.json` 的 `frozen_*`）；代码都用当前版本。
  HTTP 三题用的是当前语料（没有打补丁），检索到的文档 ID 是新 ID。
- 离线主评测不含数据库取证据与检索：证据和文档直接来自测试行。HTTP 全链路只跑了 3 题，
  只证明链路能通，不提供比例。
- 单一底座、单一适配器、单一服务配置。test300 与训练数据同一生成器，dev32 同源证据对象、只有 32 题
  （同[上一轮](../sft-obligation-8b-20261017/README.md)）。
- 延迟在客户端并发 8 下测量，含排队；p50 约 16 s 主要是约 23k token 的 prefill，不代表单请求延迟。
- 有一题（`te-00055`）在两次 LoRA 字母序运行间判定翻转（5/332 题计划不同，原因未定位）；
  重复运行 test300 auto 为 233 和 234，表中报第二次（与另外两臂同一脚本版本）。
- 服务校验通过不等于回答正确：LoRA 已交付的 300 题里，评测器判出约 4 题语义错误。
- 没有改产品默认模型或默认传输顺序；`sort_schema_properties` 只能显式开启。
- 评测脚本、监督检查器与适配器未公开，外部读者不能直接重跑本表；可核对的是公开源码、单元测试与 `summary.json`。

## 复现（需要自备适配器与评测脚本）

```bash
# GPU 实例
python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-8B --served-model-name qwen3-8b \
  --host 127.0.0.1 --port 18110 --dtype bfloat16 --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 --structured-outputs-config '{"backend":"xgrammar"}' \
  --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules wind-planner=/path/to/adapter \
  --seed 0 --disable-log-requests --enable-prefix-caching
# 本地（隧道到 127.0.0.1:18110），在未公开的评测工作包内
python run_service_eval.py --model wind-planner --name lora_sorted
python run_service_eval.py --model qwen3-8b --name base_default --no-sort
python run_service_eval.py --model wind-planner --name lora_unsorted --no-sort
```

服务端 HTTP 入口（带数据库、检索与请求登记）：

```powershell
pwsh -File tools/dev/run-local.ps1 -Command @(
  '.venv/Scripts/python.exe', '-B', 'tools/dev/serve_inference.py',
  '--provider-url', 'http://127.0.0.1:18110/v1', '--model', 'wind-planner', '--port', '18113',
  '--response-mode', 'program_requested', '--generation-profile', 'offline-greedy-v1',
  '--provider-timeout', '60', '--sort-schema-properties'
)
```
