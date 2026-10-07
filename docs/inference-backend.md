# 可选 vLLM 结果助手

这是结果助手的可选推理路径，不替换预测模型、训练 worker 或普通 CPU 服务。
默认仍为原 provider；只有显式运行下面的宿主入口才启用 vLLM。已验证的是单机
服务集成、质量约束下的缓存对照和请求恢复边界，不是自研推理内核或生产集群。
这是首发v0.1.0之后的维护候选；旧首发版本没有此入口，使用前核对实际源码版本。

## 1. 独立数据库与 Python 环境

先按[独立复现](public-reproduction.md)第1节创建新 Compose 项目和端口，执行
`start-app.ps1 -Build`；新库会运行当前迁移。不要为了试用而对已有业务库自动升级。
需解释 ENGIE 结果时，按第2、3节校验附件并导入原模型记录。空库没有可解释的实验。

可选宿主 API 使用本仓库自己的 Python 3.12 开发环境；从仓库根目录执行
`uv sync --locked`。该锁包含 Windows CUDA 开发依赖，安装体积较大；不能把这里
称作“无需宿主 Python 的 CPU Docker 入口”。不要向这个环境安装 vLLM。
Linux GPU 引擎独立运行；本轮交付没有新建 GPU 环境，复用了原已验证环境。

宿主 API 此处只用于结果问答、健康和请求状态。容器训练工件位于 Docker 卷，
不能假定宿主能够直接读取；训练、模型回放与发布仍使用普通 CPU 服务。
ENGIE 问答读取已导入的 PG 指标/身份，不在问答中加载模型或拟合。

## 2. Linux GPU 引擎

已测环境：单卡48GB、vLLM 0.11.2、Torch 2.9.0+cu128、XGrammar 0.1.25；
模型为 `Qwen/Qwen3-14B-FP8`，revision
`9a283b4a5efbc09ce247e0ae5b02b744739e525a`。FP8是权重格式，运行/KV使用BF16。
完整已见请求约22k至26k token，不能通过删证据或压短问题伪装适配。
以下命令以已经准备好的独立 GPU 环境及固定版本模型目录为前提，不是在任意
GPU上测试过的一键安装器；文件和依赖版本仍需自行核对，不能仅凭目录名信任模型。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/pinned-Qwen3-14B-FP8 \
  --served-model-name wind-qwen14b \
  --host 127.0.0.1 --port 18110 \
  --dtype bfloat16 --kv-cache-dtype auto \
  --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization 0.85 \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --disable-log-requests --enable-prefix-caching
```

引擎不直接暴露公网。在宿主另一个终端建立自己的 SSH 隧道，使用个人认证文件，
不要把密码、主机凭据或模型权重提交到仓库：

```powershell
ssh -N -L 127.0.0.1:18110:127.0.0.1:18110 -p <SSH端口> <用户>@<主机>
```

此路径的模型身份检查只核对 `/v1/models` 的 served name 和上下文长度；
它不证明权重hash、APC已开启或答案正确。APC作用于prefill，不能宣称加速decode。
引擎环境、固定revision和完整请求需分别核验。

## 3. 只读检查，然后显式启动

在同一仓库根目录、沿用新 Compose 项目的环境变量执行：

```powershell
pwsh -File tools/dev/run-local.ps1 -Command @(
  '.venv/Scripts/python.exe', '-B', 'tools/dev/serve_inference.py',
  '--provider-url', 'http://127.0.0.1:18110/v1',
  '--model', 'wind-qwen14b', '--port', '18113', '--capacity', '1', '--check'
)
```

`--check`只读取有效Compose连接配置、当前PG迁移head/请求表、模型目录，并检查
独立端口；不迁移、不训练、不发生成探针。端口不得与普通API、隧道或已有监听冲突。
检查通过后，去掉 `--check` 才真正启动，仍仅绑定 `127.0.0.1`。显式CLI端口不会
被 `run-local.ps1` 注入的普通API端口覆盖。按Ctrl+C关闭此API，再结束自己的隧道；
云GPU需要另行停机，不会随本机API自动关闭。

典型错误码：`inference_database_migration_required` 表示当前连接的数据库不是
源码迁移head或缺请求表；检查不会替你升级。`inference_model_identity_mismatch`
表示服务名不符；`inference_context_insufficient`表示未报告至少32768上下文。
失败不回显密码或远端错误正文，也不会静默回退到默认provider。

## 4. 一次问答与状态恢复

从普通服务的 `GET /engie/imports` 获取本次导入对象ID，不使用维护者的旧UUID。
按当前对象/模型填写请求，先在客户端保留自己的请求ID：

```powershell
$requestId = [guid]::NewGuid().ToString()
$body = @{
  question = '比较这个对象的 lightgbm_l1_shrink 与 persistence 的 MAE。'
  contexts = @(@{kind='engie_import'; id='<本次导入UUID>'; model='lightgbm_l1_shrink'})
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:18113/assistant/answers' `
  -Headers @{'X-Request-ID'=$requestId} -ContentType 'application/json; charset=utf-8' -Body $body
Invoke-RestMethod "http://127.0.0.1:18113/assistant/requests/$requestId"
```

POST会调用模型；完整输入/来源绑定、最多两次业务调用与75秒总预算保持。
数值、版本和引文检查不等于自然语言蕴含证明，仍需审阅正文，尤其开发采用门与
最终留出的关系。证据不足或校验失败不交付假成功。

`capacity`是单API进程上限（1至4），不是vLLM吞吐量或集群全局配额。客户端断开
不立刻释放仍在执行的槽位；满时返回429。已登记同ID返回409，不自动重复请求。
超时/丢失SSE结束标记/中断窗口可能成为 `unknown`；先查询原身份，不盲换ID重发。
PG只存输入指纹、调用状态及结果hash，**不保存答案用于重放**；用户需自行保留已收到的回答。

## 5. 验证和公开边界

[测试说明](public-tests.md)区分默认离线/真实PG与显式真实GPU检查。
`test_inference_entrypoint.py` 使用真实PG、TCP和宿主CLI，但模型目录由测试注入，
不证明真实14B质量。I3的[真实服务结果](results/inference-i3-20261007/README.md)
单独保留；I2的[配对缓存结果](results/inference-i2-20261006/README.md)请求合同不同，
其10.20%不能继承为当前HTTP服务提速。

`tools/diagnostics/inference_service.py`是显式故障入口，会创建/删除自有随机库并
强杀自己启动的API，读取已有导入记录并真实调用模型；不作为普通pytest自动故障演练。
应先确认连接的只读来源、端口空闲、费用和私有输出目录，再在自有环境显式调用：

```powershell
pwsh -File tools/dev/run-local.ps1 -Command @(
  '.venv/Scripts/python.exe', '-B', 'tools/diagnostics/inference_service.py',
  '--provider-url', 'http://127.0.0.1:18110/v1', '--model', 'wind-qwen14b',
  '--api-port', '18213', '--proxy-port', '18214', '--output', '.local/inference-drill-unique'
)
```

演练还使用api-port之后的三个独立端口（跳过proxy-port），所以上例需空闲18213至18217。
原始请求、回答、数据库快照和进程日志只写指定的私有输出；不得直接公开这个目录。
技术检查通过后仍要逐份语义复核。历史 `inference_baseline.py capture` 依赖未公开
开发题源，不能在发行包开箱使用；另外三个测量模块同样需要显式冻结输入，导出它们
是为了保留测试依赖和方法源码，不是宣称历史六题已经随源码完整重跑。
