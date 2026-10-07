# 测试与副作用

产品测试随源码保留，按实际输入分组，不因缺真实依赖自动切换mock。维护者的导出工具、
精选清单和模板导航检查仅在维护仓执行，不属于产品发行包；公开根README不依赖维护模板。

| 入口 | 实际验证 | 必需输入及副作用 |
| --- | --- | --- |
| `python -m pytest tests/unit` | 局部校验与规则 | 已安装Python依赖；不发provider请求；PowerShell入口测试需pwsh |
| `tests/integration/postgres/test_inference_entrypoint.py` | 只读迁移/模型目录检查、旧迁移与缺表拒绝、真实CLI/TCP生命周期 | 随机PG库；注入模型目录、不生成回答；只结束自有API |
| `pwsh -File tools/dev/check-service.ps1` | unit、真实PG/模型组件与三个HTTP实验流程 | 已启动独立Compose；随机测试数据库；e2e实际训练并登记新任务 |
| 同命令加 `-OriginalReplay` | 九个2014原包及五个2015原包的真实PG导入/回放/故障边界 | 显式准备两个附件；未设置路径直接失败；不训练 |
| `npm test`，默认software组 | 前端坐标、新Ridge对象与导航 | `npm ci`、Playwright浏览器、上述e2e生成的ui-delivery.json |
| `WIND_UI_SUITE=original` | 原ENGIE页面及注入前端故障 | 两套原包已导入，不发provider |
| `WIND_UI_SUITE=provider` | 真实结果问答与页面引文 | 原最终包已导入、已配置密钥；会收费，显式选择 |
| `WIND_UI_SUITE=all` | 上述及完整交互任务测试 | 会提交额外训练、模型请求，不能作为默认离线检查 |

原附件准备目录分别设置到 `WIND_ENGIE_DEVELOPMENT_SOURCE` 和 `WIND_ENGIE_FINAL_SOURCE`。
容器入口会将它们挂到独立只读目录，消费者不去维护者的历史`.local`找模型。
原模型joblib只可加载通过源码信任锚校验的附件，不能运行来源不明的模型。

固定模型延迟监测的`tests/integration/postgres/test_engie_monitor.py`归入`original_replay`，
使用最终附件、随机PG库；其中实际TCP检查启动/停止自己的API进程和独立CLI，不接管
运行服务。离线时间/计数规则在`tests/unit/test_engie_monitor.py`。具体消费与限制见
[延迟监测](engie-monitor.md)。不将该本地原模型组算入默认远端CI成绩。

浏览器默认Playwright Chromium；Windows可显式设置 `WIND_BROWSER_CHANNEL=msedge`。
独立端口设置 `WIND_TEST_BASE_URL`，浏览器输出全部写`.local/runtime`，不覆盖原历史报告。
本机Python锁包含Windows CUDA开发依赖；推荐公开运行使用固定CPU Docker路径。
CPU依赖/测试依赖下载需要网络，“不发模型请求”不等于“不联网安装”。

历史研究/负载执行脚本保留作来源与纯函数回归；它们需要维护者的冻结输入和工作包，
不属于普通启动。`run_w1_load.py`明确要求 `--checker`，不再猜测兄弟Agent仓库。
公共CI默认不发provider请求；原附件和付费路径由显式本地命令验证，远端CI须发布后另核验。

## CI实际覆盖与缺口

[verify.yml](../.github/workflows/verify.yml) 使用独立 CPU Compose，构建静态资源后运行软件组：
局部规则、真实 PostgreSQL/模型组件和两个显式 Q1 开发实验（默认组合、Ridge），再检查
实验回执驱动的浏览器。第三个候选回放实验属于默认本地检查的完整 e2e 组，当前远端
workflow 只选择 `test_experiment_flow.py`，没有选择 `test_candidate_delivery.py`。
结束只退出 `wind-ci` 自有服务和卷。默认 CI 不载入原 ENGIE 附件、不准备向量模型，也不
发送真实 provider；对应原模型回放和付费问答由上表显式入口覆盖。结构化回答故障、合法
阶段回答和可选捕获失败使用注入 provider，不能替代真实模型质量或远端执行成绩。

CPU容器没有 `pwsh`，其中八项PowerShell配置/启动边界检查会跳过。独立
`windows-entrypoints` job 使用Python 3.12和固定pytest 8.4.2，只运行
`test_runtime_context.py -k "not direct_diagnostic"` 的八项；先要求pwsh存在，不同步
项目uv锁，不下载CUDA或应用依赖，不启动真实Docker。该文件另外两项Python诊断会
导入psycopg，继续由CPU组件组覆盖。Windows路径使用合成配置，不证明真实服务启动。

本页说明工作流配置与本地组件验证范围；具体提交的远端成绩以对应Actions运行记录为准，
不将配置本身当作执行通过。一次原拒答及后续诊断取舍见[困难案例](public-assistant-case.md)。

## 可选推理覆盖

上述service job选择所有unit/integration，因此也会收集推理模块、provider协议、
请求登记和可选入口测试；不需要额外下载vLLM或GPU模型。真实PG组包含迁移head/
请求表检查与CLI启动，provider响应为显式注入，不能算真实14B质量或性能。
本地导出候选另验证其自身src和测量器依赖，不从维护仓补文件；对应结果见
[I4](results/inference-i4-20261007/README.md)。未发布新提交前，没有这版远端CI成绩。

[可选后端说明](inference-backend.md)中的`inference_service.py`才是显式真实模型
故障入口，会读现有导入、创建/删除随机库、强杀自己的API并留下私有回答/日志。
默认pytest仅测试其纯验证规则，不运行这套演练。历史baseline capture依赖未公开
题源，其他测量命令需要冻结输入，不宣称发行包可直接重跑历史六题性能。
