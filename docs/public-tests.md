# 测试与副作用

所有测试随源码保留，按实际输入分组，不因缺真实依赖自动切换mock。

| 入口 | 实际验证 | 必需输入及副作用 |
| --- | --- | --- |
| `python -m pytest tests/unit` | 局部校验与规则 | 已安装Python依赖；不发provider请求；PowerShell入口测试需pwsh |
| `pwsh -File tools/dev/check-service.ps1` | unit、真实PG/模型组件与三个HTTP实验流程 | 已启动独立Compose；随机测试数据库；e2e实际训练并登记新任务 |
| 同命令加 `-OriginalReplay` | 九个2014原包及五个2015原包的真实PG导入/回放/故障边界 | 显式准备两个附件；未设置路径直接失败；不训练 |
| `npm test`，默认software组 | 前端坐标、新Ridge对象与导航 | `npm ci`、Playwright浏览器、上述e2e生成的ui-delivery.json |
| `WIND_UI_SUITE=original` | 原ENGIE页面及注入前端故障 | 两套原包已导入，不发provider |
| `WIND_UI_SUITE=provider` | 真实结果问答与页面引文 | 原最终包已导入、已配置密钥；会收费，显式选择 |
| `WIND_UI_SUITE=all` | 上述及完整交互任务测试 | 会提交额外训练、模型请求，不能作为默认离线检查 |

原附件准备目录分别设置到 `WIND_ENGIE_DEVELOPMENT_SOURCE` 和 `WIND_ENGIE_FINAL_SOURCE`。
容器入口会将它们挂到独立只读目录，消费者不去维护者的历史`.local`找模型。
原模型joblib只可加载通过源码信任锚校验的附件，不能运行来源不明的模型。

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

目前只有本地对应命令/组件证据；尚未发布仓库、触发远端 workflow，因此不写“GitHub CI
已通过”。一次原拒答及后续诊断取舍见[困难案例](public-assistant-case.md)。
