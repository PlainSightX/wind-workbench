# 风电预测任务与结果分析工作台

一个可恢复的Python实验服务：显式提交训练任务，按模型版本回放历史预测，并由结果助手解释数值和选型依据。
数据是公开历史代理场景，项目没有生产客户或电网部署。

自有代码采用 [MIT](LICENSE)。仓库为
[PlainSightX/wind-workbench](https://github.com/PlainSightX/wind-workbench)，首发版本 `v0.1.0`。
原模型附件见版本 Release；各提交的远端检查结果以对应 Actions 运行记录为准。

## 开始使用

推荐入口是PowerShell 7 + Docker的CPU容器，无需宿主CUDA。
依照 [独立复现](docs/public-reproduction.md) 创建独立项目、执行样本实验及获取原模型附件。
正常启动、页面查询和预测不训练；只有显式提交实验才训练。

默认关键词结果助手仅需环境变量 `DEEPSEEK_API_KEY`：

```powershell
pwsh -File tools/dev/prepare-assistant.ps1
```

该命令保存配置，不发送模型请求、不下载embedding。随后在页面选择结果再提问。
可选向量路径使用 `-Vector` 下载固定embedding并建立真实PG索引；`-ProbeProvider`才额外发送探针请求。
密钥只在忽略提交的运行目录保存，不写入源码或报告。

已有独立GPU环境时，可显式启用[可选vLLM后端](docs/inference-backend.md)：
宿主API通过loopback隧道消费模型，启动前只读检查迁移和模型名；不会改普通服务默认配置。
该路径需要单独的宿主Python开发环境，不是CPU容器的一项环境变量开关。

## 结构与证据

- [架构与代码阅读路径](docs/public-architecture.md)：任务、事务、模型包、结果助手职责。
- [延迟实况监测](docs/engie-monitor.md)：固定模型的预测后评分、影子比较与中断续跑；历史到达为模拟。
- [助手困难案例](docs/public-assistant-case.md)：一次保留边界的失败、诊断和可复核路径。
- [精选结果与限制](docs/public-results.md)：效果、负结果与独立复现的区别。
- [测试入口](docs/public-tests.md)：普通软件、原模型、浏览器与显式付费问答。
- [推理交付](docs/results/inference-i4-20261007/README.md)：可选后端与公开复现的验证范围。
- [数据出处](data/README.md) 与 [第三方许可](THIRD-PARTY-NOTICES.md)。

源码包含全部业务模块、迁移、产品测试和前端源文件；原模型附件单独交付，并随附
[内容与许可说明](docs/third-party/MODEL-ARTIFACT-NOTICE.md)。维护者导出工具、精选清单及
其导航检查留在维护仓，不作为产品测试分发。
内部工作包、密码、原始请求/训练日志、旧Git历史和私人运行目录不属于公开树。
报告中的原始历史路径仅作出处记录，不是外部程序的输入路径。
