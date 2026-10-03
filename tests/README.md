# 测试入口

完整分组、所需附件、平台和副作用见 [公开测试说明](../docs/public-tests.md)。
推荐先启动独立CPU Compose，再执行 `pwsh -File tools/dev/check-service.ps1`。
默认运行软件组，包含局部规则、真实PG/模型组件和三个HTTP开发实验；原模型组必须
显式加 `-OriginalReplay` 并提供两个准备目录。缺少所选真实依赖会失败，不切换mock。

## 职责与状态

- `unit/` 检查独立规则与明确注入的替身，不能作为真实集成成绩。
- `integration/forecasting/` 使用真实本地模型；`integration/postgres/` 使用真实PG。
- PG夹具创建随机 `wind_test_*` 数据库、执行Alembic并只清理自己创建的库。
- `e2e/` 通过独立API、Redis和worker提交任务，实际训练、登记并保留任务/模型。
- 跨进程测试不负责启动或停止Docker。浏览器分组见公开测试说明。

每次e2e使用新的幂等键；超时仅表示没有观察到终态，不表示任务已经取消。
运行回执、JUnit和浏览器输出放忽略提交的 `.local/runtime/`，不覆盖历史结果。
原模型测试不训练，必须消费经过固定信任锚验证的附件；不能加载陌生joblib。
默认测试不调用真实provider，付费问答由明确选择的路径执行。

本机Python测试使用工程自己的uv锁和现有环境；公开CPU路径不要求宿主CUDA。
Linux CPU容器没有pwsh时会跳过对应入口测试，Windows应另行验证这些用例。
历史故障演练不是普通测试入口，不为日常验证停止已有服务。
