# 许可与来源

自有代码的拟发布许可见 [MIT候选](LICENSE.candidate)。尚未正式发布；最终许可将在发布轮确认。
下述数据、模型和改编代码分别遵守各自许可，不由该候选许可重新授权。

| 资源 | 精确范围和来源 | 许可 |
| --- | --- | --- |
| Chart.js 4.5.1 | npm锁定版本；编译入 web_static/app.js | [完整MIT](docs/third-party/Chart.js-MIT.txt) |
| @kurkle/color 0.3.4 | Chart.js传递依赖，见前端锁 | [完整MIT](docs/third-party/kurkle-color-MIT.txt) |
| Lucide 1.47.0 | npm锁定版本；图标编译入静态bundle | [完整ISC](docs/third-party/Lucide-ISC.txt) |
| TimeXer改编 | forecasting/method_comparison 下的时序候选，保留源码中的固定revision/修改说明 | [完整MIT](src/power_forecast_service/forecasting/method_comparison/LICENSE-TimeXer.txt) |
| Q1代理样本 | Mendeley DOI 10.17632/fdfftr3tc2.1；作者元数据、裁取加工及原单位见 [data说明](data/README.md) | CC BY 4.0 |
| ENGIE SCADA | ENGIE La Haute Borne，OpenOA固定提交的数据副本，更新日2019-10-09；外部下载 | Etalab Open Licence 2.0，见 [复现说明](docs/public-reproduction.md) |
| BGE-small-zh-v1.5 | BAAI固定 revision 7999e1d3359715c523056ef9478215996d62a620；可选下载，不进源码包 | MIT，[固定模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/7999e1d3359715c523056ef9478215996d62a620/README.md) |

原模型附件还含起报时间、实况目标值和有效性掩码；不是只分发预测数组。
[附件内容与许可](docs/third-party/MODEL-ARTIFACT-NOTICE.md)应随两份原ZIP发布，记录
ENGIE来源、更新日、Open Licence 2.0及加工范围；ZIP原字节保持，不暗示数据提供方背书。
其他Python与容器依赖由锁文件安装/拉取，未将其源代码或镜像二进制放入本项目发布树。
演示画面来自本项目在公开历史数据上的运行，不含客户数据。
