# 固定模型的延迟实况监测

这条流程回答：预测提交以后，哪些时距已获得可评分实况，主用与影子模型在同一批样本上
表现怎样。首版使用已登记的 ENGIE 模型和历史数据，不训练、不自动切换模型。
**20分钟到达延迟是模拟合同，不是实际采集日志。** 2015已用于之前的评价，不能在这里
再次当作新模型选择的未见留出集。

## 时间与职责

| 时钟 | 含义 |
| --- | --- |
| 输入观测时间 | 十分钟网格；每个起报使用 issue−130 到 issue−20 的12行四机组历史 |
| 起报 issue | UTC网格，固定模型产生10、20、30、40、50、60分钟预测 |
| 目标 target | issue＋时距；不是标签进入系统的时刻 |
| 实况 available | 模拟 target＋20分钟；显式补到的标签可以更晚，不能更早 |
| processed_until | 已提交模拟时钟；与真实数据库 created_at 和机器当前时间不同 |

主用必须为 persistence，影子为同一导入对象中另一个已登记模型。选择身份后不可用同一个
request_key更改时段、模型或窗口。原登记、源数据与工件hash仍是输入依据，没有第二套
MLflow Registry或模型训练流程。四机组功率全部到齐才计算全场残差；负功率保留，缺失不填零。

## 普通操作

先按[独立复现](public-reproduction.md)准备、导入可信原模型附件，并在**自有运行栈**中运行
包含0007迁移的新源码。旧运行镜像不含本功能，不要把源文件存在误认为服务已升级。
`GET /engie/imports`给出模型ID；从同一个导入对象选择 persistence 和一个候选。

下面使用2015-01-01一天作为已见历史回放，替换两个已登记ID：

```powershell
uv run --no-sync python tools/dev/replay_engie_monitor.py `
  --base-url http://127.0.0.1:18000 `
  --champion <persistence-artifact-id> --shadow <candidate-artifact-id> `
  --start 2015-01-01T00:00:00+00:00 --end 2015-01-02T00:00:00+00:00 `
  --key my-fixed-model-replay
```

CLI使用普通HTTP合同，输出各时距的已评分、待到达、到期缺失、滚动MAE和复核提示；
`--json`给出完整报告、模型/来源身份及 bias。输入同一key及相同配置即可从已提交时钟继续，
不是重新累计一份指标。配置改变须新建key，不覆盖原比较。`--max-steps`默认12步，
每次请求最多144步，限制的是单请求工作量，不截断整个时段或削减模型上下文。
`--window-issues`默认36个已计划起报，支持12–144；它不是“最后36个成功标签”。

如果宿主没有应用依赖，可以在已启动的新API容器中使用该脚本，示例地址改为容器内的
`http://127.0.0.1:8000`。依赖准备沿公开CPU路径，不同步宿主CUDA环境。

| API | 用途 |
| --- | --- |
| `POST /engie/monitors` | 以key、模型ID、start/end和window_issues登记比较，返回ID和报告 |
| `POST /engie/monitors/{id}/advance` | `through`模拟时钟、`max_steps`有限推进；每步独立提交 |
| `GET /engie/monitors/{id}` | 从PG账本重算报告，不依赖旧进程中的累计器 |
| `POST /engie/monitors/{id}/labels` | 补充同一回放对象迟到的四机组实况 |

标签请求示例：

```json
{
  "target_time": "2015-01-01T00:10:00+00:00",
  "available_at": "2015-01-01T01:00:00+00:00",
  "turbines": {"R80711": 100.0, "R80721": null, "R80736": 120.0, "R80790": 130.0}
}
```

上面的功率只是接口形状示例，**不可作为原历史实况提交**。标签提交者负责来源真实性；
本地单使用者接口没有真实遥测鉴权、设备连接或任意标签真实性判定。
available_at不得超过已推进时钟；需要模拟更晚到达时，继续推进through即可。
名义回放结束后仍可推进并补缺失标签，replay_complete只表示计划事件已处理，不代表全部
实况齐全。已经收到的有限值不可改写；null允许后来补齐，null重投不会删除已有值。

## 恢复与可解释报告

每个十分钟步骤先生成并写入预测，再读取当时可得实况、插入成对残差，最后更新游标。
这些操作在同一PG事务；步骤中断会整体回滚，前面已提交步骤仍保留。读报告也取得同一
对象行锁，避免多条查询组合出不同时点的指标。并发推进先锁对象再重读游标。
残差主键为 monitor/issue/horizon；两个模型绑定同一目标，重投不会多计一次。
数据库外推理可在回滚后再执行，**不声称推理恰好执行一次**。

计划起报、已处理起报、合法输入、无效输入、失败与每个模型实际输出分别计数。各时距
另给已评分成对数、待到达数和到期仍缺失数。单模型成功但另一个失败的输出会保留，
不会冒充成对比较。bias定义为 prediction−actual，全场单位为kW。

### 当前窗口的样本支持

累计的`scored_pairs`不是当前MAE的样本数。MAE/bias只使用最近至多`window_issues`个
已处理起报中的`rolling_pairs`，分母为实际成对数；缺失值不填零，也不把旧窗口的已评分
样本搬进新窗口。当前窗口另外报告待到达、到期缺失、无效输入和预测失败数量：

```text
rolling_scheduled_issues = rolling_pairs + rolling_pending_labels
                         + rolling_due_missing_labels + rolling_invalid_inputs
                         + rolling_failed_issues
```

这里的窗口分母包含已处理的无效/失败起报，但不包含尚未处理的未来计划。
普通CLI同时展示累计`已评分/待到达/到期缺失`、窗口`成对/已处理`以及窗口四类缺口。
例如累计28对、窗口24/36且有缺口时，不能把MAE解释为28对或完整36对的结果。

`window_status`只描述样本完整性：尚未处理为`not_started`；有无效输入、预测失败或
到期缺失为`incomplete`；只有尚未到达的实况为`awaiting_labels`；所有窗口起报均已成对
评分才为`complete`。它没有新增百分比阈值，不是数据真实性、模型健康或统计置信度。
相对误差`alert`仍按原规则独立计算，因此`within_margin`可以与`incomplete`同时出现。
补到窗口外的旧实况只改变累计数；窗口内补齐才更新本窗口计数和指标。重复补齐不多计。
实际使用与迟到补齐检查见[W2记录](results/wind-monitor-w2-20261006/README.md)。

复核规则在首轮运行前确定：滚动同样本至少12对，且影子MAE超过主用10%，才显示
`shadow_worse_review`；不足显示`insufficient_pairs`。这是可解释的演示阈值，不是行业SLO、
统计显著性或生产采用门。`within_margin`也不保证主用模型健康，两者可能同时很差。
首版不实现跨窗口漂移检测、告警外发、概率区间、自动重训或自动发布。

## 验证与阅读

- [自然回放结果](results/wind-monitor-w1-20261005/README.md)：范围、数值和限制。
- [轻量时间合同](../src/power_forecast_service/forecasting/engie_monitor_contract.py)。
- [历史适配器](../src/power_forecast_service/forecasting/engie_predictor.py)：inputs与observation分开，预测不读取保存的targets。
- [事务与报告](../src/power_forecast_service/experiments/engie_monitor.py)。
- [普通HTTP入口](../src/power_forecast_service/api/engie_monitor.py)与[CLI](../tools/dev/replay_engie_monitor.py)。
- [真实PG验证](../tests/integration/postgres/test_engie_monitor.py)：原模型、回滚、重复、迟到、并发与实际TCP进程重启。

测试需先准备最终附件并设置`WIND_ENGIE_FINAL_SOURCE`。它只创建/删除自己的随机测试库；
实际TCP检查只启动/停止自己的API进程与空闲端口，不接管普通18000服务。
普通CI不提供原附件，因此不会自动运行这一原模型组，不能把本地验证写成新远端CI成绩。
