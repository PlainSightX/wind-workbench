# 结果报告索引

每个目录是一次有日期的实验或交付记录，目录名保留开发时的编号。按主题阅读：

## 预测方法

| 报告 | 内容 |
| --- | --- |
| [ENGIE 基线](wind-engie-baseline-20260923/README.md) | 四机组数据合同、持续性 / Ridge / LightGBM 三季度开发对照 |
| [ENGIE 原包接入](wind-engie-service-a1-20260923/README.md) | 实际评分模型原字节接入服务，禁止重新拟合 |
| [ENGIE L1 与共享收缩](wind-engie-a2-20260923/README.md) | 开发期 MAE 改善 1.80%，未过预设 3% 采用门 |
| [ENGIE 2015 最终评价](wind-engie-a3-20260923/README.md) | 冻结后一次评价：MAE −2.985%、RMSE −4.038%；五模型页面回放与发布 |
| [Q1 序列模型比较](../WIND-SEQUENCE-RESULTS.md) | Ridge、HGB、增量 Transformer 与持续性的历史留出对照 |

## 持续评价与可靠性

| 报告 | 内容 |
| --- | --- |
| [延迟实况监测](wind-monitor-w1-20261005/README.md) | 固定模型先预测、后按模拟到达评分，进程重启后续跑 |
| [缺测与样本支持](wind-monitor-w2-20261006/README.md) | 窗口成对样本、待到达与缺测和 MAE 并列展示 |

## 结果助手与推理

| 报告 | 内容 |
| --- | --- |
| [推理配置对照](inference-i2-20261006/README.md) | Qwen3-14B：提示布局 × 前缀缓存四格对照 |
| [可选推理服务](inference-i3-20261007/README.md) | vLLM 经 HTTP / PostgreSQL 消费；过载、断连、未知结果与强杀恢复 |
| [推理复现入口](inference-i4-20261007/README.md) | 宿主启动前检查、公开测试与复现范围 |
| [查询规划微调](sft-obligation-8b-20261017/README.md) | Qwen3-8B LoRA 与 32B、DeepSeek-Flash 的首答规划对照 |

## 演示

- [操作录屏](wind-demo-20260924/workbench-demo.webm)：提交实验、比较结果、历史回放与结果问答（早期界面）。
