# 原模型附件：内容、数据来源与许可

本说明应与以下两份原 ZIP 一起发布，并在 Git 中保留。ZIP 原字节及严格成员信任锚
不改变；本说明是外置旁文件，不添加到 ZIP 内。正式下载地址由发布版本提供。

| 附件 | 大小 / bytes | SHA-256 |
| --- | ---: | --- |
| `engie-original-models.zip` | 50,697,203 | `ab64ea45ab5138d3b7e47d27d38e3d81a0f13405c33172a6b82c880fa018bcce` |
| `engie-development-models.zip` | 24,146,787 | `dd1210fff0990a8015a3b3a5b08cd7a597ae08b3af739a33e44448cfb787bd85` |

## 实际内容

最终附件有八个成员：五个模型包（学习模型仅用2014数据拟合，持久性不拟合）、
2015 的 `predictions.npz`、冻结
`protocol.json` 和 `result.json`。开发附件有十四个成员：2014 Q2/Q3/Q4 各三个模型和
各一份预测数组，以及冻结 `protocol.json`、`baseline.json`。

NPZ 不只是模型预测：还含 UTC 起报纳秒 `issue_ns`、四机组六个时距的实况目标值
`targets`、输入/标签/边界有效性及可评分掩码，以及各模型预测。最终数组覆盖 52,560
个计划起报；这不等于全部输入合法，也不等于独立样本数。模型和评价身份由源码中的
固定信任锚验证，准备和导入过程不重新拟合。

附件没有完整原始 SCADA ZIP、其他气象文件、训练日志、密码或私人 Git 历史。
实况目标值是从公开数据选取、对齐并形成多时距数组的观测值，不能声称附件没有数据。
JSON 内的相对历史路径只记录出处；消费者使用平面附件名，不需要维护者的私人布局。

## 来源、归因和加工

数据提供方是 **ENGIE**，数据集为法国 La Haute Borne 风场 SCADA，来源说明的更新日
为 **2019-10-09**。本项目使用 OpenOA 固定提交
`9bfc7a3dc542b17bbfc06b692dbfb8b23c754975` 提供的公开副本：

- [固定来源及许可说明](https://raw.githubusercontent.com/NatLabRockies/OpenOA/9bfc7a3dc542b17bbfc06b692dbfb8b23c754975/examples/data/readme.md)
- [固定原始数据 ZIP](https://raw.githubusercontent.com/NatLabRockies/OpenOA/9bfc7a3dc542b17bbfc06b692dbfb8b23c754975/examples/data/la_haute_borne.zip)
- [Etalab Open Licence 2.0 正文](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)

原 ZIP 为 36,762,939 bytes，SHA-256
`be5ea66a3355286e491f5618250dc83e85252a8cb337748d7ba19edc50df6138`。
只消费固定 SCADA 成员，不重新分发其中其他气象文件。本项目的加工包括按时间和机组
对齐、形成 10--60 分钟的六时距目标和有效性掩码、在 2014 数据拟合模型，以及输出
2014 开发/2015 最终评价的预测与指标。它们是基于公开数据形成的附件，不是本项目
自主采集的数据，也不暗示 ENGIE、OpenOA 或数据提供方认可、背书本项目。

ENGIE 来源采用 **Etalab Open Licence 2.0**，复用应保留提供方、来源、更新日和本项目
加工说明。自有代码的项目许可仍为候选，不能据此重新授权第三方数据。下载者应连同
本说明保留数据归因，区分模型、观测目标值、预测与代码的权利来源。

`joblib` 会执行 Python 反序列化指令，只使用与源码固定信任锚一致的原附件；不要加载
来源不明的模型。附件取得、校验和导入流程见仓库的 `docs/public-reproduction.md`。
