# 公开数据与加工范围

源码自带 `data/sample/wind_2019_q1.csv`：公开来源2019年Q1固定连续片段，
用于单序列一小时后预测的历史代理场景。

- 大小2,232,552 bytes；SHA-256为
  `c4f6cfd2fd412dd203c0aae1fe3f070446874e402c4a82974da1f6e36fa0d47e`。
- 来源：Renewable Energy and Electricity Demand Time Series Dataset with Exogenous Variables at 5-minute Interval，
  Mendeley v1，[DOI](https://doi.org/10.17632/fdfftr3tc2.1)。
- 作者、版本和来源归因见源码的 `data/provenance/mendeley_fdfftr3tc2_v1_metadata.json`，
  文件清单见同目录 `mendeley_fdfftr3tc2_v1_files.json`。
- 原来源元数据声明CC BY 4.0；加工仅选Q1片段，不宣称自主采集。

时间是无时区 `source_time`，不能当作UTC；字段和单位保留来源限制，不推断地区、
容量或经济收益。Q1已用于既有开发/评价，不是未见盲测；不是中国电网或客户生产数据。
全量原始文件、私人质量探针和迁移记录不分发，运行不依赖这些文件。

ENGIE四机组SCADA不随源码复制；从固定OpenOA提交获取数据，来源标注Open Licence 2.0。
原开发/最终模型附件分别校验固定信任锚，准备时不拟合。
具体获取方式、许可链接、ZIP与模型附件hash见 `docs/public-reproduction.md`。
项目代码的MIT许可候选不重新定义第三方数据许可。
