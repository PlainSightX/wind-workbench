# 独立环境复现

这里有两条独立的路径：新开发实验验证软件流程，原ENGIE模型包验证历史预测回放。
新训练的模型不能代替原评分包。当前已实测Windows PowerShell 7 + Docker Desktop
Linux容器；容器使用固定CPU依赖，不要求宿主安装CUDA。本机Python开发仍按原uv锁。
宿主还需安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 并加入PATH：
`check-service.ps1`用它从已提交的uv.lock导出测试依赖。普通容器启动不要求宿主Python
环境或CUDA同步；缺uv时检查脚本会在任何Docker操作前明确退出。

## 1. 新软件流程

在独立仓库副本根目录执行。端口若占用，选择其他未占用端口；项目名必须与其他实例不同。
已有卷需要原密码，启动脚本不会覆盖或重置它。

```powershell
$env:COMPOSE_PROJECT_NAME = 'wind-demo'
$env:WIND_HTTP_PORT = '18081'
$env:WIND_DB_PORT = '15481'
$env:WIND_RUNTIME_ROOT = './.local/runtime'
pwsh -File tools/dev/start-app.ps1 -Build
pwsh -File tools/dev/check-service.ps1 -ReportName public-flow
```

输入是自带的Q1样本，来源为Mendeley数据集 `10.17632/fdfftr3tc2.1`，采用CC BY 4.0；
具体作者和加工范围见[data说明](../data/README.md)及其元数据。这是公开历史代理数据，
不是客户或中国电网数据。新实例生成自己的密码，迁移空数据库，并通过真实Redis/worker
执行默认、Ridge与候选回放三个开发实验；测试核对重复请求、任务/attempt身份、
默认两包交付与Ridge三包交付，并验证选版预测。Ridge浏览器输入回执在
`.local/runtime/ui-delivery.json`，候选另有3872行回放验收。
启动和查询不训练；测试中的显式POST会训练并留下结果。不要反复运行来冒充独立案例。

普通使用只需上述 `start-app.ps1`，检查脚本是可选验收，运行后会新增三个实验。打开
自己设置端口的页面，在“实验任务”选择对照组合并提交；页面返回 task ID，然后在任务
详情查看 attempt、错误码和 run，在“结果比较”查看同一评分协议的模型，再在“历史预测”
选择已登记模型与有效历史窗口。初次空库没有结果，显式提交后才产生模型；不需要用户
准备冻结报告、手填内部清单或找维护者的旧 UUID。输入、样本范围、输出身份和使用边界见
[数据说明](../data/README.md)及[架构](public-architecture.md)。

任务失败时先保留 task/attempt 身份和错误码，不以相同页面操作反复提交替代恢复。
系统处理允许的投递/租约恢复；“原样重跑”用户主动结对入口仍未实现，不能声称已可点击。
结果助手需要单独配置 provider，失败或证据不足会明确返回状态；它不决定预测或模型采用。
用户负责选择实验/模型/问题及审阅结论，系统负责冻结配置、调度、身份校验和持久交付。

## 2. 原模型附件与数据

模型附件名为 `engie-original-models.zip`，50,697,203 bytes，SHA-256：
`ab64ea45ab5138d3b7e47d27d38e3d81a0f13405c33172a6b82c880fa018bcce`。
这是已导出并核验的本地候选附件，**尚无公开下载链接**；正式发布时必须一并提供该附件，
不能只发布源码后声称外部已可重放。它包含原五模型、原预测数组和冻结result/protocol；
不含原始数据、训练日志、密码或私人Git历史。原result中的相对历史路径仅记录出处，
消费者通过平面文件名取资源，不使用那些路径。

开发附件为 `engie-development-models.zip`，24,146,787 bytes，SHA-256：
`dd1210fff0990a8015a3b3a5b08cd7a597ae08b3af739a33e44448cfb787bd85`。
它包含2014开发阶段原九包及预测，准备时必须显式指定 `--release development-2014`；
其信任锚与最终五包分开，不能互相代替。两个附件都需在正式发布时提供。

SCADA数据归属于ENGIE，OpenOA固定提交的数据说明标注Open Licence 2.0及更新日
2019-10-09。原模型和预测是基于该数据的派生结果，不暗示ENGIE或OpenOA背书。
数据不随源码Git分发：

- [固定原始ZIP](https://raw.githubusercontent.com/NatLabRockies/OpenOA/9bfc7a3dc542b17bbfc06b692dbfb8b23c754975/examples/data/la_haute_borne.zip)
- [来源及许可说明](https://raw.githubusercontent.com/NatLabRockies/OpenOA/9bfc7a3dc542b17bbfc06b692dbfb8b23c754975/examples/data/readme.md)
- [Open Licence正文](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)

原始ZIP应为36,762,939 bytes，SHA-256：
`be5ea66a3355286e491f5618250dc83e85252a8cb337748d7ba19edc50df6138`。
导入只消费固定SCADA成员，程序还校验该成员hash；不将ZIP内其他气象文件重新分发。

## 3. 准备和导入

将模型附件放在忽略提交的 `.local/replay/` 目录。以下容器命令使用仓库自己的代码与CPU依赖，
无需先安装Windows/CUDA虚拟环境。`prepare`省略 `--source-zip` 时从上述固定URL下载；
若网络代理造成问题，可先下载原ZIP，再加 `--source-zip /replay/source.zip`，校验要求不变。

```powershell
docker compose --profile app run --rm --no-deps `
  --volume "${PWD}/tools/dev:/tools:ro" --volume "${PWD}/.local/replay:/replay" `
  api python /tools/prepare_engie_replay.py prepare `
  --bundle /replay/engie-original-models.zip --destination /replay/prepared

docker compose --profile app run --rm --no-deps `
  --volume "${PWD}/tools/dev:/tools:ro" --volume "${PWD}/.local/replay:/replay" `
  api python /tools/import_engie.py import --release final-2015 `
  --source /replay/prepared --receipt /replay/import.json
```

需要2014页面和完整原模型测试时，再准备开发附件：

```powershell
docker compose --profile app run --rm --no-deps `
  --volume "${PWD}/tools/dev:/tools:ro" --volume "${PWD}/.local/replay:/replay" `
  api python /tools/prepare_engie_replay.py prepare --release development-2014 `
  --bundle /replay/engie-development-models.zip --destination /replay/prepared-development

docker compose --profile app run --rm --no-deps `
  --volume "${PWD}/tools/dev:/tools:ro" --volume "${PWD}/.local/replay:/replay" `
  api python /tools/import_engie.py import --release development `
  --source /replay/prepared-development --receipt /replay/import-development.json

$env:WIND_ENGIE_FINAL_SOURCE = "$PWD/.local/replay/prepared"
$env:WIND_ENGIE_DEVELOPMENT_SOURCE = "$PWD/.local/replay/prepared-development"
pwsh -File tools/dev/check-service.ps1 -OriginalReplay `
  -TestPath tests/integration/postgres -ReportName original-replay
```

准备先核对源码锚定的原result/protocol及其模型/预测hash，再发布完整成员；已有不同
内容会拒绝覆盖。导入禁止模型与scaler拟合，五包全部验证后一次登记；重复导入保持
同一身份。joblib可以执行Python对象恢复，因此只加载通过固定信任锚的原附件。
不要把接口解释为接受任意陌生模型。

打开独立实例页面，在ENGIE区域选 `2015-final` 与具体模型，可以回放、发布、刷新读回。
历史数据包含实况供核验，预测输入仍遵守起报时间和到达延迟。2015已曝光，不是新留出。

## 4. 代表回放检查

[verify_public_flow.py](../tools/dev/verify_public_flow.py)专用于上述演示：一个新Q1开发run，
一组原最终导入。它验证五模型各三个原评分窗口，发布一次预测并检查同key重复返回，
还检查新Q1对象不继承旧选型来源。这个验证器不是任意业务数据库的全量检查工具。

```powershell
$api = docker compose --profile app ps -q api
$image = docker inspect --format '{{.Image}}' $api
$artifacts = "$($env:COMPOSE_PROJECT_NAME)_experiment_artifacts"
$q1 = (Get-Content .local/runtime/ui-delivery.json -Raw | ConvertFrom-Json).run_id
docker run --rm --network "container:$api" `
  --label "com.docker.compose.project=$env:COMPOSE_PROJECT_NAME" `
  --label "com.docker.compose.project.working_dir=$PWD" --label wind.role=verification `
  --volume "${PWD}/tools/dev:/tools:ro" --volume "${PWD}/.local/replay:/evidence" `
  --volume "${PWD}/.local/runtime/app-password.txt:/run/secrets/app_password:ro" `
  --volume "${artifacts}:/app/artifacts:ro" `
  --env WIND_DB_HOST=postgres --env WIND_DB_PORT=5432 `
  --env WIND_DB_PASSWORD_FILE=/run/secrets/app_password --env WIND_ARTIFACT_ROOT=/app/artifacts `
  $image python /tools/verify_public_flow.py --q1-run-id $q1 --output /evidence/replay.json
```

命令按第1节示例运行目录编写；自定义 `WIND_RUNTIME_ROOT` 时，相应调整密码挂载。
保留原回执，显式重启API并等待健康后对同一命令追加 `--readback`，复核15份响应和发布记录。
不能重复初次检查后把多次发布当成新任务收益，也不能把fixture、新训练与原模型回放混用。
已有输出或对应的 `.pending.json` 会在HTTP前拒绝。验证器先持久化request_key，
在真正发布前另保存完整delivery_request；中断后保留pending，不用新key盲目重发。
若pending为 `complete`，可使用原命令的 `--output` 加 `--recover-receipt`，只在本地
补发完整回执，不再请求服务。若仍为 `delivery_prepared`，先依据其中的同key请求
核对服务结果，再决定恢复；不能把未完整验证的pending当通过证据。
`--readback`只证明响应和发布记录一致，API确实已重启须由外部进程操作另外留证。

## 5. 助手与浏览器

设置环境变量 `DEEPSEEK_API_KEY` 后执行 `pwsh -File tools/dev/prepare-assistant.ps1`。
默认只保存配置，不下载模型、不请求provider；在页面选择已导入/已完成对象后提问才收费。
它消费实际Compose运行目录，支持自定义 `WIND_RUNTIME_ROOT`，不会另写到默认目录。
可选 `-Vector` 下载固定BGE-small-zh-v1.5、校验文件并建立PG向量索引；
`-ProbeProvider` 才额外发送付费探针。常规API只读助手目录，准备容器单独获得写权限。

在 `web/` 执行 `npm ci`、`npx playwright install chromium`；设置
`WIND_TEST_BASE_URL` 为独立HTTP地址后执行 `npm test`。默认software组需要第1节回执；
`WIND_UI_SUITE=original` 需要两组原包。分组和付费边界见 [测试说明](public-tests.md)。

当前实测支持Windows PowerShell 7 + Docker Desktop Linux CPU容器；没有在纯Linux宿主
运行初始化脚本，也没有验证远端CI。附件、许可候选和代码均已整理为本地发布候选，
具体结果与负例见 [W2记录](public-release-w2.md)；GitHub发布仍是后续阶段。

维护者从原件导出附件使用 `prepare_engie_replay.py export --output <新ZIP路径>`，
开发包另加 `--release development-2014`。它需要历史原件，是维护者入口，不是外部使用前提。
