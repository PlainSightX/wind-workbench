"""准备精确原模型回放：export 是维护者动作，prepare 是外部使用入口。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export", help="导出原五模型和评分/协议，不带原始数据")
    export.add_argument("--repository", type=Path, default=Path.cwd())
    export.add_argument("--output", type=Path, required=True)
    prepare = actions.add_parser("prepare", help="校验附件并获取固定原始数据")
    prepare.add_argument("--bundle", type=Path, required=True)
    prepare.add_argument("--destination", type=Path, required=True)
    prepare.add_argument("--source-zip", type=Path, help="已下载的原始ZIP；省略则从固定URL下载")
    for command in (export, prepare):
        command.add_argument("--release", choices=["development-2014", "final-2015"], default="final-2015")
    args = parser.parse_args()
    from power_forecast_service.storage.engie_replay_bundle import export_bundle, prepare_bundle
    if args.action == "export":
        result = export_bundle(args.repository, args.output, args.release)
    else:
        result = prepare_bundle(args.bundle, args.destination, args.source_zip, args.release)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
