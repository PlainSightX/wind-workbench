"""打包可信来源，或在目标CPU运行时验证并登记；两个步骤均禁止重新拟合。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "verify", "import"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--release", choices=("development", "final-2015"), default="development")
    args = parser.parse_args()
    from power_forecast_service.storage.engie_imports import prepare_import, publish_import, stage_sources
    if args.release == "final-2015":
        from power_forecast_service.storage.engie_final_imports import prepare_import, stage_sources

    if args.action == "stage":
        result = stage_sources(args.repository, args.source)
    else:
        from lightgbm import LGBMRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from power_forecast_service.settings import Settings
        from power_forecast_service.storage.database import make_sync_engine

        def forbidden(*args, **kwargs):
            raise AssertionError("engie_import_must_not_fit")

        Ridge.fit = StandardScaler.fit = LGBMRegressor.fit = forbidden
        settings = Settings.from_environment() if args.action == "import" else None
        root = args.artifact_root or (settings.artifact_root if settings else None)
        if root is None:
            parser.error("verify requires --artifact-root")
        prepared, result = prepare_import(args.source, root)
        if settings:
            engine = make_sync_engine(settings)
            try:
                result["import_ids"] = publish_import(engine, prepared)
            finally:
                engine.dispose()
        result["training_forbidden"] = True
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
