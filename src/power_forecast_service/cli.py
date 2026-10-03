from __future__ import annotations

import argparse
import json
from pathlib import Path

from .forecasting.data import load_wind_frame
from .forecasting.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local wind-power prototype")
    parser.add_argument("--data", type=Path, required=True, help="Path to a validated wind CSV")
    parser.add_argument("--output", type=Path, help="Write the JSON result to this path")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run non-destructive quality audit without fitting a model",
    )
    args = parser.parse_args()
    if args.audit_only:
        _, report = load_wind_frame(args.data, strict=False)
        result = {
            "audit_type": "wind-data-quality",
            "input_file": str(args.data),
            "quality": report.as_dict(),
        }
    else:
        result = run_experiment(args.data)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
