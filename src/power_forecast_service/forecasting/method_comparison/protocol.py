"""事前固定搜索空间；顺序同时决定验证指标并列时的取舍。"""

import hashlib
import json

from ..development_protocol import INPUT_SHA256, WINDOWS
from ..sequence_protocol import CONFIG, LOOKBACK, SEQUENCE_FEATURES

VERSION = "q1-external-method-comparison-v1"
SOURCE_REVISION = "4e938a1767106324dd753b2a44832bf870a0252e"
EXPECTED_COUNTS = {"early_1": (8020, 2008), "early_2": (11245, 2815), "main": (14471, 3621)}


def recipes():
    items = [{"id": "ridge_original", "family": "ridge_original", "alpha": 0.1}]
    items += [{"id": f"ridge_history_{alpha:g}", "family": "ridge_history", "alpha": alpha}
              for alpha in (100.0, 10.0, 1.0, 0.1)]
    items += [{"id": f"lightgbm_{leaves}_{minimum}", "family": "lightgbm",
               "num_leaves": leaves, "min_child_samples": minimum}
              for leaves in (15, 31) for minimum in (200, 50)]
    items += [{"id": f"encoder_{target}", "family": "encoder", "target": target}
              for target in ("direct", "delta")]
    items += [{"id": f"timexer_{lr:g}", "family": "timexer", "learning_rate": lr}
              for lr in (0.0001, 0.001)]
    return items


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def protocol(device):
    if device not in ("cpu", "cuda"):
        raise ValueError("unsupported_comparison_device")
    return {"version": VERSION, "input_sha256": INPUT_SHA256, "windows": WINDOWS,
            "lookback": LOOKBACK, "features": SEQUENCE_FEATURES, "horizon_steps": 12,
            "inner_fraction": 0.2, "purge": "target_strictly_before_validation_cutoff",
            "recipes": recipes(), "fits": 39, "seed": 42, "threads": 2, "device": device,
            "encoder": {**CONFIG, "epochs": 60},
            "timexer": {"d_model": 32, "heads": 4, "layers": 2, "feedforward": 64,
                        "patch": 6, "dropout": 0.1, "activation": "gelu", "use_norm": True,
                        "source_revision": SOURCE_REVISION},
            "lightgbm": {"learning_rate": 0.03, "n_estimators": 1000,
                         "early_stopping_rounds": 50, "objective": "regression",
                         "selection_metric": "nonnegative_clipped_mae"},
            "neural_training": {"epochs": 60, "loss": "mse", "optimizer": "AdamW",
                                "weight_decay": 0.01, "batch_size": 256, "gradient_clip": 1.0,
                                "selection_metric": "nonnegative_clipped_mae"},
            "selection": "inner_validation_only; earliest_epoch_then_recipe_order",
            "refit_on_inner_validation": False, "fit_budget_seconds": 5400,
            "outer_exposure": "all_39_terminal_attempts_and_selection_lock_before_prediction",
            "evidence_scope": "previously_seen_Q1_development_not_independent_test",
            "investment_screen": {"mean_mae_gain": 0.03, "improving_windows": 2,
                                  "max_window_mae_regression": 0.05,
                                  "max_low_change_mae_regression": 0.05,
                                  "mean_rmse_must_not_worsen": True}}
