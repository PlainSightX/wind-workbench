"""序列实验的轻量合同；HTTP可导入，不提前加载torch。"""

from .development_protocol import INPUT_SHA256, TEST_START, WINDOWS

LOOKBACK = 24
SEQUENCE_KEYS = ("transformer_direct", "transformer_delta")
SEQUENCE_FEATURES = ["wind_power", "wind_speed", "humidity", "temperature",
                     "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
SEQUENCE_VERSION = "history-sequence-24-v1"
EVALUATION_VERSION = "q1-common24-three-window-v1"
CONFIG = {"lookback": 24, "d_model": 32, "heads": 4, "layers": 2,
          "feedforward": 64, "dropout": 0.1, "epochs": 20, "batch_size": 256,
          "learning_rate": 0.001, "weight_decay": 0.01, "gradient_clip": 1.0}


def sequence_model_set(key):
    if key not in SEQUENCE_KEYS:
        raise ValueError("unsupported_sequence_key")
    return ["persistence", "hist_gradient_boosting", "ridge_0_1", key]


def sequence_protocol():
    return {"version": EVALUATION_VERSION, "input_sha256": INPUT_SHA256,
            "windows": WINDOWS, "test_start": TEST_START, "lookback": LOOKBACK,
            "config": dict(CONFIG), "keys": list(SEQUENCE_KEYS), "seed": 42,
            "confirmation_seeds": [43, 44], "test_scored": False,
            "selection": "equal_mean_of_three_window_mae",
            "train_rule": "common_24_point_history_and_labels_before_evaluation_cutoff"}
