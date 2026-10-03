"""Q1开发比较的轻量冻结合同；不导入训练库，旧spec及旧包指纹保持不变。"""

from typing import Literal

PROTOCOL_VERSION = "q1-fixed-three-window-v1"
INPUT_SHA256 = "c4f6cfd2fd412dd203c0aae1fe3f070446874e402c4a82974da1f6e36fa0d47e"
WINDOWS = {
    "early_1": ("2019-02-04T23:35:00", "2019-02-18T10:10:00"),
    "early_2": ("2019-02-18T23:35:00", "2019-03-04T10:10:00"),
    "main": ("2019-03-04T23:35:00", "2019-03-18T10:10:00"),
}
TEST_START = "2019-03-18T11:15:00"
RIDGE_ALPHAS = {"ridge_0_1": 0.1, "ridge_1": 1.0, "ridge_10": 10.0}
CandidateKey = Literal["none", "ridge_0_1", "ridge_1", "ridge_10", "hgb_delta"]


def protocol_document() -> dict:
    return {
        "version": PROTOCOL_VERSION, "input_sha256": INPUT_SHA256,
        "windows": {key: {"start": start, "end": end, "samples": 3872}
                    for key, (start, end) in WINDOWS.items()},
        "test_start": TEST_START, "test_scored": False,
        "selection": "equal_mean_of_three_window_mae",
        "ridge_alphas": dict(RIDGE_ALPHAS), "random_seed": 42,
        "train_rule": "target_timestamp < evaluation_first_cutoff",
        "power_quantiles": [1 / 3, 2 / 3], "change_quantiles": [0.5, 0.9],
        "final_refit": "all_labels_before_test_first_cutoff_with_train_only_preprocessing",
    }


def candidate_model_set(key: str) -> list[str]:
    if key not in ("none", *RIDGE_ALPHAS, "hgb_delta"):
        raise ValueError("unsupported_candidate_key")
    return ["persistence", "hist_gradient_boosting"] + ([] if key == "none" else [key])
