"""轻量预测协议；读取配置不应导入训练库或启动训练。"""

from typing import Literal

HORIZON_STEPS = 12
MODEL_VERSION = "hist-gradient-boosting-v0.1"
FEATURE_CONTRACT_VERSION = "lag-only-source-time-v0.1"
SPLIT_VERSION = "temporal-70-15-15-label-isolated-v0.2"
MODEL_SEED = 42
TrainingPolicy = Literal["auto_early_stopping", "fixed_iterations"]
DEFAULT_TRAINING_POLICY: TrainingPolicy = "auto_early_stopping"
EXPERIMENT_SPEC_VERSION = "experiment-v2"

# 已发布的参数用不可变条目保存；旧配置解释不随调用者修改字典而漂移。
_HGB_V1 = (
    ("learning_rate", 0.06), ("max_iter", 180),
    ("max_leaf_nodes", 31), ("l2_regularization", 0.2),
)
HGB_PARAMETERS = dict(_HGB_V1)


def legacy_hgb_parameters() -> dict:
    """返回历史四参数合同的副本，不补造历史训练观测。"""
    return dict(_HGB_V1)


def legacy_execution_parameters() -> dict:
    """冻结旧合同隐含的库默认语义；不随新策略函数调整而重新解释。"""
    return {
        **legacy_hgb_parameters(), "early_stopping": "auto", "validation_fraction": 0.1,
        "n_iter_no_change": 10, "tol": 1e-7, "scoring": "loss", "loss": "squared_error",
    }


def training_parameters(policy: TrainingPolicy = DEFAULT_TRAINING_POLICY) -> dict:
    """仅开放两种策略；除早停开关外的参数一致，禁止任意参数搜索。"""
    if policy not in ("auto_early_stopping", "fixed_iterations"):
        raise ValueError("unsupported_training_policy")
    return {
        **legacy_hgb_parameters(),
        "early_stopping": "auto" if policy == "auto_early_stopping" else False,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 1e-7,
        "scoring": "loss",
        "loss": "squared_error",
    }
