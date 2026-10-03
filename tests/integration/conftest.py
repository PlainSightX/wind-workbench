"""模型交付集成测试共用同次真实拟合，不在每个事务失败场景里重复训练。"""

import pytest


@pytest.fixture(scope="session")
def trained_product(sample_path):
    from power_forecast_service.forecasting.pipeline import train_experiment

    return train_experiment(sample_path)
