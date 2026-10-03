"""结构消费与量纲合同；随机输入只证明实现，不是预测收益。"""

import numpy as np
import pytest
import torch

from power_forecast_service.forecasting.method_comparison.neural import NeuralPredictor, TARGET_LAST
from power_forecast_service.forecasting.method_comparison.timexer import TimeXer


def model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    return TimeXer().eval()


def test_timexer_shape_batch_independence_and_latest_patch():
    net = model()
    x = torch.randn(3, 24, 8)
    together = net(x)
    apart = torch.cat([net(row[None]) for row in x])
    torch.testing.assert_close(together, apart, atol=1e-6, rtol=1e-6)
    captured = []
    hook = net.en_embedding.value_embedding.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    net(x)
    hook.remove()
    assert captured[0].shape == (3, 4, 6)
    normalized = (x[:, :, -1] - x[:, :, -1].mean(1, keepdim=True))
    normalized /= torch.sqrt(x[:, :, -1].var(1, keepdim=True, unbiased=False) + 1e-5)
    torch.testing.assert_close(captured[0].reshape(3, 24), normalized)
    with pytest.raises(ValueError, match="incomplete_patch"):
        TimeXer(patch=16)


def test_target_column_and_exactly_one_inverse_scaling():
    recipe = {"id": "timexer_0.0001", "family": "timexer", "learning_rate": 0.0001}
    predictor = NeuralPredictor(recipe, np.arange(8) * 10 + 100, np.arange(8) + 2, 100., 2.)
    windows = np.random.default_rng(42).normal(50, 10, (3, 24, 8))
    predictor.net.head.linear.weight.data.zero_()
    predictor.net.head.linear.bias.data.zero_()
    np.testing.assert_allclose(predictor.predict(windows), windows[:, :, 0].mean(1), atol=1e-4)
    assert TARGET_LAST == [1, 2, 3, 4, 5, 6, 7, 0]


def test_exogenous_temporal_shape_reaches_target_and_gradients():
    net = model()
    x = torch.randn(4, 24, 8)
    altered = x.clone()
    altered[:, :, 0] = torch.flip(altered[:, :, 0], dims=[1])
    assert torch.max(torch.abs(net(x) - net(altered))).item() > 1e-6
    net(x).sum().backward()
    for name in ("query_projection", "key_projection", "value_projection"):
        gradient = getattr(net.encoder.layers[0].cross_attention, name).weight.grad
        assert gradient is not None and gradient.abs().sum().item() > 0


def test_invalid_input_shape_is_rejected():
    with pytest.raises(ValueError, match="input_shape"):
        model()(torch.zeros(2, 23, 8))
