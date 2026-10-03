"""共用训练观察与权重选择；外层数据不在本模块接口中。"""

import copy
import json
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..model_diagnosis import measurements
from ..sequence_model import HistoryEncoder
from ..sequence_protocol import CONFIG, SEQUENCE_FEATURES
from .timexer import TimeXer

TARGET_LAST = [*range(1, 8), 0]


def prediction_metrics(actual, raw):
    raw = np.asarray(raw, dtype=float)
    if raw.shape != np.asarray(actual).shape or not np.isfinite(raw).all():
        raise ValueError("comparison_prediction_shape_or_finiteness")
    return {"raw": measurements(actual, raw), "clipped": measurements(actual, np.maximum(raw, 0)),
            "clipped_fraction": float(np.mean(raw < 0))}


class Checkpoints:
    """严格小于保留最早并列epoch，复制参数以免后续optimizer更新污染历史。"""

    def __init__(self):
        self.best = float("inf")
        self.best_epoch = None
        self.states, self.epochs = {}, {}

    def observe(self, epoch, score, net):
        if score < self.best:
            self.best, self.best_epoch = score, epoch
            self.states["best60"] = copy.deepcopy(net.state_dict())
            self.epochs["best60"] = epoch
            if epoch <= 20:
                self.states["best20"] = copy.deepcopy(net.state_dict())
                self.epochs["best20"] = epoch
        if epoch == 20:
            self.states["epoch20"] = copy.deepcopy(net.state_dict())
            self.epochs["epoch20"] = epoch


class NeuralPredictor:
    def __init__(self, recipe, mean, scale, target_mean, target_scale):
        self.recipe = recipe
        self.mean, self.scale = np.array(mean), np.array(scale)
        self.target_mean, self.target_scale = target_mean, target_scale
        self.order = TARGET_LAST if recipe["family"] == "timexer" else list(range(8))
        self.net = TimeXer() if recipe["family"] == "timexer" else HistoryEncoder(CONFIG)

    def tensor(self, windows):
        return torch.tensor(((windows - self.mean) / self.scale)[:, :, self.order], dtype=torch.float32)

    def restore(self, values, windows):
        result = values.astype(float) * self.target_scale + self.target_mean
        if self.recipe.get("target") == "delta":
            result += windows[:, -1, 0]
        return result

    def predict(self, windows, *, device="cpu"):
        self.net.to(device).eval()
        values = self.tensor(windows)
        with torch.inference_mode():
            output = torch.cat([self.net(batch.to(device)).cpu() for batch in values.split(512)]).numpy()
        return self.restore(output, windows)

    def save(self, path, checkpoints, details):
        states = {key: {name: tensor.detach().cpu() for name, tensor in state.items()}
                  for key, state in checkpoints.states.items()}
        torch.save({"recipe": self.recipe, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                    "target_mean": self.target_mean, "target_scale": self.target_scale,
                    "feature_order": [SEQUENCE_FEATURES[index] for index in self.order],
                    "states": states, "epochs": checkpoints.epochs, "details": details}, path)

    @classmethod
    def load(cls, path, checkpoint="best60"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        item = cls(saved["recipe"], saved["mean"], saved["scale"], saved["target_mean"], saved["target_scale"])
        if saved["feature_order"] != [SEQUENCE_FEATURES[index] for index in item.order]:
            raise ValueError("comparison_checkpoint_feature_mapping")
        item.net.load_state_dict(saved["states"][checkpoint], strict=True)
        return item


def fit_neural(data, recipe, artifact, *, device, epochs=60, observe_training=True, history_path=None,
               deadline=None):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    targets = data.train.target_power.to_numpy(float)
    if recipe["family"] == "timexer":
        # 原生窗口反归一化回到输入尺度；只在包装器中恢复一次原始功率。
        target_mean, target_scale = float(data.mean[0]), float(data.scale[0])
    else:
        if recipe.get("target") == "delta":
            targets = targets - data.train_windows[:, -1, 0]
        target_mean, target_scale = float(targets.mean()), max(float(targets.std()), 1e-8)
    predictor = NeuralPredictor(recipe, data.mean, data.scale, target_mean, target_scale)
    net = predictor.net.to(device)
    inputs = predictor.tensor(data.train_windows)
    target_tensor = torch.tensor((targets - target_mean) / target_scale, dtype=torch.float32)
    loader = DataLoader(TensorDataset(inputs, target_tensor), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    optimizer = torch.optim.AdamW(net.parameters(), lr=recipe.get("learning_rate", 0.001),
                                 weight_decay=0.01)
    checkpoints, history = Checkpoints(), []
    before = next(net.parameters()).detach().cpu().clone()
    max_gradient = 0.0
    for epoch in range(1, epochs + 1):
        net.train()
        total = 0.0
        for x, y in loader:
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("comparison_neural_budget_exhausted")
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(net(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("comparison_nonfinite_loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            max_gradient = max(max_gradient, float(norm))
            optimizer.step()
            total += float(loss.detach()) * len(x)
        # eval/inference模式及张量切片不创建随机采样器，不消耗训练随机流。
        val_raw = predictor.predict(data.validation_windows, device=device)
        validation = prediction_metrics(data.validation.target_power.to_numpy(float), val_raw)
        checkpoints.observe(epoch, validation["clipped"]["mae"], net)
        record = {"epoch": epoch, "train_mode_standardized_mse": total / len(targets),
                  "validation": validation, "elapsed_seconds": perf_counter() - started}
        if observe_training:
            record["train_eval"] = prediction_metrics(data.train.target_power.to_numpy(float),
                                                       predictor.predict(data.train_windows, device=device))
        history.append(record)
        if history_path is not None:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        if epoch == 1 or epoch % 10 == 0:
            print(f"{recipe['id']} epoch={epoch} inner_mae={validation['clipped']['mae']:.3f}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    change = float((next(net.parameters()).detach().cpu() - before).abs().max())
    if not change or not max_gradient:
        raise ValueError("comparison_parameters_not_updated")
    details = {"fit_elapsed_seconds": perf_counter() - started, "history": history,
               "selected_epoch": checkpoints.best_epoch, "checkpoint_epochs": checkpoints.epochs,
               "device": device, "seed": 42, "max_gradient_norm": max_gradient,
               "parameter_max_change": change, "parameters": sum(p.numel() for p in net.parameters()),
               "scaler_mean": data.mean.tolist(), "scaler_scale": data.scale.tolist(),
               "target_mean": target_mean, "target_scale": target_scale,
               "feature_order": [SEQUENCE_FEATURES[index] for index in predictor.order],
               "validation": history[checkpoints.best_epoch - 1]["validation"],
               "resume_scope": "inference_checkpoints_only_not_optimizer_rng_resume"}
    predictor.save(artifact, checkpoints, details)
    predictor.net.load_state_dict(checkpoints.states["best60"])
    before_reload = predictor.predict(data.validation_windows, device="cpu")
    restored = NeuralPredictor.load(artifact).predict(data.validation_windows)
    np.testing.assert_allclose(restored, before_reload, rtol=1e-6, atol=1e-3)
    details["reload_max_difference"] = float(np.max(np.abs(restored - before_reload)))
    return predictor, details
