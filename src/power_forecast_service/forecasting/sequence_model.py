"""小型encoder回归器；训练统计、参数和state_dict构成可重载预测对象。"""

import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .sequence_data import history_windows, observation_features
from .sequence_protocol import CONFIG, LOOKBACK, SEQUENCE_FEATURES, SEQUENCE_KEYS


class HistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["d_model"]
        self.projection = nn.Linear(len(SEQUENCE_FEATURES), width)
        position = torch.arange(LOOKBACK).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-np.log(10000.0) / width))
        encoding = torch.zeros(LOOKBACK, width)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("position", encoding.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(width, config["heads"], config["feedforward"],
                                           config["dropout"], batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)
        # Encoder克隆结构但不应让层间权重完全相同；按已设seed分别初始化矩阵。
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, values):
        return self.head(self.encoder(self.projection(values) + self.position)[:, -1]).squeeze(-1)


class SequenceRegressor:
    def __init__(self, key, config=None):
        if key not in SEQUENCE_KEYS:
            raise ValueError("unsupported_sequence_key")
        self.key, self.config = key, dict(CONFIG if config is None else config)

    def fit(self, frame, train, *, seed=42, device="cpu"):
        started = perf_counter()
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        windows, indices = history_windows(frame, train.timestamp)
        # 同一历史观测只计一次，重叠窗口不能把末端重复24倍拟合scaler。
        unique = np.unique((indices[:, None] - np.arange(LOOKBACK)).ravel())
        raw = observation_features(frame)[unique]
        self.mean, self.scale = raw.mean(axis=0), raw.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        target = train.target_power.to_numpy(float)
        if self.key == "transformer_delta":
            target = target - windows[:, -1, 0]
        self.target_mean, self.target_scale = float(target.mean()), max(float(target.std()), 1e-8)
        inputs = torch.tensor((windows - self.mean) / self.scale, dtype=torch.float32)
        targets = torch.tensor((target - self.target_mean) / self.target_scale, dtype=torch.float32)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=self.config["batch_size"],
                            shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        net = HistoryEncoder(self.config).to(device)
        before = net.projection.weight.detach().cpu().clone()
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.config["learning_rate"],
                                     weight_decay=self.config["weight_decay"])
        losses, gradient_max = [], 0.0
        for epoch in range(self.config["epochs"]):
            net.train()
            total = 0.0
            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite_sequence_loss")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(net.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
                gradient_max = max(gradient_max, float(norm))
                optimizer.step()
                total += float(loss.detach()) * len(x)
            losses.append(total / len(train))
            print(f"{self.key} seed={seed} epoch={epoch + 1} train_mse={losses[-1]:.6f}", flush=True)
        if device == "cuda":
            torch.cuda.synchronize()
        self.net = net.cpu().eval()
        change = float((self.net.projection.weight.detach() - before).abs().max())
        if not change or not gradient_max:
            raise ValueError("sequence_parameters_not_updated")
        self.details = {"model_key": self.key, "config": self.config, "seed": seed,
                        "device": device, "fit_elapsed_seconds": perf_counter() - started,
                        "epochs": len(losses), "train_loss": losses, "parameter_max_change": change,
                        "max_gradient_norm": gradient_max,
                        "parameters": sum(p.numel() for p in self.net.parameters()),
                        "preprocessing": "sequence_train_only_standardization",
                        "target_representation": "increment_from_current" if self.key.endswith("delta") else "direct_power",
                        "scaler_observations": len(unique), "scaler_last_time": frame.timestamp.iloc[unique[-1]].isoformat()}
        return self

    def predict_windows(self, windows):
        self.net.eval()
        values = torch.tensor((windows - self.mean) / self.scale, dtype=torch.float32)
        with torch.inference_mode():
            output = torch.cat([self.net(batch) for batch in values.split(512)]).numpy().astype(float)
        output = output * self.target_scale + self.target_mean
        if self.key == "transformer_delta":
            output += windows[:, -1, 0]
        return np.maximum(output, 0.0)

    def predict_frame(self, frame, cutoffs):
        return self.predict_windows(history_windows(frame, cutoffs)[0])

    def save(self, path):
        torch.save({"key": self.key, "config": self.config, "state_dict": self.net.state_dict(),
                    "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                    "target_mean": self.target_mean, "target_scale": self.target_scale,
                    "details": self.details}, path)

    @classmethod
    def load(cls, path):
        # 只还原权重与简单元数据，不依赖pickle中的可执行模型类或训练代码。
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        item = cls(checkpoint["key"], checkpoint["config"])
        item.net = HistoryEncoder(item.config)
        item.net.load_state_dict(checkpoint["state_dict"], strict=True)
        item.net.eval()
        item.mean, item.scale = np.array(checkpoint["mean"]), np.array(checkpoint["scale"])
        item.target_mean, item.target_scale = checkpoint["target_mean"], checkpoint["target_scale"]
        item.details = checkpoint["details"]
        return item
