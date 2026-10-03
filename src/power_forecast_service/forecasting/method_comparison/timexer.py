"""TimeXer作者实现的单目标适配：历史目标patch与外生变量交互。

源自THUML/Time-Series-Library，revision见protocol.py；许可见LICENSE-TimeXer.txt。
保留原层次、初始化、归一化和dropout；移除未使用的多目标/任务分支与通用loader。
输入已经重排为功率末列。输出是一项t+12回归，时间语义由本地标签合同负责。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        position = torch.arange(5000).float().unsqueeze(1)
        divisor = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.0) / width))
        pe = torch.zeros(5000, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FullAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) / math.sqrt(queries.shape[-1])
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", weights, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.inner_attention = FullAttention(dropout)
        self.query_projection = nn.Linear(width, width)
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.out_projection = nn.Linear(width, width)
        self.n_heads = heads

    def forward(self, queries, keys, values):
        batch, length, width = queries.shape
        q = self.query_projection(queries).view(batch, length, self.n_heads, -1)
        k = self.key_projection(keys).view(batch, keys.shape[1], self.n_heads, -1)
        v = self.value_projection(values).view(batch, values.shape[1], self.n_heads, -1)
        out = self.inner_attention(q, k, v).view(batch, length, width)
        return self.out_projection(out)


class EnEmbedding(nn.Module):
    def __init__(self, width, patch, dropout):
        super().__init__()
        self.patch_len = patch
        self.value_embedding = nn.Linear(patch, width, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, 1, width))
        self.position_embedding = PositionalEmbedding(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [B,1,24] -> [B,4,32]，再附加一个汇聚外生信息的global token。
        glb = self.glb_token.repeat(x.shape[0], 1, 1, 1)
        x = x.unfold(-1, self.patch_len, self.patch_len)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.cat([x.reshape(glb.shape[0], 1, -1, x.shape[-1]), glb], dim=2)
        return self.dropout(x.reshape(-1, x.shape[-2], x.shape[-1]))


class InvertedEmbedding(nn.Module):
    def __init__(self, length, width, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(length, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x.permute(0, 2, 1)))


class EncoderLayer(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.self_attention = AttentionLayer(width, heads, dropout)
        self.cross_attention = AttentionLayer(width, heads, dropout)
        self.conv1 = nn.Conv1d(width, feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(feedforward, width, kernel_size=1)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x)))
        # 单目标模式下每个batch只有一个global token，不跨样本拼接变量。
        global_token = x[:, -1:, :]
        global_token = self.norm2(global_token + self.dropout(
            self.cross_attention(global_token, cross, cross)))
        x = torch.cat([x[:, :-1, :], global_token], dim=1)
        y = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Encoder(nn.Module):
    def __init__(self, width, heads, layers, feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(width, heads, feedforward, dropout)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        return self.norm(x)


class FlattenHead(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class TimeXer(nn.Module):
    def __init__(self, length=24, width=32, heads=4, layers=2, feedforward=64,
                 patch=6, dropout=0.1):
        super().__init__()
        if length % patch or width % heads:
            raise ValueError("timexer_incomplete_patch_or_heads")
        self.length = length
        self.en_embedding = EnEmbedding(width, patch, dropout)
        self.ex_embedding = InvertedEmbedding(length, width, dropout)
        self.encoder = Encoder(width, heads, layers, feedforward, dropout)
        self.head = FlattenHead(width * (length // patch + 1), dropout)

    def forward(self, values):
        if values.ndim != 3 or values.shape[1:] != (self.length, 8):
            raise ValueError("timexer_input_shape")
        means = values.mean(1, keepdim=True).detach()
        centered = values - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = centered / stdev
        target = self.en_embedding(normalized[:, :, -1:].permute(0, 2, 1))
        external = self.ex_embedding(normalized[:, :, :-1])
        encoded = self.encoder(target, external)
        prediction = self.head(encoded.transpose(1, 2))
        return prediction[:, 0] * stdev[:, 0, -1] + means[:, 0, -1]
