"""GFS 起报可得性、时效插值与轮毂高度换算的纯规则；不读网络、不读数据文件。

时间一律为 tz-aware UTC。三个时间对象必须分开：
- issue：预测起报时刻 t，模型只能使用此刻已经“发布”的信息；
- reftime：GFS 一次运行的初始时刻（00/06/12/18 UTC）；
- valid：目标有效时刻 v = t + horizon，也是 reftime + lead。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

RUN_INTERVAL = pd.Timedelta(hours=6)
STEP = pd.Timedelta(hours=3)  # 本研究只取 0.25° 3 小时步长的输出
PUBLICATION_DELAY = pd.Timedelta(hours=5)  # 假设值：reftime + 5h 之后才视为可用
MAX_FALLBACK_RUNS = 2  # 最新合格运行缺失时，最多再往前退两次运行
FIRST_RUN = pd.Timestamp("2015-01-15T00:00Z")  # RDA d084001 0.25° 的首个可用运行

# 站点（OpenOA 资产表四台机组平均）与 GFS 0.25° 包围格点
SITE_LAT, SITE_LON = 48.452, 5.588
GRID_LATS = (48.25, 48.5)
GRID_LONS = (5.5, 5.75)
HUB_HEIGHT_M = 80.0


def _utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("nwp_time_requires_timezone")
    return value.tz_convert("UTC")


def latest_qualifying_run(issue, *, delay=PUBLICATION_DELAY, first_run=FIRST_RUN):
    """返回 reftime + delay <= issue 的最新运行；没有合格运行时返回 None。"""
    issue = _utc(issue)
    latest = (issue - delay).floor(RUN_INTERVAL)
    if latest < _utc(first_run):
        return None
    return latest


def qualifying_runs(issue, *, fallbacks=MAX_FALLBACK_RUNS, delay=PUBLICATION_DELAY,
                    first_run=FIRST_RUN):
    """按优先顺序列出可用运行：最新合格运行，然后最多 fallbacks 个更早运行。"""
    latest = latest_qualifying_run(issue, delay=delay, first_run=first_run)
    if latest is None:
        return []
    runs = [latest - k * RUN_INTERVAL for k in range(fallbacks + 1)]
    return [run for run in runs if run >= _utc(first_run)]


@dataclass(frozen=True)
class Bracket:
    """valid 时刻落在某次运行的两个 3h 输出步之间；weight 是后一步的线性权重。"""

    reftime: pd.Timestamp
    lead: pd.Timedelta  # valid - reftime，记录实际使用的时效
    lower: int  # 前一步预报时效（小时，3 的倍数）
    upper: int  # 后一步预报时效；恰好落在步长上时与 lower 相同
    weight: float


def bracket(reftime, valid):
    """reftime + lead = valid；在 3h 步之间按时间线性插值。lead 必须为正。"""
    reftime, valid = _utc(reftime), _utc(valid)
    lead = valid - reftime
    if lead <= pd.Timedelta(0):
        raise ValueError("nwp_lead_must_be_positive")
    hours = lead / pd.Timedelta(hours=1)
    lower = int(np.floor(hours / 3.0) * 3)
    upper = lower if hours == lower else lower + 3
    weight = 0.0 if upper == lower else (hours - lower) / 3.0
    return Bracket(reftime, lead, lower, upper, float(weight))


def assert_no_future_information(issue, reftime, *, delay=PUBLICATION_DELAY):
    """泄漏守卫：任何被使用的运行都必须在起报前已经发布。"""
    if _utc(reftime) + delay > _utc(issue):
        raise ValueError("nwp_run_not_published_at_issue_time")


def bilinear_weights(lat=SITE_LAT, lon=SITE_LON, lats=GRID_LATS, lons=GRID_LONS):
    """返回 [lat_index, lon_index] 的 2x2 权重；站点必须位于四个格点之内。"""
    fy = (lat - lats[0]) / (lats[1] - lats[0])
    fx = (lon - lons[0]) / (lons[1] - lons[0])
    if not (0 <= fy <= 1 and 0 <= fx <= 1):
        raise ValueError("nwp_site_outside_grid_cell")
    return np.array([[(1 - fy) * (1 - fx), (1 - fy) * fx], [fy * (1 - fx), fy * fx]])


def fit_power_law_alpha(speed_low, speed_high, *, z_low=10.0, z_high=100.0, min_speed=1.0):
    """只用训练期两层风速拟合常数风切变指数：alpha = median(ln(U_high/U_low)/ln(z_high/z_low))。

    低于 min_speed 的小时比值不稳定，排除；调用方负责只传入训练期数据。
    """
    low, high = np.asarray(speed_low, float), np.asarray(speed_high, float)
    keep = np.isfinite(low) & np.isfinite(high) & (low >= min_speed) & (high >= min_speed)
    if keep.sum() < 100:
        raise ValueError("nwp_alpha_insufficient_hours")
    return float(np.median(np.log(high[keep] / low[keep]) / np.log(z_high / z_low)))


def hub_speed(speed_ref, alpha, *, z_ref=100.0, z_hub=HUB_HEIGHT_M):
    """幂律 U(z_hub) = U(z_ref) * (z_hub / z_ref) ** alpha。"""
    return np.asarray(speed_ref, float) * (z_hub / z_ref) ** alpha
