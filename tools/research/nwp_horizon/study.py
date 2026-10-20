"""气象信息按时距的价值：SCADA / +GFS / +ERA5先知 三臂对照，事前冻结后一次评价。

复用 ENGIE 合同的数据准入与 60 维特征（engie_contract.load_source / make_batch）和
A2/A3 的 LightGBM L1 参数（engie_l1.L1_PARAMETERS，seed 42）。三臂只差特征集合。

阶段：
- freeze：写 protocol.json（设计、切分、特征、指标、措辞、代码与输入 hash）；不训练。
- run：核对冻结未变 → 内层早停选树数并封存 → 重拟合 → 才构造测试期标签并评分。
- report：从 result.json 生成图（SVG）与 Markdown 表；不训练、不重算。

用法（项目根目录）：
    python tools/research/nwp_horizon/study.py freeze --source-zip <la_haute_borne.zip>
    python tools/research/nwp_horizon/study.py run --source-zip <la_haute_borne.zip>
    python tools/research/nwp_horizon/study.py report
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from lightgbm import LGBMRegressor, early_stopping  # noqa: E402

from power_forecast_service.forecasting.engie_contract import (  # noqa: E402
    ARRIVAL_LAG, FEATURES, LOOKBACK, ROSTER, STEP, load_source, make_batch,
)
from power_forecast_service.forecasting.engie_l1 import L1_PARAMETERS  # noqa: E402
from timing import (  # noqa: E402
    MAX_FALLBACK_RUNS, PUBLICATION_DELAY, assert_no_future_information, bracket,
    fit_power_law_alpha, hub_speed, qualifying_runs,
)

OUTPUT = ROOT / "docs/results/nwp-horizon-2026-10-20"
INPUTS = OUTPUT / "inputs"
VERSION = "nwp-horizon-v1"
HORIZONS = (10, 20, 30, 60, 120, 180, 240)  # 分钟；15 min 夹在 10 与 20 之间，不另造
TRAIN_START = pd.Timestamp("2015-01-15", tz="UTC")
TEST_START = pd.Timestamp("2015-09-01", tz="UTC")
TEST_END = pd.Timestamp("2016-01-01", tz="UTC")  # 不含
INNER_DAYS = 28
CAPACITY_KW = 8200.0  # 4 x 2050 kW
ARMS = ("scada", "gfs", "era5")
WEATHER = ("ws_hub", "dir_sin", "dir_cos", "ws10", "gust", "t2m", "sp")
BOOTSTRAP = {"block_steps": 1008, "samples": 2000, "seed": 42, "confidence": 0.95}
CODE = ("tools/research/nwp_horizon/study.py", "tools/research/nwp_horizon/timing.py",
        "tools/research/nwp_horizon/build_inputs.py", "tools/research/nwp_horizon/fetch_gfs.py",
        "tools/research/nwp_horizon/fetch_era5.py",
        "src/power_forecast_service/forecasting/engie_contract.py",
        "src/power_forecast_service/forecasting/engie_l1.py",
        "src/power_forecast_service/forecasting/engie_baselines.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    data = Path(path).read_bytes()
    return {"path": Path(path).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(data).hexdigest(), "bytes": len(data)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 气象输入

def load_gfs():
    frame = pd.read_csv(INPUTS / "gfs_site_2015.csv")
    frame["reftime"] = pd.to_datetime(frame.reftime, utc=True)
    return {(row.reftime, int(row.lead_h)): row for row in frame.itertuples(index=False)}


def gfs_features(issues, horizons, table):
    """每个 (起报, 时距) 选最新合格运行；缺文件按顺序回退。返回特征、有效掩码与使用记录。

    features: [issue, horizon, 7]；lead_min/fallback: [issue, horizon]；未找到时为 NaN/-1。
    """
    features = np.full((len(issues), len(horizons), len(WEATHER)), np.nan)
    lead_min = np.full((len(issues), len(horizons)), np.nan)
    fallback = np.full((len(issues), len(horizons)), -1, dtype=int)
    for i, issue in enumerate(issues):
        runs = qualifying_runs(issue)
        for k, minutes in enumerate(horizons):
            valid = issue + pd.Timedelta(minutes=minutes)
            for depth, run in enumerate(runs):
                assert_no_future_information(issue, run)  # 泄漏守卫
                b = bracket(run, valid)
                low, high = table.get((run, b.lower)), table.get((run, b.upper))
                if low is None or high is None:
                    continue
                # 时间线性插值在分量上做，然后计算风速与风向
                mix = {name: (1 - b.weight) * getattr(low, name) + b.weight * getattr(high, name)
                       for name in ("u10", "v10", "u80", "v80", "gust", "t2m", "sp")}
                features[i, k] = _weather_vector(mix["u80"], mix["v80"], mix["u10"], mix["v10"],
                                                 mix["gust"], mix["t2m"], mix["sp"])
                lead_min[i, k] = b.lead / pd.Timedelta(minutes=1)
                fallback[i, k] = depth
                break
    return features, np.isfinite(features).all(axis=2), lead_min, fallback


def _weather_vector(u_hub, v_hub, u10, v10, gust, t2m, sp, hub_scale=1.0):
    """风向用来向的 sin/cos（u/v 为去向分量）；hub_scale 只缩放轮毂风速。"""
    speed = np.hypot(u_hub, v_hub)
    direction = np.arctan2(-u_hub, -v_hub)
    return np.array([speed * hub_scale, np.sin(direction), np.cos(direction),
                     np.hypot(u10, v10), gust, t2m, sp], dtype=float)


def load_era5():
    frame = pd.read_csv(INPUTS / "era5_site_2015.csv")
    frame.index = pd.to_datetime(frame.time, utc=True)
    return frame.drop(columns="time")


def era5_alpha(era5):
    """风切变指数只用训练期（测试开始前）的小时拟合。"""
    train = era5.loc[(era5.index >= TRAIN_START) & (era5.index < TEST_START)]
    return fit_power_law_alpha(np.hypot(train.u10, train.v10), np.hypot(train.u100, train.v100))


def era5_features(issues, horizons, era5, alpha):
    """先知：目标有效时刻的 ERA5，逐时值线性插值到 10 分钟。不是预报。"""
    valid = issues.asi8[:, None] + np.array(horizons)[None, :] * pd.Timedelta(minutes=1).value
    flat = pd.to_datetime(valid.ravel(), utc=True)
    hourly = era5.astype(float)
    # 先在完整 10 分钟日历上插值（只在两端观测之间），再按有效时刻取值
    grid = pd.date_range(hourly.index[0], hourly.index[-1], freq=STEP)
    dense = hourly.reindex(hourly.index.union(grid)).interpolate(method="time", limit_area="inside")
    position = dense.index.get_indexer(flat)
    values = dense.iloc[np.maximum(position, 0)].reset_index(drop=True)
    values.loc[position < 0] = np.nan
    speed = hub_speed(np.hypot(values.u100, values.v100), alpha)
    direction = np.arctan2(-values.u100.to_numpy(), -values.v100.to_numpy())
    out = np.column_stack([speed, np.sin(direction), np.cos(direction),
                           np.hypot(values.u10, values.v10), values.gust10, values.t2m, values.sp])
    out = out.reshape(len(issues), len(horizons), len(WEATHER))
    return out, np.isfinite(out).all(axis=2)


# ---------------------------------------------------------------- 标签与样本

def targets(source, issues, horizons, *, label_end):
    """[issue, turbine, horizon] 原始 kW；冲突/缺测为 NaN，不裁剪负值。

    有效时刻 >= label_end 的标签不读取（泄漏守卫：训练期末尾起报的长时距目标
    会落进测试期，这些值在封存前一律不进入数组）。
    """
    valid = issues.asi8[:, None] + np.array(horizons)[None, :] * pd.Timedelta(minutes=1).value
    index = source.times.get_indexer(pd.to_datetime(valid.ravel(), utc=True)).reshape(valid.shape)
    index[valid >= pd.Timestamp(label_end).value] = -1
    out = np.full((len(issues), len(ROSTER), len(horizons)), np.nan)
    rows, cols = np.nonzero(index >= 0)
    out[rows, :, cols] = source.values[index[rows, cols], :, 0]
    return out


def label_available(issues, horizons):
    """(起报, 时距) 标签在 issue + h + 20 min 到达；返回 UTC 纳秒整数 [issue, horizon]。"""
    return issues.asi8[:, None] + (np.array(horizons, dtype=np.int64)[None, :] * 60_000_000_000
                                   + ARRIVAL_LAG.value)


def design(batch, weather, horizons, turbine):
    """单台机组的跨时距合并设计矩阵 [issue, horizon, 60 + 1 (+7)]。

    列顺序：自身 12 条历史 x 5 字段（与 ENGIE 合同一致）、horizon_min、气象 7 维。
    气象是站点量，四台共用同一值。
    """
    n, h = len(batch.issues), len(horizons)
    base = np.broadcast_to(batch.features[:, None, turbine], (n, h, batch.features.shape[2]))
    horizon = np.broadcast_to(np.array(horizons, float)[None, :, None], (n, h, 1))
    parts = [base, horizon] if weather is None else [base, horizon, weather]
    return np.concatenate(parts, axis=2)


def persistence_kw(batch, horizons):
    """最新可得功率（起报前 20 分钟）原值，四台各自保持。[issue, turbine, horizon]。"""
    last = batch.features[:, :, (LOOKBACK - 1) * len(FEATURES)]
    return np.repeat(last[:, :, None], len(horizons), axis=2)


# ---------------------------------------------------------------- 训练

def fit_arm(batch, weather, y, train, validation, refit):
    """内层早停选树数 → 用全部训练期重拟合；返回 4 个模型与选择记录。

    y: [issue, turbine, horizon]（测试期为 NaN）；掩码 [issue, horizon]，行 = (起报, 时距)。
    """
    models, detail = [], []
    for t in range(len(ROSTER)):
        xt, yt = design(batch, weather, HORIZONS, t), y[:, t]
        started = perf_counter()
        model = LGBMRegressor(**L1_PARAMETERS)
        model.fit(xt[train], yt[train], eval_set=[(xt[validation], yt[validation])],
                  callbacks=[early_stopping(40, first_metric_only=True, verbose=False)])
        iterations = int(model.best_iteration_ or L1_PARAMETERS["n_estimators"])
        final = LGBMRegressor(**{**L1_PARAMETERS, "n_estimators": iterations})
        final.fit(xt[refit], yt[refit])
        models.append(final)
        detail.append({"turbine": ROSTER[t], "iterations": iterations,
                       "inner_rows": int(train.sum()), "validation_rows": int(validation.sum()),
                       "refit_rows": int(refit.sum()), "seconds": perf_counter() - started})
    return models, detail


def predict_arm(models, batch, weather, mask):
    out = np.full((len(batch.issues), len(ROSTER), len(HORIZONS)), np.nan)
    for t, model in enumerate(models):
        xt = design(batch, weather, HORIZONS, t)
        column = out[:, t]
        column[mask] = model.booster_.predict(xt[mask], num_threads=2)
    return out


# ---------------------------------------------------------------- 指标与区间

def horizon_metrics(actual, predicted, mask):
    """四台合计、逐时距：MAE、RMSE、偏差、准确率 1 - RMSE / Cap。"""
    result = {}
    farm_y, farm_p = actual.sum(axis=1), predicted.sum(axis=1)
    for k, minutes in enumerate(HORIZONS):
        keep = mask[:, k]
        error = farm_p[keep, k] - farm_y[keep, k]
        if not np.isfinite(error).all():
            raise ValueError("nwp_scored_key_nonfinite")
        rmse = float(np.sqrt(np.square(error).mean()))
        result[str(minutes)] = {"keys": int(keep.sum()), "mae_kw": float(np.abs(error).mean()),
                                "rmse_kw": rmse, "bias_kw": float(error.mean()),
                                "accuracy": 1.0 - rmse / CAPACITY_KW}
    return result


def block_bootstrap(losses, mask, *, block_steps, samples, seed):
    """非循环配对时间块：同一组起始索引作用于所有臂与所有时距；缺测位置保留在日历上。

    losses: {arm: [issue, horizon] 绝对误差}；mask: [issue, horizon] 共同评分键。
    返回每个时距各臂的重抽样 MAE，形状 [samples]。
    """
    count = mask.shape[0]
    rng = np.random.default_rng(seed)
    lengths = np.full(int(np.ceil(count / block_steps)), block_steps, dtype=int)
    lengths[-1] = count - int(lengths[:-1].sum())
    starts = rng.integers(0, count - block_steps + 1, size=(samples, len(lengths)))
    ends = starts + lengths
    replicates = {}
    for k in range(mask.shape[1]):
        keep = mask[:, k]
        prefix = np.r_[0, np.cumsum(keep)]
        denominators = (prefix[ends] - prefix[starts]).sum(axis=1)
        if (denominators == 0).any():
            raise ValueError("nwp_bootstrap_empty_resample")
        for arm, values in losses.items():
            sums = np.r_[0.0, np.cumsum(np.where(keep, values[:, k], 0.0))]
            totals = (sums[ends] - sums[starts]).sum(axis=1)
            replicates.setdefault(arm, []).append(totals / denominators)
    return {arm: np.stack(v, axis=1) for arm, v in replicates.items()}  # [samples, horizon]


def wording(low, high, positive, negative):
    """事前冻结的措辞：区间完全低于 0 才称改善。"""
    if high < 0:
        return positive
    if low > 0:
        return negative
    return "未检出差异（区间含0）"


# ---------------------------------------------------------------- 冻结

def protocol():
    return {
        "version": VERSION,
        "question": ("At which forecast horizons does weather information add skill for aggregate "
                     "wind-farm power, and how much of the oracle upper bound does a genuinely "
                     "available forecast capture?"),
        "horizons_min": list(HORIZONS),
        "horizon_mapping": {"15min": "between 10 and 20 (both reported)", "30min": 30, "1h": 60,
                            "2h": 120, "3h": 180, "4h": 240},
        "arms": {
            "persistence": "latest available power (issue - 20 min) per turbine, summed; reference",
            "scada": "existing 60-dim own-turbine history + horizon_min",
            "gfs": "scada + GFS " + ",".join(WEATHER) + " at target valid time",
            "era5": "scada + ERA5 " + ",".join(WEATHER) + " at target valid time; ORACLE upper bound, not a forecast",
        },
        "model": {"family": "LightGBM regression_l1, one pooled-horizon model per turbine, farm = sum",
                  "parameters": L1_PARAMETERS, "early_stopping_rounds": 40,
                  "selection": "inner early stopping on last 28 training days, then refit all training rows with selected trees",
                  "no_tuning_on_test": True},
        "gfs_rule": {"publication_delay_hours": PUBLICATION_DELAY / pd.Timedelta(hours=1),
                     "eligibility": "reftime + delay <= issue", "choice": "latest eligible run",
                     "fallback": f"previous eligible runs, at most {MAX_FALLBACK_RUNS}, if a needed step file is missing",
                     "time_interpolation": "linear between 3 h steps of the same run, on u/v components",
                     "space_interpolation": "bilinear over enclosing 0.25 deg cell, on u/v components",
                     "hub_wind": "GFS 80 m u/v directly (hub height 80 m)",
                     "temperature": "2 m", "gust": "surface gust", "pressure": "surface"},
        "era5_rule": {"source": "Open-Meteo ERA5 archive, 4 enclosing grid points, bilinear",
                      "time_interpolation": "linear between hourly values",
                      "hub_wind": "100 m speed * (80/100)^alpha, alpha = median log-ratio of 100 m/10 m speed on hours before test start",
                      "direction": "100 m"},
        "split": {"train": [TRAIN_START.isoformat(), TEST_START.isoformat()],
                  "inner_validation_days": INNER_DAYS,
                  "test": [TEST_START.isoformat(), TEST_END.isoformat()],
                  "label_rule": "row (issue,h) enters a training set only if issue + h + 20 min < boundary",
                  "exposure": "2015 was the A3 final evaluation year; not a blind test of earlier models"},
        "rows": ("identical keys for all arms: label complete for all 4 turbines AND SCADA, GFS and "
                 "ERA5 inputs all valid; applies to training and scoring; dropped counts reported"),
        "metrics": {"per_horizon": ["mae_kw", "rmse_kw", "bias_kw", "accuracy = 1 - RMSE / 8200 kW"],
                    "accuracy_note": "approximates, does not replicate, Chinese provincial assessment rules",
                    "improvement": "MAE % vs persistence and vs scada arm",
                    "bootstrap": {**BOOTSTRAP, "method": "paired non-circular 7-day blocks over full test calendar, percentile CI",
                                  "differences": ["gfs - scada", "era5 - scada", "scada - persistence",
                                                  "gfs - persistence", "era5 - persistence"]},
                    "capture": "(MAE_scada - MAE_gfs) / (MAE_scada - MAE_era5), only where era5-scada CI < 0"},
        "decision_wording": {
            "gfs": "'GFS 在 h 显著改善' only if the 95% CI of MAE(gfs) - MAE(scada) is entirely < 0; "
                   "'GFS 在 h 显著变差' if entirely > 0; else '未检出差异（区间含0）'",
            "era5": "same rule with '先知 ERA5' wording; never called a forecast",
            "summary": "list the horizons satisfying each statement; no pooled significance claim",
        },
        "inputs": {name: file_record(INPUTS / name)
                   for name in ("gfs_site_2015.csv", "era5_site_2015.csv", "sources.json")},
        "code": [file_record(ROOT / name) for name in CODE],
        "environment": {n: version(n) for n in ("numpy", "pandas", "lightgbm")},
    }


def checked_protocol():
    frozen = read_json(OUTPUT / "protocol.json")
    if frozen["protocol"] != protocol():
        raise SystemExit("nwp_protocol_changed_after_freeze")
    return sha256((OUTPUT / "protocol.json").read_bytes()).hexdigest()


# ---------------------------------------------------------------- 运行

def prepare(source_zip):
    source = load_source(Path(source_zip))
    issues = pd.date_range(TRAIN_START, TEST_END, freq=STEP, inclusive="left")
    # make_batch 的 label_end 只影响其 10-60 min 目标；本研究自行构造长时距标签
    batch = make_batch(source, issues, label_end=TRAIN_START)
    gfs, gfs_ok, lead_min, fallback = gfs_features(issues, HORIZONS, load_gfs())
    era5 = load_era5()
    alpha = era5_alpha(era5)
    oracle, era5_ok = era5_features(issues, HORIZONS, era5, alpha)
    return source, batch, {"gfs": gfs, "era5": oracle}, {
        "scada": np.repeat(batch.input_valid[:, None], len(HORIZONS), axis=1),
        "gfs": gfs_ok, "era5": era5_ok}, lead_min, fallback, alpha


def coverage(mask_by, label_ok, period):
    common = label_ok & mask_by["scada"] & mask_by["gfs"] & mask_by["era5"]
    out = {}
    for k, minutes in enumerate(HORIZONS):
        p = period
        out[str(minutes)] = {
            "planned": int(p.sum()), "label_complete": int((p & label_ok[:, k]).sum()),
            **{f"{arm}_inputs_valid": int((p & mask_by[arm][:, k]).sum()) for arm in ARMS},
            "common_scored": int((p & common[:, k]).sum()),
        }
    return out


def run(args):
    frozen = checked_protocol()
    if (OUTPUT / "result.json").exists() or (OUTPUT / "test-exposure-started.json").exists():
        raise SystemExit("nwp_already_evaluated_results_are_final")
    source, batch, weather, valid, lead_min, fallback, alpha = prepare(args.source_zip)
    issues = batch.issues
    train_period = issues < TEST_START
    # 训练阶段只构造训练期起报的标签；测试期标签在模型封存之后才读取
    y_train = np.full((len(issues), len(ROSTER), len(HORIZONS)), np.nan)
    y_train[train_period] = targets(source, issues[train_period], HORIZONS, label_end=TEST_START)
    label_ok = np.isfinite(y_train).all(axis=1)
    inputs_ok = valid["scada"] & valid["gfs"] & valid["era5"]
    arrival = label_available(issues, HORIZONS)
    inner_start = TEST_START - pd.Timedelta(days=INNER_DAYS)
    in_train = train_period[:, None]
    base = in_train & label_ok & inputs_ok
    refit = base & (arrival < TEST_START.value)
    inner_train = base & (arrival < inner_start.value)
    validation = refit & (issues >= inner_start)[:, None]
    selection, models = {"freeze_sha256": frozen, "started_at": now(), "alpha": alpha}, {}
    for arm in ARMS:
        models[arm], selection[arm] = fit_arm(batch, None if arm == "scada" else weather[arm],
                                              y_train, inner_train, validation, refit)
        print(arm, [d["iterations"] for d in selection[arm]], flush=True)
    selection["sealed_at"] = now()
    write_json(OUTPUT / "selection.json", selection)
    write_json(OUTPUT / "test-exposure-started.json",
               {"freeze_sha256": frozen, "selection_sha256": file_record(OUTPUT / "selection.json")["sha256"],
                "started_at": now()})
    period = ~train_period
    y_all = y_train.copy()
    y_all[period] = targets(source, issues[period], HORIZONS, label_end=TEST_END)
    label_ok = np.isfinite(y_all).all(axis=1)
    common = period[:, None] & label_ok & inputs_ok
    predictions = {"persistence": persistence_kw(batch, HORIZONS)}
    for arm in ARMS:
        predictions[arm] = predict_arm(models[arm], batch, None if arm == "scada" else weather[arm],
                                       common)
    rows = slice(int(np.argmax(period)), None)
    y, mask = y_all[rows], common[rows]
    metrics = {arm: horizon_metrics(y, p[rows], mask) for arm, p in predictions.items()}
    losses = {arm: np.abs(p[rows].sum(axis=1) - y.sum(axis=1)) for arm, p in predictions.items()}
    reps = block_bootstrap(losses, mask, **{k: BOOTSTRAP[k] for k in ("block_steps", "samples", "seed")})
    pairs = {"gfs-scada": ("gfs", "scada"), "era5-scada": ("era5", "scada"),
             "scada-persistence": ("scada", "persistence"), "gfs-persistence": ("gfs", "persistence"),
             "era5-persistence": ("era5", "persistence")}
    intervals = {}
    for name, (a, b) in pairs.items():
        diff = reps[a] - reps[b]
        gain = 100 * (1 - reps[a] / reps[b])
        intervals[name] = {str(h): {
            "delta_mae_kw": metrics[a][str(h)]["mae_kw"] - metrics[b][str(h)]["mae_kw"],
            "delta_mae_kw_95ci": np.quantile(diff[:, k], [0.025, 0.975]).tolist(),
            "gain_percent": 100 * (1 - metrics[a][str(h)]["mae_kw"] / metrics[b][str(h)]["mae_kw"]),
            "gain_percent_95ci": np.quantile(gain[:, k], [0.025, 0.975]).tolist(),
        } for k, h in enumerate(HORIZONS)}
    statements, capture = {}, {}
    for k, h in enumerate(HORIZONS):
        g = intervals["gfs-scada"][str(h)]["delta_mae_kw_95ci"]
        e = intervals["era5-scada"][str(h)]["delta_mae_kw_95ci"]
        statements[str(h)] = {"gfs": wording(*g, "GFS 显著改善", "GFS 显著变差"),
                              "era5": wording(*e, "先知 ERA5 显著改善", "先知 ERA5 显著变差")}
        if e[1] < 0:
            ratio = (reps["scada"][:, k] - reps["gfs"][:, k]) / (reps["scada"][:, k] - reps["era5"][:, k])
            point = ((metrics["scada"][str(h)]["mae_kw"] - metrics["gfs"][str(h)]["mae_kw"])
                     / (metrics["scada"][str(h)]["mae_kw"] - metrics["era5"][str(h)]["mae_kw"]))
            capture[str(h)] = {"point": point, "95ci": np.quantile(ratio, [0.025, 0.975]).tolist()}
        else:
            capture[str(h)] = None
    test_lead = lead_min[rows][mask]
    test_fallback = fallback[rows][mask]
    result = {
        "version": VERSION, "freeze_sha256": frozen, "completed_at": now(),
        "era5_alpha": alpha,
        "coverage": {"train_refit_rows_per_turbine": int(refit.sum()),
                     "train_period_dropped_for_missing_inputs": int((in_train & label_ok & ~inputs_ok).sum()),
                     "test": coverage(valid, label_ok, period)},
        "gfs_lead_used_test": {"min_minutes": float(test_lead.min()), "max_minutes": float(test_lead.max()),
                               "mean_minutes": float(test_lead.mean()),
                               "fallback_depth_counts": {str(d): int((test_fallback == d).sum())
                                                         for d in range(MAX_FALLBACK_RUNS + 1)}},
        "metrics": metrics, "intervals": intervals, "statements": statements, "capture": capture,
    }
    write_json(OUTPUT / "result.json", result)
    print(json.dumps({h: {a: round(metrics[a][h]["mae_kw"], 1) for a in metrics} for h in metrics["scada"]}))


# ---------------------------------------------------------------- 报告

def svg_figure(result, path):
    """无第三方绘图依赖的折线图：MAE 相对 SCADA 臂的改善百分比及 95% 区间。"""
    width, height, left, right, top, bottom = 640, 360, 70, 20, 30, 50
    hs = [str(h) for h in HORIZONS]
    series = {"gfs-scada": ("#1f77b4", "(b) +GFS"), "era5-scada": ("#d62728", "(c) +ERA5 先知")}
    values = [v for name in series for h in hs
              for v in (*result["intervals"][name][h]["gain_percent_95ci"],
                        result["intervals"][name][h]["gain_percent"])]
    lo, hi = min(min(values), -1.0), max(max(values), 1.0)
    lo, hi = min(lo, 0), max(hi, 0)
    lo, hi = np.floor(lo) - 1, np.ceil(hi) + 1

    def x(i):
        return left + i * (width - left - right) / (len(hs) - 1)

    def y(v):
        return top + (hi - v) * (height - top - bottom) / (hi - lo)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="sans-serif" font-size="12">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<line x1="{left}" y1="{y(0):.1f}" x2="{width - right}" y2="{y(0):.1f}" stroke="#888"/>']
    for tick in np.linspace(lo, hi, 6):
        parts.append(f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for i, h in enumerate(hs):
        parts.append(f'<text x="{x(i):.1f}" y="{height - bottom + 18}" text-anchor="middle">{h}</text>')
    parts.append(f'<text x="{(left + width - right) / 2}" y="{height - 10}" text-anchor="middle">'
                 f'时距（分钟，非等距排列）</text>')
    parts.append(f'<text x="16" y="{(top + height - bottom) / 2}" transform="rotate(-90 16 '
                 f'{(top + height - bottom) / 2})" text-anchor="middle">MAE 相对 (a) SCADA 改善 %</text>')
    for j, (name, (color, label)) in enumerate(series.items()):
        points = []
        for i, h in enumerate(hs):
            row = result["intervals"][name][h]
            # gain_percent 定义为 100*(1 - MAE_arm/MAE_ref)；表示为“改善为正”
            g = row["gain_percent"]
            a, b = row["gain_percent_95ci"]
            off = (j - 0.5) * 6
            parts.append(f'<line x1="{x(i) + off:.1f}" y1="{y(a):.1f}" x2="{x(i) + off:.1f}" '
                         f'y2="{y(b):.1f}" stroke="{color}" stroke-width="2"/>')
            points.append(f"{x(i) + off:.1f},{y(g):.1f}")
            parts.append(f'<circle cx="{x(i) + off:.1f}" cy="{y(g):.1f}" r="3" fill="{color}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}"/>')
        parts.append(f'<text x="{left + 10 + 140 * j}" y="{top - 10}" fill="{color}">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def report():
    result = read_json(OUTPUT / "result.json")
    svg_figure(result, OUTPUT / "improvement-vs-horizon.svg")
    m, iv = result["metrics"], result["intervals"]
    lines = ["| 时距 | 持久性 MAE | (a) MAE | (b) MAE | (c) MAE | 准确率 p / a / b / c | "
             "(a) 较持久性 | (b)−(a) kW [95%CI] | (c)−(a) kW [95%CI] | 捕获比 |",
             "|---|---:|---:|---:|---:|---|---:|---|---|---:|"]
    for h in (str(v) for v in HORIZONS):
        acc = " / ".join(f"{100 * m[a][h]['accuracy']:.1f}%" for a in ("persistence", *ARMS))
        g, e = iv["gfs-scada"][h], iv["era5-scada"][h]
        cap = result["capture"][h]
        cap_text = "—" if cap is None else f"{cap['point']:.2f} [{cap['95ci'][0]:.2f}, {cap['95ci'][1]:.2f}]"
        lines.append(
            f"| {h} min | {m['persistence'][h]['mae_kw']:.1f} | {m['scada'][h]['mae_kw']:.1f} | "
            f"{m['gfs'][h]['mae_kw']:.1f} | {m['era5'][h]['mae_kw']:.1f} | {acc} | "
            f"{iv['scada-persistence'][h]['gain_percent']:+.1f}% | "
            f"{g['delta_mae_kw']:+.1f} [{g['delta_mae_kw_95ci'][0]:+.1f}, {g['delta_mae_kw_95ci'][1]:+.1f}] | "
            f"{e['delta_mae_kw']:+.1f} [{e['delta_mae_kw_95ci'][0]:+.1f}, {e['delta_mae_kw_95ci'][1]:+.1f}] | "
            f"{cap_text} |")
    print("\n".join(lines))
    for h, s in result["statements"].items():
        print(h, s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "report"))
    parser.add_argument("--source-zip", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.phase == "freeze":
        if (OUTPUT / "protocol.json").exists():
            raise SystemExit("nwp_already_frozen")
        write_json(OUTPUT / "protocol.json", {"frozen_at": now(), "protocol": protocol()})
        print("protocol frozen; no model trained, no test metric computed")
    elif args.phase == "run":
        if args.source_zip is None:
            raise SystemExit("--source-zip is required")
        run(args)
    else:
        report()


if __name__ == "__main__":
    main()
