"""ENGIE 原始 SCADA 的独立十分钟合同；不改变旧 Q1 的数据或服务对象。"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


VERSION = "engie-rolling-raw-kw-v1"
ROSTER = ("R80711", "R80721", "R80736", "R80790")
RAW_COLUMNS = ("P_avg", "Ws_avg", "Wa_avg", "Ot_avg")
FEATURES = ("power_kw", "wind_speed", "direction_sin", "direction_cos", "temperature")
STEP = pd.Timedelta(minutes=10)
ARRIVAL_LAG = pd.Timedelta(minutes=20)
LOOKBACK = 12
HORIZONS = (10, 20, 30, 40, 50, 60)
SCADA_MEMBER = "la-haute-borne-data-2014-2015.csv"
ARCHIVE_SHA256 = "be5ea66a3355286e491f5618250dc83e85252a8cb337748d7ba19edc50df6138"
MEMBER_SHA256 = "9be32aabe7e6b911f58ad3a9f292aed1e5b48cdc603b35d3feccb94f4c043cf4"
SOURCE_URL = (
    "https://raw.githubusercontent.com/NatLabRockies/OpenOA/"
    "9bfc7a3dc542b17bbfc06b692dbfb8b23c754975/examples/data/la_haute_borne.zip"
)
DEVELOPMENT_START = pd.Timestamp("2014-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2015-01-01", tz="UTC")
SOURCE_END = pd.Timestamp("2016-01-01", tz="UTC")
QUARTERS = {
    "2014-Q2": ("2014-04-01", "2014-07-01"),
    "2014-Q3": ("2014-07-01", "2014-10-01"),
    "2014-Q4": ("2014-10-01", "2015-01-01"),
}


@dataclass(frozen=True)
class SiteSource:
    # values: [source_time, turbine, raw_field]；冲突键不选择代表行，整键变为不可用。
    times: pd.DatetimeIndex
    values: np.ndarray
    conflicting: np.ndarray
    missing: np.ndarray
    quality: dict


@dataclass(frozen=True)
class RollingBatch:
    issues: pd.DatetimeIndex
    features: np.ndarray  # [issue, turbine, 12 * 5]，每台仅消费自身合法历史。
    targets: np.ndarray  # [issue, turbine, horizon]，保留期标签不进入此数组。
    input_valid: np.ndarray
    label_valid: np.ndarray
    boundary_valid: np.ndarray
    conflicting_input: np.ndarray
    unavailable_input: np.ndarray

    @property
    def scoreable(self):
        return self.input_valid & self.label_valid & self.boundary_valid

    def counts(self, mask):
        selected = np.asarray(mask, dtype=bool)
        total = int(selected.sum())
        scored = int((selected & self.scoreable).sum())
        return {
            "planned": total,
            "input_valid": int((selected & self.input_valid).sum()),
            "labels_complete": int((selected & self.label_valid).sum()),
            "protected_label_boundary": int((selected & ~self.boundary_valid).sum()),
            "conflicting_input": int((selected & self.conflicting_input).sum()),
            "unavailable_input": int((selected & self.unavailable_input).sum()),
            "scoreable": scored,
            "scoreable_fraction": scored / total if total else 0.0,
        }


def utc(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("engie_time_requires_timezone")
    return value.tz_convert("UTC")


def source_from_frame(raw, *, start, end):
    """显式建立 UTC 网格；保留缺数/冲突证据，不用排序或平均猜测源时钟。"""
    required = {"Wind_turbine_name", "Date_time", *RAW_COLUMNS}
    if not required.issubset(raw.columns):
        raise ValueError(f"engie_missing_columns:{sorted(required - set(raw.columns))}")
    if set(raw.Wind_turbine_name) != set(ROSTER):
        raise ValueError("engie_fixed_roster_mismatch")
    stamps = raw.Date_time.astype(str)
    if not stamps.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all():
        raise ValueError("engie_source_offset_required")
    times = pd.to_datetime(stamps, format="mixed", utc=True, errors="raise")
    grid = pd.date_range(utc(start), utc(end), freq=STEP, inclusive="left")
    if not times.isin(grid).all():
        raise ValueError("engie_timestamp_outside_declared_grid")
    data = raw[["Wind_turbine_name", *RAW_COLUMNS]].copy()
    data["time"] = times
    for column in RAW_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    keys = ["time", "Wind_turbine_name"]
    duplicates = data.duplicated(keys, keep=False)
    duplicate_keys = data.loc[duplicates, keys].drop_duplicates()
    unique = data.loc[~duplicates].set_index(keys)
    full_index = pd.MultiIndex.from_product([grid, ROSTER], names=keys)
    values = unique.reindex(full_index)[list(RAW_COLUMNS)].to_numpy(float)
    values = values.reshape(len(grid), len(ROSTER), len(RAW_COLUMNS))
    conflict_index = pd.MultiIndex.from_frame(duplicate_keys[keys])
    conflict_mask = full_index.isin(conflict_index).reshape(len(grid), len(ROSTER))
    observed_index = pd.MultiIndex.from_frame(data[keys].drop_duplicates())
    missing_mask = (~full_index.isin(observed_index)).reshape(len(grid), len(ROSTER))
    groups = []
    for turbine in ROSTER:
        rows = data.loc[data.Wind_turbine_name == turbine]
        unique_times = pd.DatetimeIndex(rows.time.unique()).sort_values()
        group = {
            "turbine": turbine, "source_rows": len(rows), "unique_keys": len(unique_times),
            "duplicate_keys": int((duplicate_keys.Wind_turbine_name == turbine).sum()),
            "backward_steps_in_source_order": int((rows.time.diff() < pd.Timedelta(0)).sum()),
            "missing_keys": len(grid.difference(unique_times)),
            "nonfinite_required_rows": int((~np.isfinite(rows[list(RAW_COLUMNS)].to_numpy(float)).all(axis=1)).sum()),
        }
        groups.append(group)
    quality = {
        "source_rows": len(data), "grid_start": grid[0].isoformat(), "grid_end": grid[-1].isoformat(),
        "source_offsets": stamps.str[-6:].value_counts().to_dict(),
        "per_turbine": groups,
        "conflicting_keys": [
            {"time": row.time.isoformat(), "turbine": row.Wind_turbine_name}
            for row in duplicate_keys.itertuples(index=False)
        ],
        "duplicate_policy": "all_rows_at_duplicate_key_unavailable_no_selection_or_mean",
        "missing_policy": "no_interpolation_no_zero_fill_fixed_roster",
        "power_policy": "raw_kw_negative_and_low_power_not_removed",
        "time_bucket_edge_confirmed": False, "observed_arrival_times": False,
        "holdout_value_distributions_computed": False,
    }
    return SiteSource(grid, values, conflict_mask, missing_mask, quality)


def load_source(archive_path: Path):
    """只接受核实过的公开快照；原 CSV 留在 ZIP 中，解析不覆盖它。"""
    payload = archive_path.read_bytes()
    if sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("engie_archive_sha256_mismatch")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"engie_archive_crc_failed:{bad}")
        raw_bytes = archive.read(SCADA_MEMBER)
        if sha256(raw_bytes).hexdigest() != MEMBER_SHA256:
            raise ValueError("engie_scada_sha256_mismatch")
        raw = pd.read_csv(BytesIO(raw_bytes))
        description = archive.read("SCADA_data_description.csv").decode("utf-8-sig")
    if "P;Active_power;kW;" not in description:
        raise ValueError("engie_power_unit_not_confirmed")
    source = source_from_frame(raw, start=DEVELOPMENT_START, end=SOURCE_END)
    source.quality.update({"archive_sha256": ARCHIVE_SHA256, "scada_sha256": MEMBER_SHA256,
                           "archive_bytes": len(payload), "scada_bytes": len(raw_bytes),
                           "archive_crc": "passed", "source_url": SOURCE_URL,
                           "license": "Etalab Open Licence 2.0; attribution ENGIE via OpenOA"})
    return source


def make_batch(source: SiteSource, issues, *, label_end=HOLDOUT_START):
    """起报/输入合法性独立于标签；label_end 之后的功率从不写入建模目标数组。"""
    issues = pd.DatetimeIndex(issues)
    if issues.tz is None or not issues.is_unique or not issues.is_monotonic_increasing:
        raise ValueError("engie_issue_times_require_ordered_unique_timezone")
    issues = issues.tz_convert("UTC")
    if np.any(issues.asi8 % STEP.value):
        raise ValueError("engie_issue_times_off_grid")
    latest = issues - ARRIVAL_LAG
    history_times = latest.asi8[:, None] - np.arange(LOOKBACK - 1, -1, -1)[None, :] * STEP.value
    hidx = source.times.get_indexer(pd.to_datetime(history_times.ravel(), utc=True)).reshape(-1, LOOKBACK)
    present = hidx >= 0
    history = source.values[np.maximum(hidx, 0)].copy()  # [issue, lookback, turbine, raw]
    history[~present] = np.nan
    conflicts = source.conflicting[np.maximum(hidx, 0)] & present[:, :, None]
    input_valid = np.isfinite(history).all(axis=(1, 2, 3))
    # 方位角的首尾相接；不把 359 和 1 度表示成数值上相距很远。
    angle = np.deg2rad(history[..., 2])
    encoded = np.stack([history[..., 0], history[..., 1], np.sin(angle), np.cos(angle),
                        history[..., 3]], axis=-1)
    features = encoded.transpose(0, 2, 1, 3).reshape(len(issues), len(ROSTER), -1)
    target_times = issues.asi8[:, None] + np.array(HORIZONS)[None, :] * pd.Timedelta(minutes=1).value
    allowed = target_times < utc(label_end).value
    boundary_valid = allowed.all(axis=1)
    target_idx = source.times.get_indexer(pd.to_datetime(target_times.ravel(), utc=True)).reshape(-1, 6)
    targets = np.full((len(issues), len(ROSTER), 6), np.nan)
    safe = allowed & (target_idx >= 0)
    rows, horizons = np.nonzero(safe)
    targets[rows, :, horizons] = source.values[target_idx[rows, horizons], :, 0]
    label_valid = np.isfinite(targets).all(axis=(1, 2))
    return RollingBatch(issues, features, targets, input_valid, label_valid, boundary_valid,
                        conflicts.any(axis=(1, 2)), ~np.isfinite(history).all(axis=(1, 2, 3)))


def before_training_boundary(batch, boundary):
    """最后一个标签也必须已到达；严格早于边界，与特征时钟分开。"""
    available = batch.issues + pd.Timedelta(minutes=max(HORIZONS)) + ARRIVAL_LAG
    return batch.scoreable & (available < utc(boundary))


def persistence(features):
    values = np.asarray(features)
    if values.ndim != 3 or values.shape[1:] != (4, LOOKBACK * len(FEATURES)):
        raise ValueError("engie_feature_shape_mismatch")
    return np.repeat(values[:, :, -len(FEATURES), None], len(HORIZONS), axis=2)


def admit(source, batch):
    reports = {}
    for name, (start, end) in QUARTERS.items():
        mask = (batch.issues >= pd.Timestamp(start, tz="UTC")) & (batch.issues < pd.Timestamp(end, tz="UTC"))
        reports[name] = batch.counts(mask)
    passed = all(report["scoreable_fraction"] >= 0.8 for report in reports.values())
    return {"status": "admitted" if passed else "rejected", "minimum_coverage": 0.8,
            "scope": "fixed_roster_development_rolling_baseline_not_realtime_readiness",
            "quality": source.quality, "quarters": reports}
