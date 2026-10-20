"""把 GFS 缓存与 ERA5 原始 JSON 双线性插值到站点，写成可提交的小 CSV 与来源清单。

输出（--output 目录）：
- gfs_site_2015.csv：每行一次运行 reftime 的一个 3h 预报时效 lead_h；
  列 u10/v10/u80/v80（m/s）、gust（地面阵风 m/s）、t2m（K）、sp（Pa）。
- era5_site_2015.csv：逐时 UTC；u10/v10/u100/v100、gust10、t2m（K）、sp（Pa）。
- sources.json：两份 CSV 的 sha256、行数、来源 URL 模板与许可说明。
插值在 u/v 分量上做，不对风向角做平均；风速由插值后的分量重新计算。
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from fetch_gfs import BBOX, VARIABLES, url_for
from timing import GRID_LATS, GRID_LONS, SITE_LAT, SITE_LON, bilinear_weights


def _weights_for(lats, lons):
    """按文件里实际的坐标顺序给出 [lat, lon] 权重，不假设升序。"""
    base = bilinear_weights(SITE_LAT, SITE_LON, GRID_LATS, GRID_LONS)
    rows = [GRID_LATS.index(round(float(v), 3)) for v in lats]
    cols = [GRID_LONS.index(round(float(v), 3)) for v in lons]
    return base[np.ix_(rows, cols)]


def _level(handle, name, height):
    """按变量自己的高度维取指定高度，避免依赖 height_above_ground1/4 的编号。"""
    variable = handle.variables[name]
    level_dim = variable.dimensions[1]
    levels = [float(v) for v in handle.variables[level_dim].data]
    return variable.data[0, levels.index(height)]


def read_gfs_file(path):
    with netcdf_file(path, "r", mmap=False) as handle:
        weights = _weights_for(handle.variables["latitude"].data, handle.variables["longitude"].data)
        # 时间维名称随文件变化（time/time1…），从风变量自身的第一维读取；单位为“自 reftime 起小时”
        time_name = handle.variables[VARIABLES[0]].dimensions[0]
        time_var = handle.variables[time_name]
        if b"since" not in time_var.units:
            raise ValueError("gfs_time_units_unexpected")
        lead = float(time_var.data[0])

        def site(field):
            return float((np.asarray(field, float) * weights).sum())

        u, v = VARIABLES[0], VARIABLES[1]
        return lead, {
            "u10": site(_level(handle, u, 10.0)), "v10": site(_level(handle, v, 10.0)),
            "u80": site(_level(handle, u, 80.0)), "v80": site(_level(handle, v, 80.0)),
            "gust": site(handle.variables["Wind_speed_gust_surface"].data[0]),
            "t2m": site(_level(handle, "Temperature_height_above_ground", 2.0)),
            "sp": site(handle.variables["Pressure_surface"].data[0]),
        }


def build_gfs(cache):
    rows = []
    for path in sorted(cache.glob("*.nc")):
        stamp, lead_part = path.stem.split(".")
        reftime = pd.Timestamp(f"{stamp[:8]}T{stamp[8:]}:00Z")
        lead_h = int(lead_part[1:])
        lead, values = read_gfs_file(path)
        if lead != lead_h:  # 文件名与文件内时效必须一致
            raise ValueError(f"gfs_lead_mismatch:{path.name}")
        rows.append({"reftime": reftime.strftime("%Y-%m-%dT%H:%MZ"), "lead_h": lead_h, **values})
    return pd.DataFrame(rows).sort_values(["reftime", "lead_h"]).reset_index(drop=True)


def _uv(speed, direction_deg):
    """气象风向为来向：u = -S sin(d)，v = -S cos(d)。"""
    radians = np.deg2rad(np.asarray(direction_deg, float))
    speed = np.asarray(speed, float)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def build_era5(raw_path):
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    base = bilinear_weights(SITE_LAT, SITE_LON, GRID_LATS, GRID_LONS)
    total, times = None, None
    for point in payload["points"]:
        weight = base[GRID_LATS.index(round(point["latitude"], 3)),
                      GRID_LONS.index(round(point["longitude"], 3))]
        hourly = point["hourly"]
        u10, v10 = _uv(hourly["wind_speed_10m"], hourly["wind_direction_10m"])
        u100, v100 = _uv(hourly["wind_speed_100m"], hourly["wind_direction_100m"])
        block = np.column_stack([u10, v10, u100, v100, hourly["wind_gusts_10m"],
                                 np.asarray(hourly["temperature_2m"], float) + 273.15,
                                 np.asarray(hourly["surface_pressure"], float) * 100.0])
        if times is None:
            times = hourly["time"]
        elif hourly["time"] != times:
            raise ValueError("era5_point_time_axes_differ")
        total = weight * block if total is None else total + weight * block
    frame = pd.DataFrame(total, columns=["u10", "v10", "u100", "v100", "gust10", "t2m", "sp"])
    frame.insert(0, "time", [t + "Z" for t in times])
    return frame, payload["url"]


def record(path):
    data = path.read_bytes()
    return {"file": path.name, "sha256": sha256(data).hexdigest(), "bytes": len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gfs-cache", type=Path, required=True)
    parser.add_argument("--era5-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gfs = build_gfs(args.gfs_cache)
    gfs_path = args.output / "gfs_site_2015.csv"
    gfs.to_csv(gfs_path, index=False, float_format="%.3f", lineterminator="\n")
    era5, era5_url = build_era5(args.era5_raw)
    era5_path = args.output / "era5_site_2015.csv"
    era5.to_csv(era5_path, index=False, float_format="%.3f", lineterminator="\n")
    example = url_for(pd.Timestamp("2015-01-15T00:00Z"), 3)
    # 抓取日志里仍未取得的文件（404 未归档 / 400 源文件不可读），按 (reftime, lead) 去重
    unavailable = {}
    log = args.gfs_cache / "fetch-log.jsonl"
    for line in log.read_text(encoding="utf-8").splitlines() if log.exists() else []:
        entry = json.loads(line)
        reftime = pd.Timestamp(entry["reftime"])
        if not (args.gfs_cache / f"{reftime:%Y%m%d%H}.f{entry['lead_h']:03d}.nc").exists():
            unavailable[(entry["reftime"], entry["lead_h"])] = entry["error"]
    sources = {
        "site": {"latitude": SITE_LAT, "longitude": SITE_LON,
                 "source": "OpenOA la-haute-borne_asset_table.csv, mean of 4 turbines",
                 "interpolation": "bilinear on u/v components over the enclosing 0.25 deg cell",
                 "grid_lats": list(GRID_LATS), "grid_lons": list(GRID_LONS)},
        "gfs": {**record(gfs_path), "rows": len(gfs), "runs": int(gfs.reftime.nunique()),
                "dataset": "NCAR RDA d084001 NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive",
                "doi": "https://doi.org/10.5065/D65D8PWK",
                "access": "THREDDS NetcdfSubset, bbox " + json.dumps(BBOX),
                "example_url": example,
                "leads_h": "3-15 for every run; 18-27 only for runs needed as fallback",
                "unavailable_files": [{"reftime": k[0], "lead_h": k[1], "error": v}
                                      for k, v in sorted(unavailable.items())],
                "licence": "NOAA/NCEP GFS output: US public domain; cite the RDA dataset DOI"},
        "era5": {**record(era5_path), "rows": len(era5),
                 "dataset": "ERA5 hourly single levels via Open-Meteo Historical Weather API (models=era5)",
                 "request_url": era5_url,
                 "licence": ("Contains modified Copernicus Climate Change Service information 2015; "
                             "neither the European Commission nor ECMWF is responsible for any use. "
                             "Open-Meteo data CC BY 4.0, https://open-meteo.com/"),
                 "note": "surface_pressure as delivered by Open-Meteo (elevation-adjusted)"},
    }
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"gfs_rows": len(gfs), "gfs_runs": sources["gfs"]["runs"], "era5_rows": len(era5)}))


if __name__ == "__main__":
    main()
