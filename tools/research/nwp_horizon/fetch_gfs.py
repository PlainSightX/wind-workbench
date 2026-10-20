"""从 NCAR RDA d084001（GFS 0.25° 历史预报）按点抽取站点 2x2 格点，礼貌、可续跑。

每个请求是一次运行的一个 3h 预报文件，经 THREDDS NetcdfSubset 只取站点所在格子的
四个格点（约 6 KB netCDF3）。结果缓存到 --cache；已存在且可读的文件不会重新请求。
404 记为该文件缺失（运行未归档），其他错误重试后仍失败则记入失败日志，可再次运行补抓。

用法（项目根目录）：
    python tools/research/nwp_horizon/fetch_gfs.py --cache .local/nwp-horizon/gfs \
        --start 2015-01-15 --end 2016-01-01 --leads 3 6 9 12 15
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import pandas as pd
from scipy.io import netcdf_file

BASE = "https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001"
VARIABLES = (
    "u-component_of_wind_height_above_ground",
    "v-component_of_wind_height_above_ground",
    "Wind_speed_gust_surface",
    "Pressure_surface",
    "Temperature_height_above_ground",
)
# 站点 48.452N 5.588E 位于 48.25-48.5N、5.5-5.75E 这个 0.25° 格子内
BBOX = {"north": 48.5, "south": 48.25, "west": 5.5, "east": 5.75}
MAX_WORKERS = 4  # 公共服务，并发不超过 4
RETRIES = 4
_lock = threading.Lock()


def url_for(reftime, lead):
    """RDA 文件布局：YYYY/YYYYMMDD/gfs.0p25.YYYYMMDDHH.fFFF.grib2。"""
    stamp = reftime.strftime("%Y%m%d%H")
    name = f"gfs.0p25.{stamp}.f{lead:03d}.grib2"
    query = "&".join([f"var={v}" for v in VARIABLES]
                     + [f"{k}={v}" for k, v in BBOX.items()] + ["accept=netcdf3"])
    return f"{BASE}/{reftime:%Y}/{reftime:%Y%m%d}/{name}?{query}"


def cache_path(cache, reftime, lead):
    return cache / f"{reftime:%Y%m%d%H}.f{lead:03d}.nc"


def readable(path):
    """缓存文件必须是完整 netCDF 且含全部变量，否则视为未抓取。"""
    try:
        with netcdf_file(path, "r", mmap=False) as handle:
            return all(v in handle.variables for v in VARIABLES)
    except Exception:
        return False


def fetch_one(cache, reftime, lead):
    path = cache_path(cache, reftime, lead)
    if path.exists() and readable(path):
        return "cached", None
    url = url_for(reftime, lead)
    error = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            temporary = path.with_suffix(".part")
            temporary.write_bytes(payload)
            if not readable(temporary):
                raise ValueError("unreadable_netcdf")
            temporary.replace(path)
            return "fetched", None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return "missing", "http_404"
            error = f"http_{exc.code}"
        except Exception as exc:  # 网络超时、截断等，退避后重试
            error = type(exc).__name__
        time.sleep(5 * 2 ** attempt)
    return "failed", error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start", default="2015-01-15")
    parser.add_argument("--end", default="2016-01-01")
    parser.add_argument("--leads", type=int, nargs="+", default=[3, 6, 9, 12, 15])
    parser.add_argument("--runs", nargs="*", help="只抓指定运行 YYYYMMDDHH（补抓回退所需时效）")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.runs:
        runs = [pd.Timestamp(r[:8] + "T" + r[8:] + ":00Z") for r in args.runs]
    else:
        runs = pd.date_range(pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"),
                             freq="6h", inclusive="left")
    jobs = [(run, lead) for run in runs for lead in args.leads]
    log = args.cache / "fetch-log.jsonl"
    counts = {"cached": 0, "fetched": 0, "missing": 0, "failed": 0}
    started = time.time()
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, args.cache, run, lead): (run, lead) for run, lead in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            run, lead = futures[future]
            status, error = future.result()
            counts[status] += 1
            if status in ("missing", "failed"):
                with _lock, log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"reftime": run.isoformat(), "lead_h": lead,
                                             "status": status, "error": error,
                                             "at": pd.Timestamp.now(tz="UTC").isoformat()}) + "\n")
            if done % 200 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} {counts} {time.time() - started:.0f}s", flush=True)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
