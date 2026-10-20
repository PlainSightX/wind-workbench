"""从 Open-Meteo 历史档案（models=era5）取 2015 年站点所在格子四个 ERA5 0.25° 格点的逐时值。

只请求恰好落在 0.25° 网格上的四个坐标，Open-Meteo 返回的就是这些格点本身（响应中
回显的 latitude/longitude 会被校验）。时间参数 timezone=GMT，即 UTC。
输出是原始 JSON（缓存），后续由 build_inputs.py 做双线性插值；不需要账号或密钥。
"""

import argparse
import json
from pathlib import Path
import urllib.request

from timing import GRID_LATS, GRID_LONS

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ("wind_speed_10m,wind_direction_10m,wind_speed_100m,wind_direction_100m,"
          "wind_gusts_10m,temperature_2m,surface_pressure")


def request_url(start, end):
    lats = ",".join(str(lat) for lat in GRID_LATS for _ in GRID_LONS)
    lons = ",".join(str(lon) for _ in GRID_LATS for lon in GRID_LONS)
    return (f"{URL}?latitude={lats}&longitude={lons}&start_date={start}&end_date={end}"
            f"&hourly={HOURLY}&models=era5&timezone=GMT&wind_speed_unit=ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2015-12-31")
    args = parser.parse_args()
    url = request_url(args.start, args.end)
    with urllib.request.urlopen(url, timeout=300) as response:
        payload = json.loads(response.read())
    points = payload if isinstance(payload, list) else [payload]
    expected = [(lat, lon) for lat in GRID_LATS for lon in GRID_LONS]
    echoed = [(round(p["latitude"], 3), round(p["longitude"], 3)) for p in points]
    if echoed != expected:
        raise SystemExit(f"era5_grid_points_mismatch: {echoed}")
    for point in points:
        if point.get("utc_offset_seconds") != 0:
            raise SystemExit("era5_not_utc")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"url": url, "points": points}), encoding="utf-8")
    print(f"saved {len(points)} points x {len(points[0]['hourly']['time'])} hours")


if __name__ == "__main__":
    main()
