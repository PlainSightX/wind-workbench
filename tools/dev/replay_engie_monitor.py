"""通过普通HTTP入口推进固定模型监测；相同key中断后重新运行即可继续。"""

import argparse
import json

import httpx


def run(client, *, champion, shadow, start, end, key, window=36, step_limit=12):
    response = client.post("/engie/monitors", json={
        "request_key": key, "champion_id": champion, "shadow_id": shadow,
        "start": start, "end": end, "window_issues": window})
    response.raise_for_status()
    report = response.json()
    while not report["replay_complete"]:
        response = client.post(f"/engie/monitors/{report['id']}/advance", json={
            "through": report["finish_time"], "max_steps": step_limit})
        response.raise_for_status()
        report = response.json()
        print(json.dumps({"id": report["id"], "processed_until": report["processed_until"],
                          "counts": report["counts"]}, ensure_ascii=False), flush=True)
    return report


def display(report):
    counts = report["counts"]
    print("固定模型历史监测（实况到达延迟为模拟，模型不会自动切换）")
    print(f"计划起报 {counts['planned_issues']}；已处理 {counts['attempted_issues']}；"
          f"无效输入 {counts['invalid_inputs']}；失败 {counts['failed_issues']}")
    print(f"主用：{report['contract']['champion']['model_version']}")
    print(f"影子：{report['contract']['shadow']['model_version']}")
    print("窗口MAE仅按窗口内成对样本计算；缺测/无效/失败不代表模型健康。")
    print("累计：已评分/待到达/到期缺失；窗口缺口：待到达/到期缺失/无效/失败。")
    print("时距(min)  累计已评分/待/缺  窗口成对/已处理  窗口缺口  主用MAE(kW)  影子MAE(kW)  窗口状态  复核提示")
    names = {"insufficient_pairs": "样本不足", "shadow_worse_review": "影子误差较高", "within_margin": "未超复核线"}
    statuses = {"not_started": "尚未处理", "incomplete": "窗口不完整",
                "awaiting_labels": "等待实况", "complete": "窗口完整"}
    for item in report["horizons"]:
        c, s = item["champion"]["mae_kw"], item["shadow"]["mae_kw"]
        support = f"{item['rolling_pairs']}/{item['rolling_scheduled_issues']}"
        gaps = "/".join(str(item[key]) for key in (
            "rolling_pending_labels", "rolling_due_missing_labels",
            "rolling_invalid_inputs", "rolling_failed_issues"))
        cumulative = f"{item['scored_pairs']}/{item['pending_labels']}/{item['due_missing_labels']}"
        print(f"{item['horizon_minutes']:>9}  {cumulative:>17}  {support:>15}  {gaps:>10}  "
              f"{format(c, '.3f') if c is not None else '-':>11}  "
              f"{format(s, '.3f') if s is not None else '-':>11}  "
              f"{statuses[item['window_status']]}  {names[item['alert']]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--champion", required=True)
    parser.add_argument("--shadow", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--window-issues", type=int, default=36)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="输出完整机器报告，包含偏差与身份")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        report = run(client, champion=args.champion, shadow=args.shadow,
                     start=args.start, end=args.end, key=args.key,
                     window=args.window_issues, step_limit=args.max_steps)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        display(report)


if __name__ == "__main__":
    main()
