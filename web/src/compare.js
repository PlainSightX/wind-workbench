import {
  $,
  api,
  options,
  modelName,
  policyName,
  auditTime,
  sourceTime,
  number,
  message,
} from "./common.js";
import { compareChart } from "./charts.js";
let runs = [],
  offset = 0,
  seriesOffset = 0,
  query = null,
  generation = 0,
  listToken = 0;
const knownRuns = new Map();
const size = 20;

function models(side) {
  const run = knownRuns.get($(`${side}-run`).value);
  const previous = $(`${side}-model`).value;
  options(
    `${side}-model`,
    (run?.models || []).map((key) => ({ key })),
    "key",
    (item) => modelName(item.key),
  );
  if (
    side === "right" &&
    !run?.models.includes(previous) &&
    run?.models.includes("hist_gradient_boosting")
  )
    $("right-model").value = "hist_gradient_boosting";
}
export async function loadRuns() {
  invalidate();
  const token = ++listToken,
    requestedOffset = offset;
  $("compare-submit").disabled = true;
  try {
    const rows = await api(`/runs?limit=${size + 1}&offset=${requestedOffset}`);
    if (token !== listToken) return;
    runs = rows.slice(0, size);
    for (const run of runs) knownRuns.set(run.run_id, run);
    for (const side of ["left", "right"]) {
      // 保留跨页已选运行；翻页只是浏览候选，不悄悄改变比较对象。
      const selected = knownRuns.get($(`${side}-run`).value);
      const candidates =
        selected && !runs.some((r) => r.run_id === selected.run_id)
          ? [selected, ...runs]
          : runs;
      options(
        `${side}-run`,
        candidates,
        "run_id",
        (r) =>
          `${auditTime(r.created_at)} · ${r.purpose === "final_evaluation" ? "正式留出" : "开发"} · ${policyName(r.training_policy)} · ${r.run_id.slice(0, 6)}`,
      );
      models(side);
    }
    $("run-prev").disabled = requestedOffset === 0;
    $("run-next").disabled = rows.length <= size;
    $("run-page-label").textContent = `第 ${requestedOffset / size + 1} 页`;
    $("compare-submit").disabled =
      !$("left-run").value || !$("right-run").value;
    message(
      "compare-status",
      runs.length ? "" : "尚无已完成运行。先在实验任务中提交一次实验。",
    );
  } catch (error) {
    if (token === listToken) message("compare-status", error.message, true);
  }
}
async function compare() {
  if (!query) return;
  const current = ++generation;
  $("compare-submit").disabled = true;
  message("compare-status", "正在核验评分证据…");
  $("comparison-output").hidden = true;
  try {
    const result = await api(
      `/runs/compare-series?${new URLSearchParams({ ...query, offset: seriesOffset })}`,
    );
    if (current !== generation) return;
    const cmp = result.comparison;
    if (cmp.status !== "comparable") {
      message(
        "compare-status",
        `无法比较：${cmp.reasons.join("；")}。不会计算差值或拼接曲线。`,
        true,
      );
      return;
    }
    message("compare-status", "评分样本与协议一致");
    $("comparison-output").hidden = false;
    $("metrics").innerHTML = [
      ["左侧 MAE", cmp.left.metrics.mae],
      ["右侧 MAE", cmp.right.metrics.mae],
      ["MAE 差值（右−左）", cmp.delta.mae],
      ["左侧 RMSE", cmp.left.metrics.rmse],
      ["右侧 RMSE", cmp.right.metrics.rmse],
    ]
      .map(
        ([label, value]) =>
          `<div><span>${label}</span><strong>${number(value)}</strong></div>`,
      )
      .join("");
    $("comparison-notes").textContent =
      `${cmp.left.context.evaluation_split === "test" ? "冻结正式留出" : "开发验证"} ${cmp.left.metrics.samples} 个样本的指标，越小越好。${cmp.warnings.join(" ")} 本数据结果不代表跨场站泛化或生产收益。`;
    $("series-range").textContent = result.rows.length
      ? `${seriesOffset + 1}–${seriesOffset + result.rows.length} / ${result.total}`
      : "无样本";
    $("series-prev").disabled = seriesOffset === 0;
    $("series-next").disabled = seriesOffset + 288 >= result.total;
    compareChart(
      result.rows,
      modelName(query.left_model),
      modelName(query.right_model),
    );
    $("series-rows").innerHTML = result.rows
      .map(
        (r) =>
          `<tr><td>${sourceTime(r.target_time)}</td><td>${number(r.actual)}</td><td>${number(r.left_prediction)}</td><td>${number(r.right_prediction)}</td></tr>`,
      )
      .join("");
  } catch (error) {
    if (current === generation) message("compare-status", error.message, true);
  } finally {
    if (current === generation) $("compare-submit").disabled = !runs.length;
  }
}
function invalidate() {
  ++generation;
  $("comparison-output").hidden = true;
  $("compare-submit").disabled = !$("left-run").value || !$("right-run").value;
}
export function setupCompare() {
  $("compare-form").onsubmit = (event) => {
    event.preventDefault();
    seriesOffset = 0;
    query = {
      left_run_id: $("left-run").value,
      right_run_id: $("right-run").value,
      left_model: $("left-model").value,
      right_model: $("right-model").value,
    };
    compare();
  };
  for (const side of ["left", "right"])
    $(`${side}-run`).onchange = () => {
      models(side);
      invalidate();
    };
  for (const side of ["left", "right"])
    $(`${side}-model`).onchange = invalidate;
  $("run-prev").onclick = () => {
    invalidate();
    offset = Math.max(0, offset - size);
    loadRuns();
  };
  $("run-next").onclick = () => {
    invalidate();
    offset += size;
    loadRuns();
  };
  $("compare-refresh").onclick = () => {
    invalidate();
    loadRuns();
  };
  $("series-prev").onclick = () => {
    seriesOffset = Math.max(0, seriesOffset - 288);
    compare();
  };
  $("series-next").onclick = () => {
    seriesOffset += 288;
    compare();
  };
}
