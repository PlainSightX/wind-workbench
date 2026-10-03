import {
  $,
  api,
  options,
  modelName,
  auditTime,
  sourceTime,
  number,
  escape,
  message,
} from "./common.js";
import { replayChart } from "./charts.js";
let artifacts = [],
  offset = 0,
  generation = 0,
  listToken = 0;
const size = 20;
async function availableWindows(artifactId, token) {
  // 只重试短暂容量冲突的只读校验；不自动重发预测或实验写入。
  for (let attempt = 0; attempt < 9; attempt++) {
    if (token !== generation) return null;
    try {
      return await api(`/artifacts/${artifactId}/replay-windows`);
    } catch (error) {
      if (error.code !== "forecast_capacity_busy" || attempt === 8) throw error;
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
}
async function selectArtifact() {
  const token = ++generation;
  $("replay-output").hidden = true;
  $("predict-submit").disabled = true;
  $("cutoff").disabled = true;
  $("artifact-context").textContent = "";
  if (!$("artifact").value) return;
  $("artifact").disabled = true;
  message("replay-status", "正在校验模型与开放时刻…");
  try {
    const windows = await availableWindows($("artifact").value, token);
    if (token !== generation) return;
    const first = windows.cutoffs[0].slice(0, 16),
      last = windows.cutoffs.at(-1).slice(0, 16);
    $("cutoff").min = first;
    $("cutoff").max = last;
    $("cutoff").value = first;
    $("cutoff").disabled = false;
    $("artifact-context").textContent =
      `训练标签截止 ${sourceTime(windows.training_label_end)}；开放 ${windows.count} 个${windows.mode === "final_historical_replay" ? "冻结正式留出" : "开发"}时刻。数据源时间，时区未知。`;
    message("replay-status", "模型校验通过，可选择历史截止时间。");
    $("predict-submit").disabled = false;
  } catch (error) {
    if (token === generation) message("replay-status", error.message, true);
  } finally {
    if (token === generation) $("artifact").disabled = false;
  }
}
export async function loadArtifacts() {
  const token = ++listToken,
    requestedOffset = offset;
  ++generation;
  $("predict-submit").disabled = true;
  $("replay-output").hidden = true;
  $("artifact-context").textContent = "";
  $("cutoff").disabled = true;
  $("artifact").disabled = true;
  try {
    const rows = await api(
      `/artifacts?limit=${size + 1}&offset=${requestedOffset}`,
    );
    if (token !== listToken) return;
    artifacts = rows.slice(0, size);
    options(
      "artifact",
      artifacts,
      "artifact_id",
      (item) =>
        `${modelName(item.model_key)} · ${auditTime(item.created_at)} · ${item.artifact_id.slice(0, 6)}`,
    );
    $("artifact-prev").disabled = requestedOffset === 0;
    $("artifact-next").disabled = rows.length <= size;
    $("artifact-page-label").textContent =
      `第 ${requestedOffset / size + 1} 页`;
    if (!artifacts.length)
      message(
        "replay-status",
        "暂无已登记模型。历史运行不一定包含可加载的模型包。",
      );
    else await selectArtifact();
  } catch (error) {
    if (token === listToken) message("replay-status", error.message, true);
  } finally {
    if (token === listToken) $("artifact").disabled = false;
  }
}
export function setupReplay() {
  $("artifact").onchange = selectArtifact;
  $("cutoff").oninput = () => {
    ++generation;
    $("replay-output").hidden = true;
    $("predict-submit").disabled = false;
  };
  $("artifact-refresh").onclick = loadArtifacts;
  $("artifact-prev").onclick = () => {
    offset = Math.max(0, offset - size);
    loadArtifacts();
  };
  $("artifact-next").onclick = () => {
    offset += size;
    loadArtifacts();
  };
  $("replay-form").onsubmit = async (event) => {
    event.preventDefault();
    const token = ++generation;
    $("predict-submit").disabled = true;
    $("replay-output").hidden = true;
    message("replay-status", "正在加载模型并预测…");
    try {
      const result = await api("/replays", {
        method: "POST",
        body: JSON.stringify({
          artifact_id: $("artifact").value,
          cutoff: $("cutoff").value,
        }),
      });
      if (token !== generation) return;
      const value = result.forecast;
      $("replay-output").hidden = false;
      $("forecast-metrics").innerHTML = [
        ["目标时间", sourceTime(value.target_time)],
        ["预测功率", number(value.prediction)],
        ["实际功率", number(result.actual)],
        ["绝对误差", number(Math.abs(value.prediction - result.actual))],
      ]
        .map(
          ([label, text]) =>
            `<div><span>${label}</span><strong>${escape(text)}</strong></div>`,
        )
        .join("");
      $("replay-details").innerHTML =
        `<dl><dt>模型版本</dt><dd>${escape(value.model_version)}</dd><dt>工件编号</dt><dd class="mono">${escape(value.artifact_id)}</dd><dt>运行编号</dt><dd class="mono">${escape(value.run_id)}</dd><dt>输入历史</dt><dd>${sourceTime(result.history[0].timestamp)} 至 ${sourceTime(value.cutoff)}，${result.history.length} 条五分钟观测</dd></dl>`;
      replayChart(result);
      message("replay-status", "预测完成，结果来自所选模型的实际加载。");
    } catch (error) {
      if (token === generation) message("replay-status", error.message, true);
    } finally {
      if (token === generation) $("predict-submit").disabled = false;
    }
  };
}
