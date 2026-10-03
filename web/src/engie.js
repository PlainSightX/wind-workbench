import { Chart } from "chart.js";
import "./charts.js";
import { $, api, options, number, escape, message } from "./common.js";

const names = {
  persistence: "持续性 · 开发参照",
  ridge: "Ridge · 比较候选",
  lightgbm: "LightGBM · 比较候选",
  lightgbm_l1: "LightGBM L1 · 比较候选",
  lightgbm_l1_shrink: "L1 + 持续性收缩 · 比较候选",
};
let imports = [],
  generation = 0,
  permitted = new Set(),
  replay = null,
  chart,
  deliveryKey = null,
  deliveryId = null;
const fmt = (value) =>
  Number.isFinite(value) ? number(value) : "未开放 / 缺测";
const utc = (value) =>
  new Date(value).toISOString().replace("T", " ").slice(0, 16) + " UTC";
const selected = () =>
  imports.find((item) => item.import_id === $("engie-quarter").value);
export function engieAssistantContexts() {
  const item = selected();
  const model = item?.models.find((m) => m.artifact_id === $("engie-model").value);
  return item && model ? [{ kind: "engie_import", id: item.import_id, model: model.family }] : [];
}
const issueISO = () =>
  $("engie-issue").value
    ? new Date(`${$("engie-issue").value}Z`).toISOString()
    : null;
const validIssue = () => permitted.has(Date.parse(issueISO()));
function invalidate() {
  ++generation;
  replay = null;
  deliveryKey = null;
  deliveryId = null;
  if ($("engie-delivery-refresh")) $("engie-delivery-refresh").hidden = true;
  if ($("engie-delivery-status")) $("engie-delivery-status").textContent = "";
  $("engie-output").hidden = true;
  chart?.destroy();
  chart = null;
  return generation;
}
function errorMessage(error) {
  const reasons = {
    engie_issue_not_open: "此起报时刻不属于所选模型开放的评价范围。",
    engie_history_unavailable: "四机组历史不完整或包含无效数据，已拒绝预测。",
    engie_package_integrity_failed: "模型或回放数据校验失败，已阻止加载。",
    engie_package_incompatible: "模型与当前运行环境不兼容。",
    engie_package_missing: "已登记模型文件缺失。",
  };
  message("engie-status", reasons[error.code] || error.message, true);
}
async function selectModel() {
  const token = invalidate();
  permitted = new Set();
  $("engie-issue").value = "";
  $("engie-issue").disabled = $("engie-submit").disabled = true;
  $("engie-context").textContent = "";
  if (!$("engie-model").value) return;
  message("engie-status", "正在校验模型与可回放时刻…");
  try {
    const data = await api(
      `/engie/artifacts/${$("engie-model").value}/windows`,
    );
    if (token !== generation) return;
    permitted = new Set(data.issue_times.map(Date.parse));
    if (!data.count) {
      message("engie-status", "暂无完整历史输入。");
      return;
    }
    const values = data.issue_times.map((v) =>
      new Date(v).toISOString().slice(0, 16),
    );
    Object.assign($("engie-issue"), {
      min: values[0],
      max: values.at(-1),
      value: values[0],
      disabled: false,
    });
    $("engie-submit").disabled = false;
    $("engie-context").textContent =
      `训练标签可得截止 ${utc(data.training_label_available)}；开放 ${number(data.count)} 个起报时刻。`;
    message("engie-status", "模型校验通过。");
  } catch (error) {
    if (token === generation) errorMessage(error);
  }
}
async function selectQuarter() {
  const item = selected();
  $("engie-comparison").hidden = !item;
  if (!item) return;
  $("engie-comparison-note").textContent =
    item.scope === "final_2015"
      ? "2015最终留出 · 四机组合计 · 六时距 · 等季度MAE/RMSE（非全年混合平均）"
      : `${item.quarter} · 四机组合计 · 六时距 · 整个季度的冻结离线结果`;
  $("engie-evaluation").hidden = !item.evaluation;
  if (item.evaluation) {
    const ci =
      item.evaluation.bootstrap.seven_day.families.lightgbm_l1_shrink
        .gain_percent_95ci;
    const metrics = Object.fromEntries(
      item.models.map((m) => [m.family, m.metrics]),
    );
    const gain =
      100 * (1 - metrics.lightgbm_l1_shrink.mae / metrics.persistence.mae);
    $("engie-evaluation").textContent =
      `共享收缩MAE改善 ${fmt(gain)}%；配对七天块95%区间 ${fmt(ci[0])}% 至 ${fmt(ci[1])}%。开发采用门未通过，默认仍为持续性。评价身份 ${item.evaluation.result_sha256.slice(0, 12)}。`;
  }
  $("engie-comparison-rows").innerHTML = item.models
    .map((model) => {
      const c = model.coverage;
      return `<tr><th scope="row">${escape(names[model.family])}</th><td>${fmt(model.metrics.mae)}</td><td>${fmt(model.metrics.rmse)}</td><td>${number(c.planned)}</td><td>${number(c.input_valid)}</td><td>${number(c.scoreable)}</td><td>${number(c.output_count)}</td></tr>`;
    })
    .join("");
  options(
    "engie-model",
    item.models,
    "artifact_id",
    (model) => names[model.family],
  );
  $("engie-model").value = item.models.find(
    (model) => model.family === "persistence",
  ).artifact_id;
  $("engie-model").disabled = false;
  await selectModel();
}
export async function loadEngie() {
  const token = invalidate();
  permitted = new Set();
  for (const id of [
    "engie-quarter",
    "engie-model",
    "engie-issue",
    "engie-submit",
  ])
    $(id).disabled = true;
  $("engie-comparison").hidden = true;
  $("engie-context").textContent = "";
  message("engie-status", "正在读取已登记的离线模型…");
  try {
    const rows = await api("/engie/imports");
    if (token !== generation) return;
    imports = rows;
    options("engie-quarter", imports, "import_id", (item) => item.quarter);
    $("engie-quarter").disabled = !imports.length;
    if (!imports.length) {
      message("engie-status", "尚未登记ENGIE离线模型。");
      return;
    }
    await selectQuarter();
  } catch (error) {
    if (token === generation) errorMessage(error);
  }
}
const sumComplete = (values) =>
  values.every(Number.isFinite) ? values.reduce((a, b) => a + b, 0) : null;
export function engieSeries(result, member = "farm") {
  const f = result.forecast,
    index = f.roster.indexOf(member),
    origin = Date.parse(f.issue_time);
  const relative = (time) => (Date.parse(time) - origin) / 60000;
  return {
    history: result.history.map((row) => ({
      x: relative(row.timestamp),
      y:
        member === "farm"
          ? sumComplete(f.roster.map((t) => row.turbines[t]?.power_kw ?? null))
          : (row.turbines[member]?.power_kw ?? null),
    })),
    predicted: f.target_times.map((time, h) => ({
      x: relative(time),
      y: member === "farm" ? f.farm_predictions[h] : f.predictions[index][h],
    })),
    actual: f.target_times.map((time, h) => ({
      x: relative(time),
      y:
        member === "farm"
          ? sumComplete(f.roster.map((_, t) => result.actual[t][h]))
          : result.actual[index][h],
    })),
  };
}
function draw() {
  if (!replay) return;
  const f = replay.forecast,
    data = engieSeries(replay, $("engie-member").value);
  chart?.destroy();
  chart = new Chart($("engie-chart"), {
    type: "line",
    data: {
      datasets: [
        { label: "历史输入", data: data.history, borderColor: "#202b32" },
        { label: "六时距预测", data: data.predicted, borderColor: "#117c75" },
        { label: "事后实况", data: data.actual, borderColor: "#3978be" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      spanGaps: false,
      parsing: false,
      elements: { line: { borderWidth: 2 }, point: { radius: 3 } },
      scales: {
        x: {
          type: "linear",
          min: -130,
          max: 60,
          title: { display: true, text: "相对起报时刻 / 分钟" },
        },
        y: { title: { display: true, text: "功率 / kW" } },
      },
      plugins: { legend: { position: "bottom" } },
    },
  });
  $("engie-value-rows").innerHTML = f.target_times
    .map(
      (time, h) =>
        `<tr><td>${escape(utc(time))}</td><td>${fmt(data.predicted[h].y)}</td><td>${fmt(data.actual[h].y)}</td></tr>`,
    )
    .join("");
}
function render(result) {
  replay = result;
  const f = result.forecast;
  options(
    "engie-member",
    [
      { key: "farm", name: "固定四机组合计" },
      ...f.roster.map((key) => ({ key, name: key })),
    ],
    "key",
    (item) => item.name,
  );
  $("engie-output").hidden = false;
  $("engie-publish").disabled = false;
  $("engie-result-title").textContent =
    `${names[f.family]} · 起报 ${utc(f.issue_time)}`;
  $("engie-details").innerHTML = `<dl>${[
    ["模型版本", f.model_version],
    ["工件编号", f.artifact_id],
    ["导入编号", f.import_id],
    [
      "输入历史",
      `${utc(result.history[0].timestamp)} 至 ${utc(f.source_cutoff)}；12条十分钟观测`,
    ],
    ["输入指纹", f.input_sha256],
    ["来源指纹", selected().source_sha256],
    ["协议", selected().protocol_version],
  ]
    .map(([key, value]) => `<dt>${escape(key)}</dt><dd>${escape(value)}</dd>`)
    .join("")}</dl>`;
  draw();
}
function showDelivery(saved) {
  deliveryId = saved.id;
  const state = {
    published: "已发布",
    expired: "已过期，结果未发布",
    pending: "处理中",
    failed: "发布失败",
  }[saved.status];
  message(
    "engie-delivery-status",
    `${state} · ${saved.id} · ${saved.reason || "数据库读回一致"}`,
    saved.status === "expired" || saved.status === "failed",
  );
  $("engie-delivery-refresh").hidden = saved.status !== "pending";
}

export function setupEngie() {
  $("predict").insertAdjacentHTML(
    "beforeend",
    `
    <section id="engie" class="engie-band" aria-labelledby="engie-title">
      <div class="section-heading"><h2 id="engie-title">ENGIE 四机组 · 历史预测</h2><button id="engie-refresh" class="icon" type="button" title="刷新ENGIE模型" aria-label="刷新ENGIE模型"><i data-lucide="refresh-cw"></i></button></div>
      <p class="muted">2014开发 / 2015最终留出 · UTC · 原始功率 kW · 历史回放，非实时服务。</p>
      <form id="engie-form" class="engie-controls">
        <label>评价范围<select id="engie-quarter" disabled></select></label>
        <label>模型版本<select id="engie-model" disabled></select></label>
        <label>起报时刻（UTC）<input id="engie-issue" type="datetime-local" step="600" required disabled></label>
        <button id="engie-submit" class="primary" type="submit" disabled><i data-lucide="play"></i>运行六点预测</button>
      </form><p id="engie-context" class="muted"></p><p id="engie-status" role="status"></p>
      <div id="engie-comparison" hidden><h3>同范围模型对照</h3><p id="engie-comparison-note" class="muted"></p><p id="engie-evaluation" hidden></p>
        <div class="table-scroll" tabindex="0" aria-label="ENGIE季度比较"><table><thead><tr><th>模型</th><th>MAE / kW</th><th>RMSE / kW</th><th>计划起报</th><th>合法输入</th><th>可评分</th><th>实际输出</th></tr></thead><tbody id="engie-comparison-rows"></tbody></table></div>
      </div>
      <div id="engie-output" hidden><div class="section-heading"><h3 id="engie-result-title"></h3><label>功率范围<select id="engie-member"></select></label></div>
        <div class="chart-wrap"><canvas id="engie-chart" role="img" aria-label="ENGIE历史输入与六时距预测"></canvas></div>
        <p class="muted">输入止于起报前20分钟；六点对应起报后10至60分钟。实况不传入模型，未开放或缺测处保留空值。</p>
        <details><summary>六个目标时刻的数值</summary><div class="table-scroll"><table><thead><tr><th>目标时间 / UTC</th><th>预测 / kW</th><th>实况 / kW</th></tr></thead><tbody id="engie-value-rows"></tbody></table></div></details>
        <details><summary>本次输入与导入模型身份</summary><div id="engie-details"></div></details>
        <button id="engie-publish" type="button"><i data-lucide="save"></i>发布本次预测</button><button id="engie-delivery-refresh" class="icon" type="button" title="刷新发布状态" aria-label="刷新发布状态" hidden><i data-lucide="refresh-cw"></i></button><p id="engie-delivery-status" role="status"></p>
      </div>
    </section>`,
  );
  $("engie-refresh").onclick = loadEngie;
  $("engie-quarter").onchange = selectQuarter;
  $("engie-model").onchange = selectModel;
  $("engie-member").onchange = draw;
  $("engie-delivery-refresh").onclick = async () => {
    if (!deliveryId) return;
    const token = generation,
      id = deliveryId;
    try {
      const saved = await api(`/engie/deliveries/${id}`);
      if (token === generation && id === deliveryId) showDelivery(saved);
    } catch (error) {
      if (token === generation)
        message("engie-delivery-status", error.message, true);
    }
  };
  $("engie-publish").onclick = async () => {
    if (!replay) return;
    const token = generation,
      current = replay;
    deliveryKey ||= crypto.randomUUID();
    $("engie-publish").disabled = true;
    message("engie-delivery-status", "正在发布…");
    try {
      const record = await api("/engie/deliveries", {
        method: "POST",
        body: JSON.stringify({
          request_key: deliveryKey,
          budget_ms: 60000,
          artifact_id: current.forecast.artifact_id,
          issue_time: current.forecast.issue_time,
          history: current.history,
        }),
      });
      if (token !== generation) return;
      if (
        record.artifact_id !== current.forecast.artifact_id ||
        record.input_sha256 !== current.forecast.input_sha256
      )
        throw new Error("发布身份与本次预测不一致。");
      const saved = await api(`/engie/deliveries/${record.id}`);
      if (token !== generation) return;
      showDelivery(saved);
    } catch (error) {
      if (token === generation) {
        message("engie-delivery-status", error.message, true);
        $("engie-publish").disabled = false;
      }
    }
  };
  $("engie-issue").oninput = () => {
    invalidate();
    $("engie-submit").disabled = !validIssue();
    message(
      "engie-status",
      validIssue()
        ? "已更换时刻，尚未生成本次预测。"
        : "请选择开放范围内、有完整四机组历史的十分钟起报时刻。",
      !validIssue(),
    );
  };
  $("engie-form").onsubmit = async (event) => {
    event.preventDefault();
    if (!validIssue()) return;
    const token = invalidate(),
      artifact = $("engie-model").value,
      issue = issueISO();
    $("engie-submit").disabled = true;
    message("engie-status", "正在加载模型并生成六时距预测…");
    try {
      const result = await api("/engie/replays", {
        method: "POST",
        body: JSON.stringify({ artifact_id: artifact, issue_time: issue }),
      });
      if (token !== generation) return;
      if (
        result.forecast.artifact_id !== artifact ||
        Date.parse(result.forecast.issue_time) !== Date.parse(issue)
      )
        throw new Error("返回结果与选择不一致，已阻止展示。");
      render(result);
      message("engie-status", "预测完成，结果来自所选离线模型的实际加载。");
    } catch (error) {
      if (token === generation) errorMessage(error);
    } finally {
      if (token === generation) $("engie-submit").disabled = !validIssue();
    }
  };
}
