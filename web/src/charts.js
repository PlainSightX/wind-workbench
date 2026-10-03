import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";
import { $, sourceTime } from "./common.js";
Chart.register(
  LineController,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
);
Chart.defaults.font.family = '"Microsoft YaHei", system-ui, sans-serif';
Chart.defaults.color = "#627079";
const charts = new Map();
function draw(id, data, options = {}) {
  charts.get(id)?.destroy();
  const chart = new Chart($(id), {
    type: "line",
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { usePointStyle: true, boxWidth: 8 },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { maxTicksLimit: 7, maxRotation: 0 },
        },
        y: { title: { display: true, text: "MW" } },
      },
      ...options,
    },
  });
  charts.set(id, chart);
}
export function compareChart(rows, left, right) {
  draw("comparison-chart", {
    labels: rows.map((row) => sourceTime(row.target_time).slice(5)),
    datasets: [
      {
        label: "实际功率",
        data: rows.map((r) => r.actual),
        borderColor: "#202b32",
        borderWidth: 1.6,
        pointRadius: 0,
      },
      {
        label: `左 · ${left}`,
        data: rows.map((r) => r.left_prediction),
        borderColor: "#3978be",
        borderWidth: 1.7,
        pointRadius: 0,
      },
      {
        label: `右 · ${right}`,
        data: rows.map((r) => r.right_prediction),
        borderColor: "#117c75",
        borderWidth: 1.7,
        pointRadius: 0,
      },
    ],
  });
}
export function replayHistoryPoints(history) {
  // 回放合同保证五分钟连续历史；最后一个输入必须落在cutoff=0，而非写死13点。
  return history.map((row, index) => ({
    x: (index - history.length + 1) * 5,
    y: row.wind_power,
  }));
}

export function replayChart(result) {
  const history = replayHistoryPoints(result.history);
  draw(
    "replay-chart",
    {
      datasets: [
        {
          label: "历史功率（模型输入）",
          data: history,
          borderColor: "#202b32",
          borderWidth: 2,
          pointRadius: 2,
        },
        {
          label: "一小时后预测",
          data: [{ x: 60, y: result.forecast.prediction }],
          borderColor: "#117c75",
          backgroundColor: "#117c75",
          pointRadius: 7,
          showLine: false,
        },
        {
          label: "一小时后实际（未传入）",
          data: [{ x: 60, y: result.actual }],
          borderColor: "#3978be",
          backgroundColor: "#3978be",
          pointRadius: 6,
          pointStyle: "rectRot",
          showLine: false,
        },
      ],
    },
    {
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          type: "linear",
          min: history[0].x,
          max: 65,
          title: { display: true, text: "相对历史截止时间 / 分钟" },
          ticks: { stepSize: 30 },
        },
        y: { title: { display: true, text: "MW" } },
      },
    },
  );
}
