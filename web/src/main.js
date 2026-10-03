import "./style.css";
import { $, api, icons } from "./common.js";
import { setupTasks, loadTasks } from "./tasks.js";
import { setupCompare, loadRuns } from "./compare.js";
import { setupReplay, loadArtifacts } from "./replay.js";
import { setupEngie, loadEngie } from "./engie.js";
import { setupAssistant } from "./assistant.js";

setupTasks();
setupCompare();
setupReplay();
setupEngie();
setupAssistant();
icons();
async function loadPredictionPage() {
  // 两种合同共享推理容量，顺序校验避免页面初次加载争抢槽位。
  await loadArtifacts();
  await loadEngie();
}
const loaders = {
  tasks: loadTasks,
  compare: loadRuns,
  predict: loadPredictionPage,
};
function navigate() {
  const view =
    location.hash.slice(1) in loaders ? location.hash.slice(1) : "tasks";
  for (const id of Object.keys(loaders)) $(id).hidden = id !== view;
  for (const link of document.querySelectorAll("nav a")) {
    if (link.hash === `#${view}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  loaders[view]();
}
window.addEventListener("hashchange", navigate);
navigate();
async function health() {
  try {
    await api("/health");
    $("connection").textContent = "API / 数据库可用";
    $("connection").className = "online";
  } catch {
    $("connection").textContent = "服务连接异常";
    $("connection").className = "error-text";
  }
}
health();
setInterval(health, 30000);
