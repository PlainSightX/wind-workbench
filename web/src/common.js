import {
  createIcons,
  Wind,
  ListChecks,
  ChartNoAxesCombined,
  Activity,
  ExternalLink,
  RefreshCw,
  Plus,
  ChevronLeft,
  ChevronRight,
  ArrowRight,
  GitCompareArrows,
  Play,
} from "lucide";

export const $ = (id) => document.getElementById(id);
export const modelName = (key) =>
  ({ persistence: "持续性基线", hist_gradient_boosting: "HGB",
     ridge_0_1: "Ridge (α=0.1)", ridge_1: "Ridge (α=1)", ridge_10: "Ridge (α=10)",
     hgb_delta: "HGB 功率增量", transformer_direct: "Transformer 直接预测",
     transformer_delta: "Transformer 功率增量" })[key] || key;
export const policyName = (key) =>
  key === "fixed_iterations" ? "HGB 固定 180 轮" : "HGB 自动提前停止";
export const statusName = (key) =>
  ({
    pending_dispatch: "等待投递",
    queued: "已排队",
    running: "执行中",
    retry_wait: "等待自动重试",
    succeeded: "已完成",
    failed: "失败",
    expired: "租约已过期",
  })[key] || key;
export const number = (value) =>
  Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 3 });
export const sourceTime = (value) =>
  value?.replace("T", " ").slice(0, 16) || "未记录";
export const auditTime = (value) =>
  new Date(value).toLocaleString("zh-CN", { hour12: false });
export const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export function icons() {
  createIcons({
    icons: {
      Wind,
      ListChecks,
      ChartNoAxesCombined,
      Activity,
      ExternalLink,
      RefreshCw,
      Plus,
      ChevronLeft,
      ChevronRight,
      ArrowRight,
      GitCompareArrows,
      Play,
    },
  });
}

const reasons = {
  final_protocol_not_frozen: "正式评价尚未冻结，不能提前评分测试集。",
  artifact_history_insufficient: "这个模型需要至少24条连续历史观测。",
  database_unavailable: "数据库暂不可用。请恢复服务后刷新。",
  dataset_unavailable: "登记数据暂不可用，未接受此次实验。",
  forecast_capacity_busy: "预测服务正在处理另一请求，请稍后重试。",
  artifact_storage_unavailable: "模型存储暂不可用，请检查本地服务。",
  artifact_incompatible:
    "该模型包与当前推理环境不兼容，请选择兼容模型或提交新实验。",
  artifact_integrity_failed: "模型包完整性校验失败，已阻止加载。",
  artifact_missing: "已登记模型文件缺失，不能预测。",
  artifact_not_ready: "这个模型尚不可用。请重新选择。",
  replay_snapshot_missing: "原任务的输入快照缺失，不能安全回放。",
  replay_snapshot_changed: "原任务输入快照已改变，已阻止回放。",
  replay_cutoff_not_allowed: "此时刻不属于该模型已开放的开发窗口。",
  replay_history_invalid: "历史窗口不足、断档或包含无效数值，已阻止预测。",
  input_or_contract_invalid: "输入数据或冻结合同校验失败。",
  unexpected_worker_error: "执行过程中出现异常，请查看本次执行记录。",
  attempt_limit_reached: "自动重试已达到上限。",
  artifact_fresh_process_verification_failed: "模型包独立进程验证失败。",
};
export const reason = (code) => reasons[code] || `服务返回：${code}`;
export function message(id, text, error = false) {
  $(id).textContent = text;
  $(id).classList.toggle("error-text", error);
}
export async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, {
      ...options,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(
        Array.isArray(data.detail)
          ? "输入不符合要求，请检查配置与时间。"
          : reason(data.detail),
      );
      error.status = response.status;
      error.code = data.detail;
      throw error;
    }
    return data;
  } catch (error) {
    if (!error.status)
      throw new Error("连接中断或请求超时。写入结果可能未知，请重试同一请求。");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
export function options(id, items, key, label) {
  const previous = $(id).value;
  $(id).innerHTML = items
    .map(
      (item) =>
        `<option value="${escape(item[key])}">${escape(label(item))}</option>`,
    )
    .join("");
  if (items.some((item) => String(item[key]) === previous))
    $(id).value = previous;
}
export const statusBadge = (status) =>
  `<span class="badge ${escape(status)}">${escape(statusName(status))}</span>`;
