import {
  $,
  api,
  escape,
  auditTime,
  policyName,
  modelName,
  statusName,
  statusBadge,
  reason,
  message,
  icons,
} from "./common.js";

let offset = 0,
  selected = null,
  loading = false,
  detailToken = 0,
  listToken = 0,
  detailValue = "";
const pageSize = 10;
const pendingKey = "wind.pending-submission.v1";
let pending = null;
try {
  pending = JSON.parse(localStorage.getItem(pendingKey));
} catch {
  /* 存储不可读时提交前会明确失败。 */
}

export async function loadTasks({ refresh = false } = {}) {
  if (loading && !refresh) return;
  const token = ++listToken,
    requestedOffset = offset;
  loading = true;
  try {
    const rows = await api(
      `/tasks?limit=${pageSize + 1}&offset=${requestedOffset}`,
    );
    if (token !== listToken) return;
    message("tasks-status", rows.length ? "" : "暂无实验任务。");
    $("task-rows").innerHTML = rows
      .slice(0, pageSize)
      .map(
        (task) =>
          `<tr class="${selected === task.task_id ? "selected" : ""}"><td>${escape(auditTime(task.created_at))}</td><td>${policyName(task.training_policy)}</td><td>${statusBadge(task.status)}</td><td><button class="icon" data-task="${task.task_id}" aria-label="查看任务 ${escape(auditTime(task.created_at))}" title="查看详情"><i data-lucide="arrow-right"></i></button></td></tr>`,
      )
      .join("");
    $("task-prev").disabled = requestedOffset === 0;
    $("task-next").disabled = rows.length <= pageSize;
    $("task-page-label").textContent =
      `第 ${requestedOffset / pageSize + 1} 页`;
    icons();
    if (!selected && rows.length) selected = rows[0].task_id;
    if (selected) await showTask(selected);
  } catch (error) {
    if (token === listToken) message("tasks-status", error.message, true);
  } finally {
    if (token === listToken) loading = false;
  }
}

async function showTask(id) {
  selected = id;
  const token = ++detailToken;
  try {
    const task = await api(`/tasks/${id}`);
    if (token !== detailToken) return;
    if (JSON.stringify(task) === detailValue) return;
    detailValue = JSON.stringify(task);
    const pendingStatus = ["pending_dispatch", "queued", "retry_wait"].includes(
      task.status,
    );
    $("task-detail").innerHTML =
      `<div class="section-heading"><h2>任务详情</h2>${statusBadge(task.status)}</div>
      <dl><dt>提交时间</dt><dd>${escape(auditTime(task.created_at))}</dd><dt>训练策略</dt><dd>${policyName(task.spec.training_policy)}</dd><dt>模型组合</dt><dd>${escape((task.spec.model_set || []).map(modelName).join(" + "))}</dd><dt>执行次数</dt><dd>${task.attempt_count}</dd><dt>预测目标</dt><dd>一小时后单点功率</dd></dl>
      ${pendingStatus ? '<p class="notice">任务等待后台执行；长时间没有变化时，请检查 worker 与消息服务。页面无法确认 worker 是否在线。</p>' : ""}
      ${task.error_code ? `<p class="error-text">${escape(reason(task.error_code))}</p>` : ""}
      <h3>执行记录</h3>${task.attempts.length ? `<ol class="attempts">${task.attempts.map((a) => `<li><strong>第 ${a.number} 次</strong><span>${escape(statusName(a.status))}</span>${a.error_code ? `<small>${escape(reason(a.error_code))}</small>` : ""}</li>`).join("")}</ol>` : '<p class="muted">尚未创建执行记录。</p>'}
      ${task.result_url ? '<a class="action-link" href="#compare">比较已登记结果 <span aria-hidden="true">→</span></a>' : ""}
      ${task.status === "failed" ? '<button disabled title="待用户触发结对接入">原样重跑（待接入）</button>' : ""}
      <details><summary>追溯信息</summary><dl><dt>任务编号</dt><dd class="mono">${escape(task.task_id)}</dd><dt>最新状态时间</dt><dd>${escape(auditTime(task.updated_at))}</dd><dt>错误码</dt><dd class="mono">${escape(task.error_code || "无")}</dd></dl></details>`;
  } catch (error) {
    if (token === detailToken) {
      detailValue = "";
      $("task-detail").innerHTML =
        `<h2>任务详情</h2><p class="error-text">${escape(error.message)}</p>`;
    }
  }
}

function pendingUI() {
  if (pending) {
    $("policy").value = pending.body.training_policy;
    $("candidate").value = pending.body.purpose === "final_evaluation" ? "final_evaluation" : pending.body.sequence_key || pending.body.candidate_key || "none";
  }
  const candidate = $("candidate").value !== "none";
  if (candidate) $("policy").value = "fixed_iterations";
  $("policy").disabled = Boolean(pending) || candidate;
  $("candidate").disabled = Boolean(pending);
  $("dataset").disabled = Boolean(pending);
  if (pending) $("policy").value = pending.body.training_policy;
  $("submit-task").querySelector("span").textContent = pending
    ? "重试同一提交"
    : "提交实验";
}

export function setupTasks() {
  $("candidate").addEventListener("change", pendingUI);
  pendingUI();
  if (pending)
    message(
      "submit-feedback",
      "存在尚未确认的提交，重试会使用原请求身份，不创建重复任务。",
    );
  $("submit-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("submit-task").disabled = true;
    try {
      if (!pending) {
        const next = {
          key: crypto.randomUUID(),
          body: {
            dataset_id: $("dataset").value,
            training_policy: $("policy").value,
            ...($("candidate").value === "final_evaluation" ? { purpose: "final_evaluation" } :
                $("candidate").value.startsWith("transformer_") ? { sequence_key: $("candidate").value } :
                $("candidate").value !== "none" ? { candidate_key: $("candidate").value } : {}),
          },
        };
        // 先持久化身份再发请求。刷新/超时后仍重发同一份业务操作。
        localStorage.setItem(pendingKey, JSON.stringify(next));
        pending = next;
      }
      pendingUI();
      message("submit-feedback", "正在确认提交…");
      const receipt = await api("/experiments", {
        method: "POST",
        headers: { "Idempotency-Key": pending.key },
        body: JSON.stringify(pending.body),
      });
      localStorage.removeItem(pendingKey);
      pending = null;
      selected = receipt.task_id;
      offset = 0;
      message("submit-feedback", "已接受实验，状态由后台执行更新。");
      await loadTasks({ refresh: true });
    } catch (error) {
      // 明确拒绝（非服务/网络异常）不会产生未知写入，可以重新配置。
      if (error.status && error.status < 500) {
        localStorage.removeItem(pendingKey);
        pending = null;
      }
      message("submit-feedback", error.message, true);
    } finally {
      $("submit-task").disabled = false;
      pendingUI();
    }
  });
  $("task-rows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-task]");
    if (button) showTask(button.dataset.task);
  });
  $("task-prev").onclick = () => {
    offset = Math.max(0, offset - pageSize);
    loadTasks({ refresh: true });
  };
  $("task-next").onclick = () => {
    offset += pageSize;
    loadTasks({ refresh: true });
  };
  $("task-refresh").onclick = () => loadTasks({ refresh: true });
  setInterval(() => {
    if (!document.hidden && !$("tasks").hidden) loadTasks();
  }, 5000);
}
