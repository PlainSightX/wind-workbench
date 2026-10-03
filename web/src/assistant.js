import { $, escape, icons } from "./common.js";
import { engieAssistantContexts } from "./engie.js";

const reasons = {
  assistant_busy: "助手正在处理另一问题，请稍后再试。",
  assistant_timeout: "助手超时，预测和报告仍可使用。",
  provider_not_configured: "模型接口尚未配置。",
  provider_unavailable: "模型接口暂不可用。",
  provider_rate_limited: "模型接口限流，请稍后再试。",
  provider_timeout: "模型接口响应超时。",
  embedding_unavailable: "文档检索模型暂不可用。",
  document_index_unavailable: "文档索引尚未就绪。",
};

const modelNames = {
  lightgbm_l1_shrink: "L1 + 持续性收缩",
  lightgbm_l1: "LightGBM L1",
  lightgbm: "LightGBM",
  persistence: "持续性",
  ridge: "Ridge",
};
const readable = value => Object.entries(modelNames).reduce(
  (text, [key, name]) => text.replaceAll(key, name), value,
);

function mount(parent, prefix, contexts, selectors) {
  $(parent).insertAdjacentHTML("beforeend", `
    <section class="assistant-band" aria-labelledby="${prefix}-title">
      <h2 id="${prefix}-title">结果助手</h2>
      <form id="${prefix}-form" class="assistant-form">
        <label for="${prefix}-question">关于所选结果的问题</label>
        <div class="assistant-input"><textarea id="${prefix}-question" rows="2" maxlength="1200" required></textarea>
        <button id="${prefix}-submit" class="primary" type="submit" aria-label="提交问题" title="提交问题"><i data-lucide="arrow-right"></i></button></div>
      </form>
      <p id="${prefix}-status" role="status"></p>
      <div id="${prefix}-result" hidden></div>
    </section>`);
  let generation = 0;
  const invalidate = () => {
    generation++;
    $(`${prefix}-result`).hidden = true;
    $(`${prefix}-status`).textContent = "";
    $(`${prefix}-submit`).disabled = false;
  };
  for (const selector of selectors) $(selector).addEventListener("change", invalidate);
  $(`${prefix}-question`).addEventListener("input", invalidate);
  window.addEventListener("hashchange", invalidate);
  $(`${prefix}-form`).onsubmit = async (event) => {
    event.preventDefault();
    const selected = contexts();
    if (!selected.length) {
      $(`${prefix}-status`).textContent = "请先选择结果。";
      return;
    }
    const token = ++generation, identity = JSON.stringify(selected);
    $(`${prefix}-submit`).disabled = true;
    $(`${prefix}-result`).hidden = true;
    $(`${prefix}-status`).textContent = "正在查询所选结果与文档依据…";
    try {
      const response = await fetch("/assistant/answers", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: $(`${prefix}-question`).value.trim(), contexts: selected }),
        signal: AbortSignal.timeout(80000),
      });
      const data = await response.json();
      if (token !== generation || identity !== JSON.stringify(contexts())) return;
      if (!response.ok) throw new Error(reasons[data.detail] || "助手暂不可用，请稍后重试。");
      $(`${prefix}-status`).textContent = reasons[data.error] || ({ answered: "回答已生成", insufficient_evidence: "现有证据不足", validation_error: "回答未通过校验，已阻止展示", dependency_error: "助手依赖暂不可用" })[data.status];
      const output = $(`${prefix}-result`);
      // 正文只去掉服务器重复插入的标签；原始指标身份和值仍完整保留在依据表。
      const answer = data.facts.reduce((text, f) => text.replaceAll(`${f.label}：`, ""), data.answer);
      output.innerHTML = `<p class="assistant-answer">${escape(readable(answer))}</p>
        ${data.facts.length ? `<details><summary>数值依据</summary><div class="table-scroll"><table><thead><tr><th>指标</th><th>原始值</th><th>统计范围</th></tr></thead><tbody>${data.facts.map(f => `<tr><td>${escape(readable(f.label))}</td><td>${escape(f.value)} ${escape(f.unit)}</td><td>${escape(f.aggregation)}</td></tr>`).join("")}</tbody></table></div></details>` : ""}
        ${data.citations.map((c, i) => `<details class="assistant-source" data-source="${i}"><summary>${escape(c.title.replace(/ \/ \.$/, ""))}</summary><pre>正在读取依据…</pre></details>`).join("")}`;
      for (const element of output.querySelectorAll(".assistant-source")) {
        element.addEventListener("toggle", async () => {
          if (!element.open || element.dataset.loaded) return;
          const citation = data.citations[Number(element.dataset.source)];
          try {
            const source = await fetch(citation.url, { signal: AbortSignal.timeout(5000) });
            if (!source.ok) throw new Error("source unavailable");
            const doc = await source.json();
            if (doc.source_sha256 !== citation.revision) throw new Error("source revision changed");
            element.querySelector("pre").textContent = doc.text;
            element.dataset.loaded = "true";
          } catch {
            element.querySelector("pre").textContent = "依据暂不可读，请重新展开。";
          }
        });
      }
      output.hidden = false;
    } catch (error) {
      if (token === generation) $(`${prefix}-status`).textContent = error.name === "TimeoutError" ? reasons.assistant_timeout : error.message;
    } finally {
      if (token === generation) $(`${prefix}-submit`).disabled = false;
    }
  };
}

export function setupAssistant() {
  mount("engie", "engie-assistant", engieAssistantContexts, ["engie-quarter", "engie-model", "engie-issue"]);
  mount("compare", "q1-assistant", () => ["left", "right"].flatMap(side => $(`${side}-run`).value ? [{ kind: "q1_run", id: $(`${side}-run`).value, model: $(`${side}-model`).value }] : []), ["left-run", "right-run", "left-model", "right-model"]);
  icons();
}
