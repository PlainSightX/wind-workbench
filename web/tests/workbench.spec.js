import { test, expect } from "@playwright/test";
import { writeFile, mkdir } from "node:fs/promises";

const evidenceDir = "../.local/runtime/wind-ui";
async function chartHasPixels(page, id) {
  await expect(page.locator(`#${id}`)).toBeVisible();
  const count = await page.locator(`#${id}`).evaluate((canvas) => {
    const data = canvas
      .getContext("2d")
      .getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    for (let i = 0; i < data.length; i += 4)
      if (
        data[i + 3] > 80 &&
        Math.max(data[i], data[i + 1], data[i + 2]) -
          Math.min(data[i], data[i + 1], data[i + 2]) >
          35
      )
        colored++;
    return colored;
  });
  expect(count).toBeGreaterThan(300);
  return count;
}
async function noPageOverflow(page) {
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  const clipped = await page
    .locator("button:visible, input:visible, select:visible, h1:visible")
    .evaluateAll((elements) =>
      elements
        .filter((el) => {
          const r = el.getBoundingClientRect();
          return r.x < -1 || r.right > innerWidth + 1 || r.width < 20;
        })
        .map((el) => el.id),
    );
  expect(clipped).toEqual([]);
}

test('@extended ' + "真实整链路、丢失响应重试、比较与旧工件回放", async ({
  page,
  request,
}) => {
  await mkdir(evidenceDir, { recursive: true });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const oldArtifacts = await (await request.get("/artifacts")).json();
  expect(oldArtifacts.length).toBeGreaterThan(0);
  let firstReceipt, firstKey, secondKey;
  // 服务实际接收后丢弃浏览器响应。刷新再发同键，检验未知写入的恢复，而不是mock成功。
  await page.route(
    "**/experiments",
    async (route) => {
      firstKey = route.request().headers()["idempotency-key"];
      const response = await route.fetch();
      expect(response.status()).toBe(202);
      firstReceipt = await response.json();
      await route.abort("connectionreset");
    },
    { times: 1 },
  );
  await page.goto("/");
  await expect(page.locator("#task-rows tr").first()).toBeVisible();
  await page.locator("#policy").selectOption("fixed_iterations");
  await page.locator("#submit-task").click();
  await expect(page.locator("#submit-feedback")).toContainText("连接中断");
  expect(firstReceipt.task_id).toBeTruthy();
  await page.reload();
  await expect(page.locator("#submit-task")).toContainText("重试同一提交");
  await expect(page.locator("#policy")).toBeDisabled();
  page.on("request", (req) => {
    if (req.url().endsWith("/experiments"))
      secondKey = req.headers()["idempotency-key"];
  });
  const receiptResponse = page.waitForResponse(
    (res) => res.url().endsWith("/experiments") && res.status() === 202,
  );
  await page.locator("#submit-task").click();
  const recovered = await (await receiptResponse).json();
  expect(recovered.task_id).toBe(firstReceipt.task_id);
  expect(secondKey).toBe(firstKey);
  await expect(page.locator("#submit-feedback")).toContainText("已接受实验");
  await expect
    .poll(
      async () =>
        (await (await request.get(recovered.status_url)).json()).status,
      { timeout: 90000 },
    )
    .toBe("succeeded");
  const detail = await (await request.get(recovered.status_url)).json();
  const runId = detail.result_url.split("/").at(-1);
  await page.locator("#task-refresh").click();
  await expect(page.locator("#task-detail")).toContainText("已完成");
  await noPageOverflow(page);
  await page.screenshot({
    path: `${evidenceDir}/tasks-desktop.png`,
    fullPage: true,
  });
  await page.locator("#task-next").click();
  await expect(page.locator("#task-page-label")).toHaveText("第 2 页");
  await page.locator("#task-prev").click();
  await page.getByRole("link", { name: "结果比较", exact: true }).click();
  await expect(page.locator("#left-run option").first()).toBeAttached();
  await page.locator("#left-run").selectOption(runId);
  await page.locator("#right-run").selectOption(runId);
  await page.locator("#left-model").selectOption("persistence");
  await page.locator("#right-model").selectOption("hist_gradient_boosting");
  await page.locator("#compare-submit").click();
  await expect(page.locator("#compare-status")).toHaveText(
    "评分样本与协议一致",
  );
  const comparePixels = await chartHasPixels(page, "comparison-chart");
  await noPageOverflow(page);
  await page.screenshot({
    path: `${evidenceDir}/compare-desktop.png`,
    fullPage: true,
  });
  await page.locator("#series-next").click();
  await expect(page.locator("#series-range")).toContainText("289");
  await page.setViewportSize({ width: 390, height: 844 });
  await chartHasPixels(page, "comparison-chart");
  await noPageOverflow(page);
  await page.screenshot({
    path: `${evidenceDir}/compare-mobile.png`,
    fullPage: true,
  });
  await page.getByRole("link", { name: "历史预测", exact: true }).click();
  await expect(page.locator("#predict-submit")).toBeEnabled();
  const older = oldArtifacts[0];
  await page.locator("#artifact").selectOption(older.artifact_id);
  await expect(page.locator("#replay-status")).toContainText("模型校验通过");
  const predictionResponse = page.waitForResponse(
    (res) => res.url().endsWith("/replays") && res.status() === 200,
  );
  await page.locator("#predict-submit").click();
  const replay = await (await predictionResponse).json();
  await expect(page.locator("#replay-status")).toContainText("预测完成");
  const replayPixels = await chartHasPixels(page, "replay-chart");
  await noPageOverflow(page);
  expect(replay.forecast.artifact_id).toBe(older.artifact_id);
  await page.screenshot({
    path: `${evidenceDir}/predict-mobile.png`,
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await chartHasPixels(page, "replay-chart");
  await noPageOverflow(page);
  await page.screenshot({
    path: `${evidenceDir}/predict-desktop.png`,
    fullPage: true,
  });
  expect(errors).toEqual([]);
  await writeFile(
    `${evidenceDir}/live-flow.json`,
    JSON.stringify(
      {
        task_id: recovered.task_id,
        run_id: runId,
        same_submission_key: firstKey === secondKey,
        old_artifact_replay: replay,
        chart_colored_pixels: {
          comparison: comparePixels,
          replay: replayPixels,
        },
        browser_errors: errors,
      },
      null,
      2,
    ),
  );
});

test('@extended ' + "界面状态合同：空列表、后台等待与失败、不可比、不兼容", async ({
  page,
}) => {
  // 仅验证界面呈现的注入响应，不将此项当成真实故障恢复或业务交付证据。
  await page.route("**/tasks?*", (route) => route.fulfill({ json: [] }));
  await page.goto("/");
  await expect(page.locator("#tasks-status")).toContainText("暂无实验任务");
  await page.unroute("**/tasks?*");
  const task = {
    task_id: "test-task",
    status: "queued",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    training_policy: "auto_early_stopping",
  };
  await page.route("**/tasks?*", (route) => route.fulfill({ json: [task] }));
  await page.route("**/tasks/test-task", (route) =>
    route.fulfill({
      json: { ...task, spec: {}, attempts: [], attempt_count: 0 },
    }),
  );
  await page.locator("#task-refresh").click();
  await expect(page.locator("#task-detail")).toContainText(
    "页面无法确认 worker 是否在线",
  );
  await page.unroute("**/tasks/test-task");
  task.status = "failed";
  await page.route("**/tasks/test-task", (route) =>
    route.fulfill({
      json: {
        ...task,
        spec: {},
        attempts: [
          {
            number: 1,
            status: "failed",
            error_code: "input_or_contract_invalid",
          },
        ],
        attempt_count: 1,
        error_code: "input_or_contract_invalid",
      },
    }),
  );
  await page.locator("#task-refresh").click();
  await expect(page.locator("#task-detail")).toContainText(
    "输入数据或冻结合同校验失败",
  );
  await expect(
    page.getByRole("button", { name: "原样重跑（待接入）" }),
  ).toBeDisabled();
  await page.route("**/runs/compare-series?*", (route) =>
    route.fulfill({
      json: {
        comparison: {
          status: "not_comparable",
          reasons: ["left:scoring_evidence_missing"],
        },
        rows: [],
        total: 0,
      },
    }),
  );
  await page.getByRole("link", { name: "结果比较", exact: true }).click();
  await expect(page.locator("#compare-submit")).toBeEnabled();
  await page.locator("#compare-submit").click();
  await expect(page.locator("#compare-status")).toContainText("无法比较");
  await expect(page.locator("#comparison-output")).toBeHidden();
  await page.route("**/artifacts/*/replay-windows", (route) =>
    route.fulfill({ status: 409, json: { detail: "artifact_incompatible" } }),
  );
  await page.getByRole("link", { name: "历史预测", exact: true }).click();
  await expect(page.locator("#replay-status")).toContainText("不兼容");
  await expect(page.locator("#predict-submit")).toBeDisabled();
});
