import { test, expect } from "@playwright/test";

test('@software ' + "模型校验短暂忙碌后可继续，不重发预测", async ({ page }) => {
  let windows = 0,
    predictions = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/replays")) predictions++;
  });
  await page.route("**/artifacts/*/replay-windows", async (route) => {
    windows++;
    if (windows === 1)
      await route.fulfill({
        status: 503,
        json: { detail: "forecast_capacity_busy" },
      });
    else await route.continue();
  });
  await page.goto("/#predict");
  await expect(page.locator("#predict-submit")).toBeEnabled();
  expect(windows).toBe(2);
  expect(predictions).toBe(0);
  await expect(page.locator("#artifact")).toBeEnabled();
});

test('@software ' + "跨页选择保留，返回比较页不把旧曲线套到新选择", async ({
  page,
  request,
}) => {
  const live = await (await request.get("/runs?limit=1")).json();
  const runs = Array.from({ length: 22 }, (_, i) => ({
    ...live[0],
    run_id: `00000000-0000-4000-8000-${String(i).padStart(12, "0")}`,
  }));
  // 浏览列表用确定性注入数据，比较响应复用真实运行结构，仅测试UI状态隔离。
  const actual = await (
    await request.get(
      `/runs/compare-series?left_run_id=${live[0].run_id}&right_run_id=${live[0].run_id}`,
    )
  ).json();
  await page.route("**/runs?*", (route) => {
    const url = new URL(route.request().url());
    const offset = Number(url.searchParams.get("offset"));
    return route.fulfill({ json: runs.slice(offset, offset + 21) });
  });
  await page.route("**/runs/compare-series?*", (route) =>
    route.fulfill({ json: actual }),
  );
  await page.goto("/#compare");
  await page.locator("#left-run").selectOption(runs[0].run_id);
  await page.locator("#run-next").click();
  await expect(page.locator("#run-page-label")).toHaveText("第 2 页");
  await expect(page.locator("#left-run")).toHaveValue(runs[0].run_id);
  await page.locator("#right-run").selectOption(runs[21].run_id);
  await page.locator("#right-model").selectOption("persistence");
  await page.locator("#compare-submit").click();
  await expect(page.locator("#comparison-output")).toBeVisible();
  await page.getByRole("link", { name: "实验任务", exact: true }).click();
  await page.getByRole("link", { name: "结果比较", exact: true }).click();
  await expect(page.locator("#right-model")).toHaveValue("persistence");
  await expect(page.locator("#comparison-output")).toBeHidden();
});

test('@software ' + "迟到的分页响应不覆盖最近选择的页", async ({ page }) => {
  let delayed = false,
    settled = false;
  const list = Array.from({ length: 32 }, (_, i) => ({
    task_id: `task-${i}`,
    status: "queued",
    created_at: new Date(2020, 0, 1, 0, i).toISOString(),
    updated_at: new Date().toISOString(),
    training_policy: "auto_early_stopping",
  }));
  await page.route("**/tasks?*", async (route) => {
    const offset = Number(
      new URL(route.request().url()).searchParams.get("offset"),
    );
    if (offset === 20) {
      delayed = true;
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
    await route.fulfill({ json: list.slice(offset, offset + 11) });
    if (offset === 20) settled = true;
  });
  await page.route("**/tasks/task-*", (route) =>
    route.fulfill({
      json: { ...list[0], spec: {}, attempts: [], attempt_count: 0 },
    }),
  );
  await page.goto("/");
  await expect(page.locator("#task-next")).toBeEnabled();
  await page.locator("#task-next").click();
  await expect(page.locator("#task-page-label")).toHaveText("第 2 页");
  await page.locator("#task-next").click();
  await expect.poll(() => delayed).toBe(true);
  await page.locator("#task-prev").click();
  await expect.poll(() => settled).toBe(true);
  await expect(page.locator("#task-page-label")).toHaveText("第 2 页");
  await expect(page.locator("#task-rows tr")).toHaveCount(10);
  await expect(page.locator("#task-rows button").first()).toHaveAttribute(
    "data-task",
    "task-10",
  );
  await page.locator("#task-prev").click();
  await expect(page.locator("#task-page-label")).toHaveText("第 1 页");
  await expect(page.locator("#task-rows tr")).toHaveCount(10);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../.local/runtime/wind-ui/tasks-mobile.png",
    fullPage: true,
  });
});
