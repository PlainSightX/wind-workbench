import { test, expect } from "@playwright/test";
import { readFile, mkdir, writeFile } from "node:fs/promises";

test("@software Ridge组合、真实结果与显式选版回放", async ({ page }) => {
  const delivery = JSON.parse(await readFile(process.env.WIND_UI_DELIVERY || "../.local/runtime/ui-delivery.json", "utf8"));
  const out = "../.local/runtime/round3-ui";
  await mkdir(out, { recursive: true });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await page.locator("#candidate").selectOption("ridge_0_1");
  await expect(page.locator("#policy")).toHaveValue("fixed_iterations");
  await expect(page.locator("#policy")).toBeDisabled();
  let submitted;
  // 这里只验新增配置的浏览器请求，不重复训练；真实队列由Python e2e负责。
  await page.route("**/experiments", async route => {
    submitted = route.request().postDataJSON();
    await route.fulfill({ status: 409, json: { detail: "dataset_unavailable" } });
  }, { times: 1 });
  await page.locator("#submit-task").click();
  await expect(page.locator("#submit-feedback")).toContainText("登记数据暂不可用");
  expect(submitted).toEqual({ dataset_id: "wind-2019-q1", training_policy: "fixed_iterations", candidate_key: "ridge_0_1" });
  await page.screenshot({ path: `${out}/tasks-desktop.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: `${out}/tasks-mobile.png`, fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("link", { name: "结果比较", exact: true }).click();
  await expect(page.locator("#left-run option").first()).toBeAttached();
  await page.locator("#left-run").selectOption(delivery.run_id);
  await page.locator("#right-run").selectOption(delivery.run_id);
  await page.locator("#left-model").selectOption("persistence");
  await page.locator("#right-model").selectOption("ridge_0_1");
  await page.locator("#compare-submit").click();
  await expect(page.locator("#compare-status")).toHaveText("评分样本与协议一致");
  await expect(page.locator("#metrics")).toContainText(delivery.metrics.ridge_0_1.mae.toLocaleString("zh-CN", { maximumFractionDigits: 3 }));
  await expect(page.locator("#comparison-chart")).toBeVisible();
  const pixels = await page.locator("#comparison-chart").evaluate(canvas => {
    const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let colored = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 80 && Math.max(data[i], data[i+1], data[i+2]) - Math.min(data[i], data[i+1], data[i+2]) > 35) colored++;
    }
    return colored;
  });
  expect(pixels).toBeGreaterThan(300);
  await page.screenshot({ path: `${out}/compare-desktop.png`, fullPage: true });
  await page.getByRole("link", { name: "历史预测", exact: true }).click();
  await expect(page.locator("#artifact")).toBeEnabled();
  await page.locator("#artifact").selectOption(delivery.artifact_id);
  await expect(page.locator("#replay-status")).toContainText("模型校验通过");
  const response = page.waitForResponse(res => res.url().endsWith("/replays") && res.status() === 200);
  await page.locator("#predict-submit").click();
  const replay = await (await response).json();
  expect(replay.forecast.artifact_id).toBe(delivery.artifact_id);
  expect(replay.forecast.model_key).toBe("ridge_0_1");
  await expect(page.locator("#replay-status")).toContainText("预测完成");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: `${out}/predict-mobile.png`, fullPage: true });
  expect(errors).toEqual([]);
  await writeFile(`${out}/evidence.json`, JSON.stringify({
    status: "passed", submission_check: "intercepted_no_training", submitted,
    actual_run_id: delivery.run_id, actual_replay: replay,
    comparison_colored_pixels: pixels, browser_errors: errors,
  }, null, 2));
});
