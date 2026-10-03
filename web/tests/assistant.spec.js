import { test, expect } from "@playwright/test";
import { mkdir } from 'node:fs/promises';

async function open(page) {
  await page.goto("/#predict");
  await expect(page.locator("#engie-submit")).toBeEnabled();
  await page.locator("#engie-quarter").selectOption({ label: "2015-final" });
  await expect(page.locator("#engie-submit")).toBeEnabled();
}

for (const width of [1440, 390]) {
  test(`@provider real provider result, curve and source ${width}`, async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width, height: 1100 } });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await open(page);
    await page.locator("#engie-model").selectOption({ label: "L1 + 持续性收缩 · 比较候选" });
    await expect(page.locator("#engie-submit")).toBeEnabled();
    await page.locator("#engie-submit").click();
    await expect(page.locator("#engie-output")).toBeVisible();
    expect(await page.locator("#engie-chart").evaluate(c => {
      const pixels = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
      let count = 0;
      for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) count++;
      return count;
    })).toBeGreaterThan(1000);
    await page.locator("#engie-assistant-question").fill("共享收缩相比持续性的MAE改善多少？为什么默认仍是持续性？");
    const response = page.waitForResponse(r => r.url().endsWith("/assistant/answers"), { timeout: 85000 });
    await page.locator("#engie-assistant-submit").click();
    const data = await (await response).json();
    expect(data.status, JSON.stringify(data)).toBe("answered");
    expect(data.facts.some(f => f.id === "c0.lightgbm_l1_shrink.mae_gain")).toBeTruthy();
    expect(data.citations.length).toBeGreaterThan(0);
    await expect(page.locator("#engie-assistant-result")).toBeVisible();
    await page.locator("#engie-assistant-result .assistant-source summary").first().click();
    await expect(page.locator("#engie-assistant-result .assistant-source pre").first()).not.toContainText("正在读取");
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBeTruthy();
    expect(errors).toEqual([]);
    await page.locator("#engie-assistant-form").scrollIntoViewIfNeeded();
    await mkdir('../.local/runtime/assistant-ui', { recursive: true });
    await page.screenshot({ path: `../.local/runtime/assistant-ui/assistant-${width}.png` });
    await context.close();
  });
}

test("@original injected slow response cannot cross selected model", async ({ page }) => {
  await open(page);
  let finish;
  await page.route("**/assistant/answers", async route => {
    await new Promise(resolve => { finish = resolve; });
    await route.fulfill({ json: { status: "answered", answer: "STALE_RESULT", facts: [], citations: [] } });
  });
  await page.locator("#engie-assistant-question").fill("当前结果如何？");
  await page.locator("#engie-assistant-submit").click();
  await expect.poll(() => Boolean(finish)).toBeTruthy();
  await page.locator("#engie-model").selectOption({ label: "Ridge · 比较候选" });
  finish();
  await expect(page.locator("#engie-assistant-result")).toBeHidden();
  await expect(page.locator("#engie-submit")).toBeEnabled();
  await page.locator("#engie-submit").click();
  await expect(page.locator("#engie-output")).toBeVisible();
});

test("@original assistant unavailable leaves actual forecasts usable", async ({ page }) => {
  await open(page);
  await page.route("**/assistant/answers", route => route.fulfill({ status: 429, json: { detail: "assistant_busy" } }));
  await page.locator("#engie-assistant-question").fill("当前结果如何？");
  await page.locator("#engie-assistant-submit").click();
  await expect(page.locator("#engie-assistant-status")).toContainText("另一问题");
  await page.locator("#engie-submit").click();
  await expect(page.locator("#engie-output")).toBeVisible();
});

test("@original editing a question invalidates its pending answer", async ({ page }) => {
  await open(page);
  let finish;
  await page.route("**/assistant/answers", async route => {
    await new Promise(resolve => { finish = resolve; });
    await route.fulfill({ json: { status: "answered", answer: "OLD_QUESTION_ANSWER", facts: [], citations: [] } });
  });
  await page.locator("#engie-assistant-question").fill("原来的问题");
  await page.locator("#engie-assistant-submit").click();
  await expect.poll(() => Boolean(finish)).toBeTruthy();
  await page.locator("#engie-assistant-question").fill("修改后的问题");
  finish();
  await expect(page.locator("#engie-assistant-result")).toBeHidden();
  await expect(page.locator("#engie-assistant-submit")).toBeEnabled();
});
