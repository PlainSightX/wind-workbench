import { test, expect } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const output = "../.local/runtime/engie-ui";
async function open(page) {
  await page.goto("/#predict");
  await expect(page.locator("#engie-submit")).toBeEnabled();
}
async function predict(page) {
  const result = page.waitForResponse(
    (r) =>
      r.url().endsWith("/engie/replays") && r.request().method() === "POST",
  );
  await page.locator("#engie-submit").click();
  const response = await result;
  expect(response.status()).toBe(200);
  await expect(page.locator("#engie-output")).toBeVisible();
  return response.json();
}

for (const [width, zone] of [
  [1440, "Asia/Shanghai"],
  [390, "America/Los_Angeles"],
]) {
  test('@original ' + `ENGIE actual replay and six-point chart ${width}`, async ({
    browser,
  }) => {
    const context = await browser.newContext({
      viewport: { width, height: 1000 },
      timezoneId: zone,
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await open(page);
    expect(await page.locator("#engie-comparison-rows tr").count()).toBe(3);
    const q2 = await predict(page);
    expect(q2.forecast.family).toBe("persistence");
    await page
      .locator("#engie-output details")
      .first()
      .locator("summary")
      .click();
    await expect(page.locator("#engie-value-rows tr")).toHaveCount(6);
    await expect(page.locator("#engie-value-rows tr").first()).toContainText(
      "2014-04-01 00:10 UTC",
    );
    await page.locator("#engie-member").selectOption("R80711");
    const expected = q2.forecast.predictions[0][0].toLocaleString("zh-CN", {
      maximumFractionDigits: 3,
    });
    await expect(
      page.locator("#engie-value-rows tr").first().locator("td").nth(1),
    ).toHaveText(expected);
    await page
      .locator("#engie-model")
      .selectOption({ label: "LightGBM · 比较候选" });
    await expect(page.locator("#engie-output")).toBeHidden();
    await expect(page.locator("#engie-submit")).toBeEnabled();
    expect((await predict(page)).forecast.family).toBe("lightgbm");
    await page.locator("#engie-quarter").selectOption({ label: "2014-Q4" });
    await expect(page.locator("#engie-submit")).toBeEnabled();
    await page.locator("#engie-issue").fill("2014-12-31T23:50");
    const q4 = await predict(page);
    expect(q4.actual).toEqual(
      Array.from({ length: 4 }, () => Array(6).fill(null)),
    );
    await expect(page.locator("#engie-value-rows")).toContainText(
      "未开放 / 缺测",
    );
    await page.locator("#engie-issue").fill("2015-01-01T00:00");
    await expect(page.locator("#engie-submit")).toBeDisabled();
    await expect(page.locator("#engie-output")).toBeHidden();
    await page.locator("#engie-issue").fill("2014-12-31T23:50");
    await predict(page);
    const pixels = await page.locator("#engie-chart").evaluate((canvas) => {
      const values = canvas
        .getContext("2d")
        .getImageData(0, 0, canvas.width, canvas.height).data;
      let colored = 0;
      for (let i = 0; i < values.length; i += 4)
        if (
          values[i + 3] &&
          Math.max(values[i], values[i + 1], values[i + 2]) -
            Math.min(values[i], values[i + 1], values[i + 2]) >
            20
        )
          colored++;
      return colored;
    });
    expect(pixels).toBeGreaterThan(100);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await mkdir(output, { recursive: true });
    await page
      .locator("#engie")
      .screenshot({ path: `${output}/engie-${width}.png` });
    expect(errors).toEqual([]);
    await context.close();
  });
}

test('@original ' + "stale model response cannot overwrite selection; failure is visible", async ({
  page,
}) => {
  await open(page);
  let release, received;
  const waiting = new Promise((resolve) => {
    received = resolve;
  });
  const hold = new Promise((resolve) => {
    release = resolve;
  });
  await page.route(
    "**/engie/replays",
    async (route) => {
      const response = await route.fetch();
      received();
      await hold;
      await route.fulfill({ response });
    },
    { times: 1 },
  );
  await page.locator("#engie-submit").click();
  await waiting;
  await page
    .locator("#engie-model")
    .selectOption({ label: "Ridge · 比较候选" });
  await expect(page.locator("#engie-submit")).toBeEnabled();
  release();
  await expect(page.locator("#engie-output")).toBeHidden();
  const result = await predict(page);
  expect(result.forecast.family).toBe("ridge");
  await page.route(
    "**/engie/replays",
    (route) =>
      route.fulfill({
        status: 409,
        contentType: "application/json",
        body: JSON.stringify({ detail: "engie_package_integrity_failed" }),
      }),
    { times: 1 },
  );
  await page.locator("#engie-submit").click();
  await expect(page.locator("#engie-status")).toContainText("校验失败");
  await expect(page.locator("#engie-output")).toBeHidden();
});
