import { test, expect } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const output = "../.local/runtime/engie-final-ui";

test('@original ' + "pending publication refreshes exact record without another POST", async ({
  page,
}) => {
  await openFinal(page);
  const replay = await predict(page);
  const id = "00000000-0000-0000-0000-000000000001";
  const pending = {
    id,
    status: "pending",
    artifact_id: replay.forecast.artifact_id,
    input_sha256: replay.forecast.input_sha256,
  };
  let posts = 0,
    reads = 0;
  await page.route("**/engie/deliveries", (route) => {
    posts++;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(pending),
    });
  });
  await page.route(`**/engie/deliveries/${id}`, (route) => {
    reads++;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        reads === 1
          ? pending
          : {
              ...pending,
              status: "expired",
              reason: "delivery_deadline_exceeded",
            },
      ),
    });
  });
  await page.locator("#engie-publish").click();
  await expect(page.locator("#engie-delivery-status")).toContainText("处理中");
  await page.locator("#engie-delivery-refresh").click();
  await expect(page.locator("#engie-delivery-status")).toContainText("已过期");
  expect(posts).toBe(1);
  expect(reads).toBe(2);
  await expect(page.locator("#engie-delivery-refresh")).toBeHidden();
});
async function openFinal(page) {
  await page.goto("/#predict");
  await expect(page.locator("#engie-submit")).toBeEnabled();
  await page.locator("#engie-quarter").selectOption({ label: "2015-final" });
  await expect(page.locator("#engie-submit")).toBeEnabled();
  await expect(page.locator("#engie-comparison-rows tr")).toHaveCount(5);
}
async function predict(page) {
  const pending = page.waitForResponse(
    (r) =>
      r.url().endsWith("/engie/replays") && r.request().method() === "POST",
  );
  await page.locator("#engie-submit").click();
  const response = await pending;
  expect(response.status()).toBe(200);
  await expect(page.locator("#engie-output")).toBeVisible();
  return response.json();
}

for (const width of [1440, 390]) {
  test('@original ' + `final models and durable publication ${width}`, async ({ browser }) => {
    const context = await browser.newContext({
      viewport: { width, height: 1000 },
      timezoneId: "America/Los_Angeles",
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await openFinal(page);
    await expect(page.locator("#engie-evaluation")).toContainText(
      "开发采用门未通过",
    );
    expect((await predict(page)).forecast.family).toBe("persistence");
    for (const label of [
      "Ridge · 比较候选",
      "LightGBM · 比较候选",
      "LightGBM L1 · 比较候选",
      "L1 + 持续性收缩 · 比较候选",
    ]) {
      await page.locator("#engie-model").selectOption({ label });
      await expect(page.locator("#engie-submit")).toBeEnabled();
    }
    const result = await predict(page);
    expect(result.mode).toBe("imported_final_replay");
    expect(result.forecast.family).toBe("lightgbm_l1_shrink");
    await page
      .locator("#engie-output details")
      .first()
      .locator("summary")
      .click();
    await expect(page.locator("#engie-value-rows tr")).toHaveCount(6);
    await expect(
      page.locator("#engie-value-rows tr").first().locator("td").nth(1),
    ).toHaveText(
      result.forecast.farm_predictions[0].toLocaleString("zh-CN", {
        maximumFractionDigits: 3,
      }),
    );
    const published = page.waitForResponse(
      (r) =>
        r.url().endsWith("/engie/deliveries") &&
        r.request().method() === "POST",
    );
    await page.locator("#engie-publish").click();
    const record = await (await published).json();
    expect(record.status).toBe("published");
    expect(record.result).toEqual(result.forecast);
    await expect(page.locator("#engie-delivery-status")).toContainText(
      `已发布 · ${record.id}`,
    );
    await expect(page.locator("#engie-publish")).toBeDisabled();
    const colored = await page.locator("#engie-chart").evaluate((canvas) => {
      const p = canvas
        .getContext("2d")
        .getImageData(0, 0, canvas.width, canvas.height).data;
      let n = 0;
      for (let i = 0; i < p.length; i += 4)
        if (
          p[i + 3] &&
          Math.max(p[i], p[i + 1], p[i + 2]) -
            Math.min(p[i], p[i + 1], p[i + 2]) >
            20
        )
          n++;
      return n;
    });
    expect(colored).toBeGreaterThan(100);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await mkdir(output, { recursive: true });
    await page
      .locator("#engie")
      .screenshot({ path: `${output}/final-${width}.png` });
    await page.locator("#engie-issue").fill("2015-12-31T23:50");
    expect((await predict(page)).actual).toEqual(
      Array.from({ length: 4 }, () => Array(6).fill(null)),
    );
    await page.locator("#engie-quarter").selectOption({ label: "2014-Q4" });
    await expect(page.locator("#engie-submit")).toBeEnabled();
    await expect(page.locator("#engie-evaluation")).toBeHidden();
    await page.locator("#engie-issue").fill("2015-01-01T00:00");
    await expect(page.locator("#engie-submit")).toBeDisabled();
    expect(errors).toEqual([]);
    await context.close();
  });
}

test('@original ' + "publication error is retryable and stale response is not shown", async ({
  page,
}) => {
  await openFinal(page);
  await predict(page);
  await page.route(
    "**/engie/deliveries",
    (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "database_unavailable" }),
      }),
    { times: 1 },
  );
  await page.locator("#engie-publish").click();
  await expect(page.locator("#engie-delivery-status")).not.toHaveText(
    "正在发布…",
  );
  await expect(page.locator("#engie-publish")).toBeEnabled();
  let release, arrived;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const waiting = new Promise((resolve) => {
    arrived = resolve;
  });
  await page.route(
    "**/engie/deliveries",
    async (route) => {
      const response = await route.fetch();
      arrived();
      await gate;
      await route.fulfill({ response });
    },
    { times: 1 },
  );
  await page.locator("#engie-publish").click();
  await waiting;
  await page
    .locator("#engie-model")
    .selectOption({ label: "Ridge · 比较候选" });
  await expect(page.locator("#engie-submit")).toBeEnabled();
  release();
  await expect(page.locator("#engie-output")).toBeHidden();
  await expect(page.locator("#engie-delivery-status")).toHaveText("");
  expect((await predict(page)).forecast.family).toBe("ridge");
});
