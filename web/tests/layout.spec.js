import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

for (const width of [390, 320]) {
  for (const id of ["replay-chart", "comparison-chart", "engie-chart"]) {
    test(`@software ${id} 在 ${width}px 下约束尚未重绘的画布`, async ({ page }) => {
      const css = await readFile(new URL("../src/style.css", import.meta.url), "utf8");
      await page.setViewportSize({ width: 1440, height: 1000 });
      // 固定旧桌面 inline 尺寸，避免依赖观察器恰好及时重绘而漏掉回归。
      await page.setContent(`<style>${css}</style><main><div class="chart-wrap"><canvas id="${id}" width="1384" height="350" style="width:1384px;height:350px"></canvas></div></main>`);
      await page.setViewportSize({ width, height: 844 });
      const layout = await page.locator(`#${id}`).evaluate(canvas => ({
        viewport: innerWidth,
        document: document.documentElement.scrollWidth,
        canvas: canvas.getBoundingClientRect().width,
        parent: canvas.parentElement.getBoundingClientRect().width,
        inlineWidth: canvas.style.width,
      }));
      expect(layout.inlineWidth).toBe("1384px");
      expect(layout.canvas).toBeLessThanOrEqual(layout.parent + 1);
      expect(layout.document).toBeLessThanOrEqual(layout.viewport + 1);
    });
  }
}
