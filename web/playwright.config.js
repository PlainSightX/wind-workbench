import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  workers: 1,
  grep: process.env.WIND_UI_SUITE === 'all' ? undefined :
    process.env.WIND_UI_SUITE === 'provider' ? /@provider/ :
    process.env.WIND_UI_SUITE === 'original' ? /@original/ :
    process.env.WIND_UI_SUITE === 'extended' ? /@extended/ : /@software/,
  timeout: 120000,
  expect: { timeout: 15000 },
  reporter: [
    ["list"],
    ["json", { outputFile: "../.local/runtime/wind-ui-browser-tests.json" }],
  ],
  use: {
    baseURL: process.env.WIND_TEST_BASE_URL || "http://127.0.0.1:18000",
    ...(process.env.WIND_BROWSER_CHANNEL ? { channel: process.env.WIND_BROWSER_CHANNEL } : {}),
    headless: true,
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
});
