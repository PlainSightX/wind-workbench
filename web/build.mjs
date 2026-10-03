import { build } from "esbuild";
import { mkdir, writeFile, readFile } from "node:fs/promises";

const out = "../src/power_forecast_service/web_static";
await mkdir(`${out}/assets`, { recursive: true });
await build({
  entryPoints: ["src/main.js"],
  bundle: true,
  minify: true,
  outfile: `${out}/assets/app.js`,
  target: ["es2022"],
  legalComments: "linked",
});
await writeFile(`${out}/index.html`, await readFile("index.html"));
