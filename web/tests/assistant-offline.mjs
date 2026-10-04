// 实际页面消费已校验的workflow回复；只注入上下文、图标模块和HTTP，不访问服务。
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';

const [root, fixture, output, dependencies] = process.argv.slice(2);
const require = createRequire(dependencies);
// 普通开发使用项目依赖；候选复核也可明确指定已存在的Playwright运行库。
const { chromium } = require('playwright');
const cases = JSON.parse(await readFile(fixture, 'utf8'));
const source = await readFile(path.join(root, 'web/src/assistant.js'), 'utf8');
const common = await readFile(path.join(root, 'web/src/common.js'), 'utf8');
const css = await readFile(path.join(root, 'web/src/style.css'), 'utf8');
const built = path.join(root, 'src/power_forecast_service/web_static');
const builtHTML = await readFile(path.join(built, 'index.html'), 'utf8');
const builtJS = await readFile(path.join(built, 'assets/app.js'), 'utf8');
const builtCSS = await readFile(path.join(built, 'assets/app.css'), 'utf8');
const imports = common.match(/import\s*\{([^}]+)\}\s*from\s*"lucide"/s)[1].split(',').map(v => v.trim()).filter(Boolean);
const lucide = imports.map(name => `export const ${name} = ${name === 'createIcons' ? '() => {}' : '{}'};`).join('\n');
const selectors = ['engie-quarter', 'engie-model', 'engie-issue', 'left-run', 'right-run', 'left-model', 'right-model'];
const html = `<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/style.css"><script type="importmap">{"imports":{"lucide":"/lucide.js"}}</script>
<main><section id="engie"></section><section id="compare"></section>
${selectors.map(id => `<select id="${id}"><option value="fixture">fixture</option></select>`).join('')}</main>
<script type="module">import { setupAssistant } from '/assistant.js'; setupAssistant(); window.loaded = true;</script>`;
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || (process.platform === 'win32' ? 'msedge' : undefined), headless: true });
const observations = [];
try {
  for (const mode of ['source', 'bundle']) {
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    let response, calls = 0, holdReply, received;
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/') return route.fulfill({ contentType: 'text/html', body: mode === 'bundle' ? builtHTML : html });
      if (pathname === '/assets/app.js') return route.fulfill({ contentType: 'text/javascript', body: builtJS });
      if (pathname === '/assets/app.css') return route.fulfill({ contentType: 'text/css', body: builtCSS });
      if (pathname === '/health') return route.fulfill({ json: { status: 'ok' } });
      if (pathname === '/artifacts') return route.fulfill({ json: [] });
      if (pathname === '/engie/imports') return route.fulfill({ json: [{
        import_id: '00000000-0000-4000-8000-000000000123', quarter: '离线fixture', scope: 'fixture',
        models: [{ family: 'persistence', artifact_id: 'fixture', metrics: { mae: 12.5, rmse: 15 },
          coverage: { planned: 1, input_valid: 1, scoreable: 1, output_count: 1 } }],
      }] });
      if (pathname === '/engie/artifacts/fixture/windows') return route.fulfill({ json: {
        count: 1, issue_times: ['2015-01-01T01:00:00Z'], training_label_available: '2014-12-31T00:00:00Z',
      } });
      if (pathname === '/assistant.js') return route.fulfill({ contentType: 'text/javascript', body: source });
      if (pathname === '/common.js') return route.fulfill({ contentType: 'text/javascript', body: common });
      if (pathname === '/style.css') return route.fulfill({ contentType: 'text/css', body: css });
      if (pathname === '/lucide.js') return route.fulfill({ contentType: 'text/javascript', body: lucide });
      if (pathname === '/engie.js') return route.fulfill({ contentType: 'text/javascript', body: 'export function engieAssistantContexts() { return [{kind:"engie_import", id:"00000000-0000-4000-8000-000000000123"}]; }' });
      if (pathname === '/assistant/answers') {
        calls++;
        if (holdReply) { received(); await holdReply; }
        return route.fulfill({ json: response });
      }
      return route.abort();
    });
    await page.goto('http://wind-test.local/#predict');
    if (mode === 'bundle') await page.waitForFunction(() => !document.querySelector('#engie-submit').disabled);
    else await page.waitForFunction(() => window.loaded);
    for (const entry of cases) {
      response = entry.answer;
      await page.locator('#engie-assistant-question').fill(entry.name);
      await page.locator('#engie-assistant-submit').click();
      await page.locator('#engie-assistant-result').waitFor({ state: 'visible' });
      const answer = page.locator('#engie-assistant-result .assistant-answer');
      const visible = await answer.textContent();
      assert.equal(visible, response.answer, 'DOM必须等于后端校验并渲染的原正文');
      if (entry.name.startsWith('rejected_')) {
        assert.ok(!visible.includes('12.5') && !visible.includes('GW'));
        assert.ok((await page.locator('#engie-assistant-status').textContent()).includes('已阻止展示'));
      } else {
        await page.locator('#engie-assistant-result details summary').first().click();
        const facts = await page.locator('#engie-assistant-result tbody').textContent();
        assert.ok(facts.includes(`${response.facts[0].value} kW`));
      }
      assert.equal(await answer.locator('img').count(), 0);
      assert.equal(await page.evaluate(() => window.injected), undefined);
      assert.equal(await page.locator('#engie-assistant-submit').isDisabled(), false);
      const dimensions = await answer.evaluate(node => ({ scroll: node.scrollWidth, client: node.clientWidth }));
      assert.ok(dimensions.scroll <= dimensions.client + 1, '正文不能横向溢出');
      observations.push({ mode, width, name: entry.name, status: response.status, visible,
        calls: entry.calls, audit_rows: entry.audit_rows });
      if (['repaired_label_sign', 'negative_fact', 'rejected_label_sign'].includes(entry.name)) {
        await answer.scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(output, `${mode}-${entry.name}-${width}.png`) });
      }
    }
    assert.equal(calls, cases.length);
    // 选择变化先使回答失效；迟到的HTTP回复不得重新显示旧对象的答案。
    let release;
    holdReply = new Promise(resolve => { release = resolve; });
    const requestStarted = new Promise(resolve => { received = resolve; });
    response = cases[0].answer;
    await page.locator('#engie-assistant-question').fill('late response');
    await page.locator('#engie-assistant-submit').click();
    await requestStarted;
    await page.locator('#engie-model').dispatchEvent('change');
    const finished = page.waitForResponse('**/assistant/answers');
    release();
    await finished;
    assert.equal(await page.locator('#engie-assistant-result').isHidden(), true);
    assert.equal(await page.locator('#engie-assistant-status').textContent(), '');
    assert.equal(await page.locator('#engie-assistant-submit').isDisabled(), false);
    observations.push({ mode, width, name: 'late_selection_invalidated', status: 'passed' });
    assert.deepEqual(errors, []);
    await page.close();
  }
  }
} finally {
  await browser.close();
}
await writeFile(path.join(output, 'observations.json'), JSON.stringify(observations, null, 2));
console.log(JSON.stringify({ result: 'passed', observations: observations.length,
  viewports: [1440, 390], consumer: 'actual source module AND generated app.js/index.html/app.css',
  scope: 'injected workflow replies, contexts and icon module; no provider/PG/live service' }));
