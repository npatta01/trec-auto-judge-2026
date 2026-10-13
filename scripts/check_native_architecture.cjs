/* Browser checks use only the synthetic, public architecture demo. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

const demo = path.resolve(__dirname, '../docs/submission-architecture.html');
assert.ok(fs.existsSync(demo), 'Public native architecture demo must exist');
const url = process.argv[2] || pathToFileURL(demo).href;
let checks = 0;
function check(actual, expected, message) {
  assert.deepEqual(actual, expected, message);
  checks++;
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.BROWSER_EXECUTABLE ? {executablePath: process.env.BROWSER_EXECUTABLE} : {})
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {
      if (message.type() === 'error') errors.push(message.text());
    });
    await page.goto(url);
    check(await page.title(), 'TREC AutoJudge · Submission architecture', 'Page identity');
    check(await page.locator('h1').textContent(), 'Our submission architecture', 'Meaningful opening screen');
    check(await page.locator('#demo-notice').isVisible(), true, 'Synthetic data disclosure is visible');
    check(await page.locator('#architecture-submissions button').count(), 6, 'All six method/track examples render');
    check(await page.locator('#architecture-submissions').textContent().then(text => text.includes('confirmed')), false, 'Demo never claims real confirmations');

    const tabs = page.getByRole('tab');
    check(await page.locator('[role="tab"][aria-pressed]').count(), 0, 'Tabs expose selection, not button-only pressed state');
    const panels = ['#architecture-overview', '#architecture-evidence', '#architecture-scoring'];
    for (const width of [1416, 390, 320]) {
      await page.setViewportSize({width, height: 1037});
      for (const theme of ['light', 'dark']) {
        await page.locator('#theme').selectOption(theme);
        check(await page.locator('html').getAttribute('data-theme'), theme, 'Theme selector updates the page');
        for (let index = 0; index < panels.length; index++) {
          await tabs.nth(index).click();
          check(await page.locator(panels[index]).isVisible(), true, 'Selected native panel is visible');
          check(await tabs.nth(index).getAttribute('aria-selected'), 'true', 'Selected tab is announced');
          check(await page.locator('.architecture-panel:visible').count(), 1, 'Inactive panels are hidden');
          check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `No overflow at ${width}px in ${theme}`);
          if (process.env.QA_SCREENSHOTS) {
            fs.mkdirSync(process.env.QA_SCREENSHOTS, {recursive:true});
            await page.screenshot({path:path.join(process.env.QA_SCREENSHOTS, `${theme}-${width}-${index}.png`), fullPage:true});
          }
        }
      }
    }

    await tabs.nth(0).click();
    await tabs.nth(0).focus();
    for (const [key, index] of [['ArrowRight', 1], ['End', 2], ['ArrowRight', 0], ['ArrowLeft', 2], ['Home', 0]]) {
      await page.keyboard.press(key);
      check(await tabs.nth(index).evaluate(node => node === document.activeElement), true, `${key} moves focus`);
      check(await page.locator(panels[index]).isVisible(), true, `${key} switches panel`);
    }
    await tabs.nth(1).click();
    const disclosure = page.locator('#architecture-evidence details').first();
    await disclosure.locator('summary').click();
    check(await disclosure.evaluate(node => node.open), true, 'Reference details open');
    await page.locator('[data-architecture-next]').click();
    check(await page.locator('#architecture-scoring').isVisible(), true, 'Evidence flow links to scoring');

    // A missing/wrong activation callback must fail these hand-checked expectations.
    for (const [panel, selector] of [['#architecture-overview', '[data-architecture-package]'], ['#architecture-scoring', '#architecture-submissions button']]) {
      await tabs.nth(panel.includes('overview') ? 0 : 2).click();
      const buttons = page.locator(`${panel} ${selector}`);
      for (let index = 0; index < 6; index++) {
        const expected = ['rag:pairwise', 'ragtime:pairwise', 'rag:evidence', 'ragtime:evidence', 'rag:combined', 'ragtime:combined'][index];
        await buttons.nth(index).click();
        check(await page.locator('#demo-package').getAttribute('data-selected'), expected, 'Artifact activation selects exact track and method');
        check(await page.locator('#demo-package h2').evaluate(node => node === document.activeElement), true, 'Artifact activation moves focus to the result');
        check(await page.locator('#demo-package').textContent().then(text => text.includes('not a TIRA receipt')), true, 'Selected package remains explicitly synthetic');
      }
    }
    check(errors, [], 'No runtime or console errors');
    console.log(`${checks} browser checks passed (synthetic demo, desktop/mobile, light/dark).`);
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
