// Render panel/public/og.png (1200x630) from the built og.html served locally.
// usage: NODE_PATH=<dir with playwright> node scripts/render_og.cjs http://127.0.0.1:8791/og.html
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8791/og.html';
  const out = path.resolve(__dirname, '..', 'panel', 'public', 'og.png');
  const b = await chromium.launch({ args: ['--no-sandbox'] });
  const pg = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  await pg.goto(url, { waitUntil: 'networkidle' });
  await pg.waitForFunction(() => window.__ogReady === true, null, { timeout: 20000 });
  await pg.waitForTimeout(800);
  await pg.screenshot({ path: out, clip: { x: 0, y: 0, width: 1200, height: 630 } });
  await b.close();
  console.log('wrote', out);
})();
