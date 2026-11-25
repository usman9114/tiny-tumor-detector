// Drive the running Gradio app with headless Chromium and save PNG frames.
// usage: node tools/capture_frames.js <port> <outdir> <img1> [img2 ...]
const { chromium } = require('playwright');
const [port, outdir, ...imgs] = process.argv.slice(2);
const fs = require('fs');
fs.mkdirSync(outdir, { recursive: true });
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  let n = 0;
  const shot = async (ms) => { await page.screenshot({ path: `${outdir}/${String(n++).padStart(2, '0')}_${ms}.png` }); };
  await page.goto(`http://127.0.0.1:${port}`);
  await page.waitForSelector('#go');
  await shot(1500);
  for (const im of imgs) {
    await page.setInputFiles('input[type=file]', im);
    await page.waitForTimeout(1500);
    await shot(1200);
    await page.click('#go');
    await page.waitForFunction(() => { const t = document.querySelector('#t2 textarea'); return t && t.value.includes('ms'); }, null, { timeout: 120000 });
    await page.waitForTimeout(800);
    await shot(3000);
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(300);
    await shot(2500);
    await page.reload();
    await page.waitForSelector('#go');
  }
  await browser.close();
})();
