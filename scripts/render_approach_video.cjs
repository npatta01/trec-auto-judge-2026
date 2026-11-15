#!/usr/bin/env node

const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { copyFile, link, mkdtemp, rm } = require("node:fs/promises");
const { tmpdir } = require("node:os");
const { join, resolve } = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");

const root = resolve(__dirname, "..");
const assets = join(root, "docs", "assets");
const source = pathToFileURL(join(root, "docs", "approach-storyboard.html")).href;
const fps = 24;

async function render() {
  const temp = await mkdtemp(join(tmpdir(), "autojudge-approach-"));
  let browser;
  try {
    browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, offline: true });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("requestfailed", (request) => errors.push(request.url()));
    await page.goto(source);
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator("img").evaluateAll((images) =>
      images.every((image) => image.complete && image.naturalWidth > 0)), true);
    const duration = await page.evaluate(() => window.approachDuration);
    assert.ok(Number.isFinite(duration) && duration > 0);

    const poster = join(temp, "poster.jpg");
    await page.evaluate(() => window.renderApproach(4, true));
    await page.screenshot({ path: poster, quality: 94 });

    let lastState;
    let lastImage;
    for (let frame = 0; frame < Math.round(duration * fps); frame++) {
      const time = frame / fps;
      const state = await page.evaluate((at) => window.renderApproach(at), time);
      const destination = join(temp, "frame-" + String(frame).padStart(4, "0") + ".jpg");
      if (state === lastState) {
        await link(lastImage, destination);
      } else {
        await page.screenshot({ path: destination, quality: 90 });
        lastImage = destination;
        lastState = state;
      }
    }
    assert.deepEqual(errors, []);

    const video = join(temp, "approach.mp4");
    execFileSync("ffmpeg", [
      "-hide_banner", "-loglevel", "error",
      "-framerate", String(fps), "-i", join(temp, "frame-%04d.jpg"),
      "-an", "-c:v", "libx264", "-threads", "4", "-crf", "20",
      "-vf", "scale=in_range=pc:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709",
      "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
      "-color_primaries", "bt709", "-color_trc", "bt709",
      "-movflags", "+faststart", video,
    ], { stdio: "inherit" });
    await copyFile(poster, join(assets, "autojudge-approach-poster.jpg"));
    await copyFile(video, join(assets, "autojudge-approach.mp4"));
    console.log("Rendered " + duration + " seconds and the approach poster.");
  } finally {
    if (browser) await browser.close();
    await rm(temp, { recursive: true, force: true });
  }
}

render().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
