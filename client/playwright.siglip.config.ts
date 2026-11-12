import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/siglip",
  fullyParallel: false,
  reporter: "list",
  timeout: 180_000,
  use: {
    baseURL: "http://127.0.0.1:4173",
    browserName: "chromium",
    headless: true,
    viewport: { width: 1280, height: 720 },
    launchOptions: { args: ["--enable-unsafe-webgpu"] },
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173/demo-pages/login.html",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
