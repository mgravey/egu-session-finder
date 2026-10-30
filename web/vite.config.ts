import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        matcher: new URL("./index.html", import.meta.url).pathname,
        method: new URL("./method.html", import.meta.url).pathname,
        privacy: new URL("./privacy.html", import.meta.url).pathname,
      },
    },
  },
});
