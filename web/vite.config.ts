import { sites } from "@openai/sites-vite-plugin";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sites()],
  base: "./",
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        matcher: new URL("./index.html", import.meta.url).pathname,
        method: new URL("./method.html", import.meta.url).pathname,
      },
    },
  },
});
