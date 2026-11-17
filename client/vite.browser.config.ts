import { defineConfig } from "vite";

// Single ES module for the engine's /dev page; bundles the @pipecat-ai dependencies.
export default defineConfig({
  build: {
    outDir: "dist-browser",
    emptyOutDir: true,
    lib: { entry: "src/index.ts", formats: ["es"], fileName: () => "client.js" },
    target: "es2022",
    rollupOptions: { output: { inlineDynamicImports: true } },
  },
});
