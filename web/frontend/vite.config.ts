import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI origin (Oracle Cloud) is injected at build time via
// VITE_API_BASE_URL. In dev, /api is proxied to the local backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 1100,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
        },
      },
    },
  },
});
