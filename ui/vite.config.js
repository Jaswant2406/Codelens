import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/index": "http://localhost:8000",
      "/ask": "http://localhost:8000",
      "/impact": "http://localhost:8000",
      "/deadcode": "http://localhost:8000",
      "/node": "http://localhost:8000",
      "/health": "http://localhost:8000"
    }
  }
});
