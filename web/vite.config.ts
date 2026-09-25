import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    // En dev, l'API FastAPI tourne à côté ; en V0, FastAPI sert le build.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
