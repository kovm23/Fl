import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      '/upload': { target: 'http://127.0.0.1:5000', changeOrigin: true },
      // pokud používáš i /extract, přidej alias:
      // '/extract': { target: 'http://127.0.0.1:5000', changeOrigin: true },
    },
  },
})
