import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During development the React app (port 5173) forwards API calls to FastAPI (port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/media': 'http://localhost:8000',
    },
  },
})
