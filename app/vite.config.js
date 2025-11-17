import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      // Allow serving files from the parent directory (to access core/docs)
      allow: ['..']
    }
  },
  resolve: {
    alias: {
      '@notebooks': resolve(__dirname, '../core/docs')
    }
  },
  assetsInclude: ['**/*.ipynb']
})
