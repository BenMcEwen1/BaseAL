import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { readFileSync } from 'fs'

// Read version from pyproject.toml (single source of truth)
const pyproject = readFileSync(resolve(__dirname, '../pyproject.toml'), 'utf-8')
const version = pyproject.match(/version\s*=\s*"([^"]+)"/)[1]

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(version)
  },
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
