import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { readFileSync } from 'fs'

// Read version from the Python package (single source of truth, also used by
// pyproject.toml's dynamic version)
const packageInit = readFileSync(resolve(__dirname, '../baseal/__init__.py'), 'utf-8')
const version = packageInit.match(/^__version__\s*=\s*["']([^"']+)["']/m)[1]

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    __APP_VERSION__: JSON.stringify(version)
  },
  server: {
    fs: {
      // Allow serving files from the parent directory (to access baseal/docs)
      allow: ['..']
    }
  },
  resolve: {
    alias: {
      '@notebooks': resolve(__dirname, '../baseal/docs')
    }
  },
  assetsInclude: ['**/*.ipynb']
})
