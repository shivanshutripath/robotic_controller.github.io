import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/robotic_controller.github.io/', // IMPORTANT: repo name, with leading+trailing slash
})
