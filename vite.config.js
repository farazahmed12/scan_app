// vite.config.js

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  
  // 💥 THE CRUCIAL FIX IS HERE 💥
  optimizeDeps: {
    // List the packages that are causing the 'default' export error.
    // Vite will then properly analyze and bundle their exports.
    include: [
      '@mediapipe/tasks-vision',
      '@mediapipe/drawing_utils',
      '@mediapipe/hands',
    ],
  },
});