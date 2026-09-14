import { defineConfig } from 'vite';

// Static site.  `/data/*` is served by Caddy from the directory the bosco process
// writes (state/panel); in dev, point it at a local copy with `bosco panel --out`.
export default defineConfig({
  base: './',
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022' },
  server: {
    proxy: {
      '/data': {
        target: 'http://127.0.0.1:8787',
        rewrite: (p) => p.replace(/^\/data/, ''),
      },
    },
  },
});
