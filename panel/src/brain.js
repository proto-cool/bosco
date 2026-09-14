// Soma-position renderer.  One dim base layer per projection (all neurons), and per
// frame only the cells with residual intensity are drawn on top, additively.  Nothing
// here is a graph: no edges, no layout; positions are where the somas are in the volume.

const HEADER = 36; // see src/bosco/panel.py HEADER

export function parseActivity(buf) {
  const dv = new DataView(buf);
  if (dv.getUint8(0) !== 0x42 || dv.getUint8(1) !== 0x4f) throw new Error('not an activity file');
  const tMs = Number(dv.getBigUint64(8, true));
  const dust = dv.getFloat32(16, true);
  const learned = dv.getFloat32(20, true);
  const kcActive = dv.getUint32(24, true);
  const nPops = dv.getUint32(28, true);
  const n = dv.getUint32(32, true);
  let o = HEADER;
  const pops = new Float32Array(buf.slice(o, o + 4 * nPops));
  o += 4 * nPops;
  const idx = new Uint16Array(buf.slice(o, o + 2 * n));
  o += 2 * n;
  const cnt = new Uint8Array(buf.slice(o, o + n));
  return { tMs, dust, learned, kcActive, pops, idx, cnt };
}

export function parseAtlas(buf, meta) {
  const n = meta.n;
  const rec = new DataView(buf);
  const x = new Uint16Array(n), y = new Uint16Array(n), z = new Uint16Array(n);
  const sc = new Uint8Array(n), flags = new Uint8Array(n), pop = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 9;
    x[i] = rec.getUint16(o, true);
    y[i] = rec.getUint16(o + 2, true);
    z[i] = rec.getUint16(o + 4, true);
    sc[i] = rec.getUint8(o + 6);
    flags[i] = rec.getUint8(o + 7);
    pop[i] = rec.getUint8(o + 8);
  }
  return { n, x, y, z, sc, flags, pop, meta };
}

// Colour groups, in legend order.  A neuron belongs to the first group that claims it.
export const GROUPS = [
  { key: 'kc', name: 'kenyon cells', test: (a, i) => a.flags[i] & 1 },
  { key: 'mbon', name: 'mushroom body output', test: (a, i) => a.flags[i] & 2 },
  { key: 'dan', name: 'dopamine', test: (a, i) => a.flags[i] & 4 },
  { key: 'dn', name: 'descending', test: (a, i) => a.flags[i] & 8 },
  { key: 'sensory', name: 'sensory', test: (a, i) => [1, 5, 9, 10, 13].includes(a.sc[i]) },
  { key: 'other', name: 'the rest', test: () => true },
];

export class BrainView {
  constructor(canvas, atlas, colors) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.atlas = atlas;
    this.colors = colors; // key -> [r,g,b]
    this.group = new Uint8Array(atlas.n);
    for (let i = 0; i < atlas.n; i++) {
      this.group[i] = GROUPS.findIndex((g) => g.test(atlas, i));
    }
    this.visible = GROUPS.map(() => true);
    this.intensity = new Float32Array(atlas.n);
    this.hot = new Set();
    this.view = 0;
    this.base = null;
    this.px = new Float32Array(atlas.n);
    this.py = new Float32Array(atlas.n);
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.resize();
    this.tick = this.tick.bind(this);
    requestAnimationFrame(this.tick);
  }

  setView(v) {
    this.view = v;
    this.layout();
  }

  setVisible(g, on) {
    this.visible[g] = on;
    this.layout();
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = Math.max(1, Math.round(r.width * dpr));
    this.h = Math.max(1, Math.round(r.height * dpr));
    this.canvas.width = this.w;
    this.canvas.height = this.h;
    this.layout();
  }

  // project into canvas pixels and paint the dim base layer once
  layout() {
    const a = this.atlas;
    const [ax, ay] = [[a.x, a.y], [a.x, a.z], [a.z, a.y]][this.view];
    const pad = 0.06;
    const sx = this.w * (1 - 2 * pad), sy = this.h * (1 - 2 * pad);
    const s = Math.min(sx, sy) / 65535;
    const ox = (this.w - 65535 * s) / 2, oy = (this.h - 65535 * s) / 2;
    for (let i = 0; i < a.n; i++) {
      this.px[i] = ox + ax[i] * s;
      this.py[i] = oy + ay[i] * s;
    }
    const base = document.createElement('canvas');
    base.width = this.w;
    base.height = this.h;
    const c = base.getContext('2d');
    c.fillStyle = this.colors.bg;
    c.fillRect(0, 0, this.w, this.h);
    const r = Math.max(0.6, 0.9 * (this.w / 900));
    for (let g = GROUPS.length - 1; g >= 0; g--) {
      if (!this.visible[g]) continue;
      const [cr, cg, cb] = this.colors[GROUPS[g].key];
      c.fillStyle = `rgba(${cr},${cg},${cb},0.22)`;
      c.beginPath();
      for (let i = 0; i < a.n; i++) {
        if (this.group[i] !== g) continue;
        const rr = a.flags[i] & 16 ? r : r * 0.6;
        c.moveTo(this.px[i] + rr, this.py[i]);
        c.arc(this.px[i], this.py[i], rr, 0, 6.2832);
      }
      c.fill();
    }
    this.base = base;
  }

  // a new second of activity arrived
  push(act) {
    for (let k = 0; k < act.idx.length; k++) {
      const i = act.idx[k];
      this.intensity[i] = Math.min(1.5, this.intensity[i] + 0.35 + 0.12 * act.cnt[k]);
      this.hot.add(i);
    }
  }

  tick() {
    const c = this.ctx;
    if (this.base) c.drawImage(this.base, 0, 0);
    const decay = this.reduced ? 0.7 : 0.9;
    const r = Math.max(1.2, 1.8 * (this.w / 900));
    c.globalCompositeOperation = 'lighter';
    for (let g = 0; g < GROUPS.length; g++) {
      if (!this.visible[g]) continue;
      const [cr, cg, cb] = this.colors[GROUPS[g].key];
      c.fillStyle = `rgb(${cr},${cg},${cb})`;
      c.beginPath();
      for (const i of this.hot) {
        if (this.group[i] !== g) continue;
        const v = this.intensity[i];
        const rr = r * (0.6 + v);
        c.moveTo(this.px[i] + rr, this.py[i]);
        c.arc(this.px[i], this.py[i], rr, 0, 6.2832);
      }
      c.globalAlpha = 0.9;
      c.fill();
    }
    c.globalAlpha = 1;
    c.globalCompositeOperation = 'source-over';
    for (const i of this.hot) {
      this.intensity[i] *= decay;
      if (this.intensity[i] < 0.03) {
        this.intensity[i] = 0;
        this.hot.delete(i);
      }
    }
    requestAnimationFrame(this.tick);
  }
}
