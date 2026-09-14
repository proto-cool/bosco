// Soma-position renderer.  Every neuron is a point at its soma in the volume; the cloud is
// rotated in three dimensions (a slow orbit, or a resting angle, or wherever it was dragged),
// projected orthographically, and painted per frame: a dim depth-shaded base layer of all
// neurons, then the cells with residual intensity on top, additively.  Nothing here is a
// graph: no edges, no layout.

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
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  const sc = new Uint8Array(n), flags = new Uint8Array(n), pop = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 9;
    // centred, isotropic, in units of the brain's largest extent (see build_panel_atlas.py)
    x[i] = (rec.getUint16(o, true) - 32767.5) / 65535;
    y[i] = (rec.getUint16(o + 2, true) - 32767.5) / 65535;
    z[i] = (rec.getUint16(o + 4, true) - 32767.5) / 65535;
    sc[i] = rec.getUint8(o + 6);
    flags[i] = rec.getUint8(o + 7);
    pop[i] = rec.getUint8(o + 8);
  }
  return { n, x, y, z, sc, flags, pop, meta };
}

// Colour groups, in draw order.  A neuron belongs to the first group that claims it.
export const GROUPS = [
  { key: 'mb', name: 'mushroom body', test: (a, i) => a.flags[i] & 7 }, // kenyon cells, output neurons, dopamine
  { key: 'dn', name: 'descending and motor', test: (a, i) => (a.flags[i] & 8) || a.sc[i] === 2 },
  { key: 'rest', name: 'the rest', test: () => true },
];

// resting angles: [yaw about the vertical axis, pitch about the horizontal]
export const VIEWS = [
  [0, 0], // front: anterior toward the viewer
  [0, Math.PI / 2], // top: dorsal toward the viewer
  [Math.PI / 2, 0], // side
];

const TWO_PI = 6.2832;

export class BrainView {
  constructor(canvas, atlas, colors) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.atlas = atlas;
    this.colors = colors; // key -> [r,g,b]
    this.group = new Uint8Array(atlas.n);
    for (let i = 0; i < atlas.n; i++) this.group[i] = GROUPS.findIndex((g) => g.test(atlas, i));
    this.intensity = new Float32Array(atlas.n);
    this.hot = new Set();
    this.px = new Float32Array(atlas.n);
    this.py = new Float32Array(atlas.n);
    this.depth = new Float32Array(atlas.n); // 0 far .. 1 near
    this.yaw = 0;
    this.pitch = 0;
    this.target = null; // [yaw, pitch] to ease toward
    this.orbit = false; // slow turn about the vertical axis; a toggle on the page, off by default
    this.orbitRate = TWO_PI / 90; // one revolution in 90 s
    this.pauseUntil = 0;
    this.hold = false; // a still: lit cells do not decay
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.dpr = 1;
    this.lastT = performance.now();
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas);
    this.resize();
    this.bindPointer();
    this.tick = this.tick.bind(this);
    requestAnimationFrame(this.tick);
  }

  setView(v) {
    this.target = VIEWS[v] || VIEWS[0];
    this.pauseUntil = performance.now() + 20000; // rest there a while before the orbit resumes
  }

  bindPointer() {
    let last = null;
    const c = this.canvas;
    c.style.touchAction = 'none';
    c.addEventListener('pointerdown', (e) => {
      last = [e.clientX, e.clientY];
      c.setPointerCapture(e.pointerId);
      this.target = null;
      this.pauseUntil = Infinity;
    });
    c.addEventListener('pointermove', (e) => {
      if (!last) return;
      this.yaw += (e.clientX - last[0]) * 0.008;
      this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch + (e.clientY - last[1]) * 0.008));
      last = [e.clientX, e.clientY];
    });
    const up = () => {
      if (!last) return;
      last = null;
      this.pauseUntil = performance.now() + 30000;
    };
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.dpr = dpr;
    this.w = Math.max(1, Math.round(r.width * dpr));
    this.h = Math.max(1, Math.round(r.height * dpr));
    this.canvas.width = this.w;
    this.canvas.height = this.h;
    this.img = this.ctx.createImageData(this.w, this.h);
    this.fit = 0.92 * Math.min(this.w, this.h); // pixels per unit of the brain's largest extent
  }

  // rotate and project every soma for the current angles
  project() {
    const a = this.atlas;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const s = this.fit, ox = this.w / 2, oy = this.h / 2;
    for (let i = 0; i < a.n; i++) {
      const x = a.x[i], y = a.y[i], z = a.z[i];
      // yaw about the vertical axis, then pitch about the horizontal
      const x1 = x * cy + z * sy;
      const z1 = -x * sy + z * cy;
      const y2 = y * cp - z1 * sp;
      const z2 = y * sp + z1 * cp;
      this.px[i] = ox + x1 * s;
      this.py[i] = oy + y2 * s;
      this.depth[i] = 0.5 - z2; // anterior (negative z) is near in the front view
    }
  }

  // the dim base layer: one small block per neuron, shaded by depth, into an image buffer
  paintBase() {
    const d = this.img.data;
    d.fill(0);
    const a = this.atlas;
    const w = this.w, h = this.h;
    const sz = this.dpr >= 2 || this.w >= 700 ? 2 : 1; // two-pixel somas on a wide canvas read as a body, not a mist
    for (let i = 0; i < a.n; i++) {
      const x = this.px[i] | 0, y = this.py[i] | 0;
      if (x < 0 || y < 0 || x + sz > w || y + sz > h) continue;
      const g = this.group[i];
      const [cr, cg, cb] = this.colors[GROUPS[g].key];
      const soma = a.flags[i] & 16 ? 1 : 0.55;
      const near = this.depth[i];
      const alpha = (g === 2 ? 0.22 : 0.8) * soma * (0.3 + 0.7 * near);
      const av = Math.round(alpha * 255);
      for (let dy = 0; dy < sz; dy++) {
        let o = ((y + dy) * w + x) * 4;
        for (let dx = 0; dx < sz; dx++, o += 4) {
          if (d[o + 3] < av) {
            d[o] = cr;
            d[o + 1] = cg;
            d[o + 2] = cb;
            d[o + 3] = av;
          }
        }
      }
    }
    this.ctx.putImageData(this.img, 0, 0);
  }

  // a new second of activity arrived
  push(act) {
    for (let k = 0; k < act.idx.length; k++) {
      const i = act.idx[k];
      this.intensity[i] = Math.min(1.5, this.intensity[i] + 0.35 + 0.12 * act.cnt[k]);
      this.hot.add(i);
    }
  }

  tick(now) {
    const dt = Math.min(0.1, (now - this.lastT) / 1000);
    this.lastT = now;
    if (this.target) {
      const [ty, tp] = this.target;
      // ease the yaw along the shorter way round
      let dy = ((ty - this.yaw + Math.PI) % TWO_PI) - Math.PI;
      if (dy < -Math.PI) dy += TWO_PI;
      this.yaw += dy * 0.08;
      this.pitch += (tp - this.pitch) * 0.08;
      if (Math.abs(dy) < 0.002 && Math.abs(tp - this.pitch) < 0.002) {
        this.yaw = ty;
        this.pitch = tp;
        this.target = null;
      }
    } else if (this.orbit && !this.reduced && now > this.pauseUntil) {
      this.yaw += this.orbitRate * dt;
      this.pitch += (0 - this.pitch) * 0.01; // settle level as it turns
    }
    this.project();
    this.paintBase();
    const c = this.ctx;
    const decay = this.reduced ? 0.7 : 0.9;
    const r = Math.max(1.2, 1.8 * (this.w / this.dpr / 900)) * this.dpr;
    c.globalCompositeOperation = 'lighter';
    for (let g = 0; g < GROUPS.length; g++) {
      const rest = GROUPS[g].key === 'rest';
      const [cr, cg, cb] = this.colors[GROUPS[g].key];
      c.fillStyle = `rgb(${cr},${cg},${cb})`;
      c.beginPath();
      for (const i of this.hot) {
        if (this.group[i] !== g) continue;
        const v = this.intensity[i];
        const rr = r * (0.6 + v) * (rest ? 0.75 : 1) * (0.7 + 0.5 * this.depth[i]);
        c.moveTo(this.px[i] + rr, this.py[i]);
        c.arc(this.px[i], this.py[i], rr, 0, TWO_PI);
      }
      c.globalAlpha = rest ? 0.45 : 0.9; // the mushroom body and the motor side read over the crowd
      c.fill();
    }
    c.globalAlpha = 1;
    c.globalCompositeOperation = 'source-over';
    if (!this.hold) {
      for (const i of this.hot) {
        this.intensity[i] *= decay;
        if (this.intensity[i] < 0.03) {
          this.intensity[i] = 0;
          this.hot.delete(i);
        }
      }
    }
    requestAnimationFrame(this.tick);
  }
}
