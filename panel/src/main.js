import '@arclux/arc-ui/base.css';
import '@arclux/arc-ui/badge';
import '@arclux/arc-ui/tabs';
import '@arclux/arc-ui/tab';
import '@arclux/arc-ui/list';
import '@arclux/arc-ui/list-item';
import './style.css';
import { BrainView, GROUPS, parseActivity, parseAtlas } from './brain.js';

const DATA = './data';
const PUBLIC_API = 'https://public.api.bsky.app/xrpc';
const $ = (id) => document.getElementById(id);

// ---- colours from tokens (any colour syntax -> rgb triplet via a 1x1 canvas) --------------
function rgbOf(token) {
  const probe = document.createElement('span');
  probe.style.color = `var(${token})`;
  document.body.appendChild(probe);
  const css = getComputedStyle(probe).color;
  probe.remove();
  const c = document.createElement('canvas');
  c.width = c.height = 1;
  const g = c.getContext('2d');
  g.fillStyle = css;
  g.fillRect(0, 0, 1, 1);
  const d = g.getImageData(0, 0, 1, 1).data;
  return [d[0], d[1], d[2]];
}
function colours() {
  const bg = rgbOf('--bg-deep');
  return {
    bg: `rgb(${bg.join(',')})`,
    kc: rgbOf('--chart-1'),
    mbon: rgbOf('--chart-2'),
    dan: rgbOf('--chart-3'),
    dn: rgbOf('--chart-4'),
    sensory: rgbOf('--chart-5'),
    other: rgbOf('--text-muted'),
  };
}

// ---- helpers ------------------------------------------------------------------------------------
const fmt = (n) => (n == null ? '–' : Number(n).toLocaleString('en-US'));
const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`);
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
function alive(ms) {
  const h = ms / 3.6e6;
  const d = Math.floor(h / 24);
  return d ? `${d}d ${Math.floor(h % 24)}h` : `${Math.floor(h)}h ${Math.floor((h % 1) * 60)}m`;
}
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

// ---- profiles and posts from the public API (nothing of other people's is stored server-side) ----
const profiles = new Map();
async function profile(did) {
  if (profiles.has(did)) return profiles.get(did);
  const p = getJSON(`${PUBLIC_API}/app.bsky.actor.getProfile?actor=${encodeURIComponent(did)}`).catch(() => null);
  profiles.set(did, p);
  return p;
}
async function posts(uris) {
  if (!uris.length) return [];
  const q = uris.map((u) => `uris=${encodeURIComponent(u)}`).join('&');
  const r = await getJSON(`${PUBLIC_API}/app.bsky.feed.getPosts?${q}`).catch(() => ({ posts: [] }));
  return r.posts || [];
}
const postLink = (uri) => {
  const m = uri.match(/^at:\/\/([^/]+)\/app\.bsky\.feed\.post\/(.+)$/);
  return m ? `https://bsky.app/profile/${m[1]}/post/${m[2]}` : '#';
};

// ---- brain ------------------------------------------------------------------------------------
let brain = null;
let popNames = [];

async function initBrain() {
  const [meta, buf] = await Promise.all([getJSON('./atlas.json'), fetch('./atlas.bin').then((r) => r.arrayBuffer())]);
  const atlas = parseAtlas(buf, meta);
  popNames = meta.pops;
  const col = colours();
  brain = new BrainView($('brain'), atlas, col);
  $('atlas-caption').textContent =
    `${fmt(meta.n)} neurons of the central brain at their somas ` +
    `(${Math.round(meta.soma_coverage * 100)}% have one in the volume; the rest sit at the mean of their targets).`;
  const legend = $('legend');
  GROUPS.forEach((g, k) => {
    const b = el('button', null, null);
    b.setAttribute('aria-pressed', 'true');
    const dot = el('span', 'dot');
    dot.style.setProperty('--_rgb', col[g.key].join(','));
    b.append(dot, el('span', null, g.name));
    b.addEventListener('click', () => {
      const on = b.getAttribute('aria-pressed') !== 'true';
      b.setAttribute('aria-pressed', String(on));
      brain.setVisible(k, on);
    });
    legend.append(b);
  });
  $('view').addEventListener('arc-change', (e) => brain.setView(Number(e.detail?.value ?? 0)));
  buildReadout();
}

function buildReadout() {
  const box = $('readout');
  box.replaceChildren();
  for (const p of popNames) {
    const row = el('div', 'ro');
    row.dataset.pop = p;
    row.append(el('span', 'k', p));
    const track = el('div', 'track');
    const fill = el('i', `fill${p === 'leave' ? ' avoid' : ''}`);
    track.append(fill, el('i', 'tick'));
    row.append(track, el('span', 'v', '–'));
    box.append(row);
  }
}

let thresholds = {};
let lastT = -1;
async function pollActivity() {
  try {
    const r = await fetch(`${DATA}/activity.bin`, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    const act = parseActivity(await r.arrayBuffer());
    setLive(true);
    if (act.tMs !== lastT) {
      lastT = act.tMs;
      brain?.push(act);
      $('firing').textContent = fmt(act.idx.length);
      $('kc-active').textContent = fmt(act.kcActive);
      popNames.forEach((p, k) => {
        const row = document.querySelector(`.ro[data-pop="${p}"]`);
        if (!row) return;
        const rate = act.pops[k], th = thresholds[p];
        const ratio = th ? rate / th : 0;
        row.querySelector('.fill').style.transform = `scaleX(${Math.min(1, ratio / 1.6)})`;
        row.querySelector('.v').textContent = `${rate.toFixed(1)} Hz`;
        row.classList.toggle('over', th != null && rate > th);
      });
    }
  } catch {
    setLive(false);
  }
}

function setLive(on) {
  $('live').classList.toggle('is-live', on);
  $('live-text').textContent = on ? 'live' : 'asleep or unreachable';
}

// ---- status ------------------------------------------------------------------------------------
function stat(label, value, sub) {
  const s = el('div', 'stat');
  s.append(el('span', 'numeral', value), el('span', 'stat-label', label));
  if (sub) s.append(el('span', 'stat-label', sub));
  return s;
}

async function pollStatus() {
  let st;
  try {
    st = await getJSON(`${DATA}/status.json`);
  } catch {
    return;
  }
  thresholds = st.readout?.thresholds || {};
  if (st.identity?.handle) {
    $('handle').textContent = `@${st.identity.handle}`;
    $('handle').href = `https://bsky.app/profile/${st.identity.handle}`;
  }
  const b = st.brain;
  $('bio-time').textContent = `${alive(b.t_ms)} lived · ${b.lag_s > 120 ? `${Math.round(b.lag_s / 60)} min behind` : 'in step'} · ${b.wall_per_bio.toFixed(2)} wall s per s`;
  $('dev').hidden = false;

  const t = st.today, a = t.actions || {};
  $('today').replaceChildren(
    stat('posts read', fmt(t.episodes)),
    stat('silence', pct(t.silence)),
    stat('replies', fmt((a.reply || 0) + (a.identity || 0))),
    stat('likes', fmt(a.like || 0)),
    stat('follows', fmt(a.follow || 0)),
    stat('own posts', fmt(a.spontaneous_post || 0)),
    stat('rewards', fmt(t.rewards)),
    stat('punishments', fmt(t.punishments)),
  );
  const topics = Object.entries(st.topics_today || {}).sort((x, y) => y[1] - x[1]).slice(0, 14);
  $('topics').replaceChildren(
    ...topics.map(([k, v]) => {
      const c = el('span', 'chip');
      c.append(el('b', null, k), document.createTextNode(` ${v}`));
      return c;
    }),
  );

  $('memory').replaceChildren(
    stat('short-term traces', fmt(b.stm_depressed), `of ${fmt(b.plastic_synapses)} plastic synapses`),
    stat('long-term traces', fmt(b.ltm_depressed)),
    stat('people he can smell', fmt((st.people || []).length)),
    stat('dust on his bristles', pct(b.dust)),
  );

  // people
  const list = $('people');
  const rows = (st.people || []).slice(0, 12);
  list.replaceChildren(
    ...rows.map((p) => {
      const item = document.createElement('arc-list-item');
      const row = el('div', 'person');
      const av = el('span', 'noavatar');
      const who = el('span', `who${p.familiarity ? ' known' : ''}`, p.did.replace('did:plc:', '').slice(0, 14));
      const bar = el('div', 'valence');
      const seg = el('i', p.learned >= 0 ? 'pos' : 'neg');
      seg.style.width = `${Math.min(50, Math.abs(p.learned) * 50)}%`;
      bar.append(seg);
      row.append(av, who, bar, el('span', 'fam', `×${p.familiarity}`));
      item.append(row);
      profile(p.did).then((pr) => {
        if (!pr) return;
        who.textContent = `@${pr.handle}`;
        if (pr.avatar) {
          const img = document.createElement('img');
          img.src = pr.avatar;
          img.alt = '';
          img.loading = 'lazy';
          av.replaceWith(img);
        }
      });
      return item;
    }),
  );
  if (!rows.length) list.replaceChildren(el('p', 'caption', 'nobody yet. he has not met anyone he remembers.'));

  // his posts
  const uris = (st.posts || []).map((p) => p.uri).slice(0, 8);
  const got = await posts(uris);
  const byUri = new Map(got.map((p) => [p.uri, p]));
  $('posts').replaceChildren(
    ...(st.posts || [])
      .slice(0, 8)
      .filter((p) => byUri.has(p.uri))
      .map((p) => {
        const post = byUri.get(p.uri);
        const link = el('a', 'post');
        link.href = postLink(p.uri);
        link.append(el('p', 't', post.record?.text || ''), el('span', 'm', `${p.kind.replace('_', ' ')} · ${ago(p.ts)} · ${post.likeCount || 0} likes`));
        return link;
      }),
  );
  if (!uris.length) $('posts').replaceChildren(el('p', 'caption', 'nothing yet.'));

  // recent windows
  $('recent').replaceChildren(
    ...(st.recent || []).slice(0, 20).map((w) => {
      const row = el('div', `win${w.action !== 'nothing' ? ' acted' : ''}`);
      row.append(
        el('span', null, ago(w.ts)),
        el('span', null, w.kind === 'spontaneous' ? 'landing' : w.mentioned ? 'mention' : 'browse'),
        el('span', 'did', w.did ? w.did.replace('did:plc:', '') : '–'),
        el('span', 'why', w.topics?.length ? w.topics.join(' ') : w.note || ''),
        el('span', null, w.action === 'nothing' ? 'silence' : `${w.behaviour} → ${w.action}`),
      );
      if (w.did) profile(w.did).then((pr) => pr && (row.querySelector('.did').textContent = `@${pr.handle}`));
      return row;
    }),
  );

  $('fine').textContent =
    `memory ${b.digest.slice(0, 12)} · corpus ${st.corpus_digest.slice(0, 12)} · ` +
    `${fmt(b.neurons)} neurons · ${fmt(b.synapses)} synapses · status ${ago(st.generated)}`;
}

// ---- go ----------------------------------------------------------------------------------------
initBrain().catch((e) => {
  console.error(e);
  $('atlas-caption').textContent = 'the atlas did not load.';
});
pollStatus();
pollActivity();
setInterval(pollActivity, 1000);
setInterval(pollStatus, 20000);
