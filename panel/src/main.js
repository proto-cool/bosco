import '@arclux/arc-ui/base.css';
import './style.css';
import { BrainView, parseActivity, parseAtlas } from './brain.js';

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
    mb: rgbOf('--accent-primary'),
    dn: rgbOf('--accent-secondary'),
    rest: rgbOf('--text-secondary'),
  };
}

// ---- helpers ------------------------------------------------------------------------------------
const fmt = (n) => (n == null ? '–' : Number(n).toLocaleString('en-US'));
const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`);
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172800) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
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
function b(text) {
  return el('b', null, text);
}
// a sentence from parts: strings and <b> nodes
function sentence(node, parts) {
  node.replaceChildren(...parts.map((p) => (typeof p === 'string' ? document.createTextNode(p) : p)));
}
const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : many}`;
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
const short = (did) => did.replace('did:plc:', '').slice(0, 12);
function nameInto(node, did) {
  node.textContent = short(did);
  profile(did).then((p) => p && (node.textContent = `@${p.handle}`));
}

// ---- brain ------------------------------------------------------------------------------------
let brain = null;
let popNames = [];

async function initBrain() {
  const [meta, buf] = await Promise.all([getJSON('./atlas.json'), fetch('./atlas.bin').then((r) => r.arrayBuffer())]);
  const atlas = parseAtlas(buf, meta);
  popNames = meta.pops;
  brain = new BrainView($('brain'), atlas, colours());
  $('atlas-caption').textContent =
    `${fmt(meta.n)} neurons of the central brain at their somas; ${Math.round(meta.soma_coverage * 100)}% have one in the ` +
    `brain, the rest sit at the mean of their targets. the mushroom body in blue, descending and motor cells in ` +
    `violet, the rest in grey. no synapses are drawn. drag to turn him.`;
  for (const btn of document.querySelectorAll('.views button[data-view]')) {
    btn.addEventListener('click', () => {
      for (const o of document.querySelectorAll('.views button[data-view]')) o.setAttribute('aria-pressed', String(o === btn));
      brain.setView(Number(btn.dataset.view));
    });
  }
  $('orbit').addEventListener('click', () => {
    brain.orbit = !brain.orbit;
    brain.pauseUntil = 0;
    $('orbit').setAttribute('aria-pressed', String(brain.orbit));
  });
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
    track.append(el('i', `fill${p === 'leave' ? ' avoid' : ''}`), el('i', 'tick'));
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
    if (!brain) return; // the atlas is still loading; try this second again
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
  const br = st.brain;
  $('bio-time').textContent =
    `${alive(br.t_ms)} lived · ${br.lag_s > 120 ? `${Math.round(br.lag_s / 60)} min behind` : 'in step'}`;

  // today: one figure, one sentence
  const t = st.today, a = t.actions || {};
  $('silence').textContent = t.silence == null ? '–' : pct(t.silence);
  const replies = (a.reply || 0) + (a.identity || 0);
  if (t.episodes) {
    const acts = [
      [replies, 'reply', 'replies'],
      [a.like || 0, 'like', 'likes'],
      [a.follow || 0, 'follow', 'follows'],
      [a.spontaneous_post || 0, 'post of his own', 'posts of his own'],
    ].filter(([n]) => n > 0);
    const parts = ['he read ', b(plural(t.episodes, 'post', 'posts')), ' and said nothing to ',
      b(fmt(Math.round(t.episodes * (t.silence || 0)))), ' of them. '];
    if (acts.length) {
      acts.forEach(([n, one, many], i) => {
        parts.push(b(plural(n, one, many)), i < acts.length - 1 ? ', ' : '. ');
      });
    } else {
      parts.push('he did nothing else. ');
    }
    if (t.rewards || t.punishments) {
      parts.push(b(plural(t.rewards, 'reward', 'rewards')), ' and ', b(plural(t.punishments, 'punishment', 'punishments')), '.');
    } else {
      parts.push('nothing rewarded or punished him yet.');
    }
    sentence($('today-sentence'), parts);
  } else {
    $('today-sentence').textContent = 'nothing read yet today.';
  }
  const topics = Object.entries(st.topics_today || {}).sort((x, y) => y[1] - x[1]).slice(0, 6);
  $('topics-sentence').textContent = topics.length
    ? `smelled most: ${topics.map(([k, v]) => `${k} ×${v}`).join(', ')}.`
    : '';

  // memory: one figure, one sentence, a short list
  const people = st.people || [];
  $('people-count').textContent = fmt(people.length);
  const hunger =
    br.appetite == null ? '' : br.appetite > 0.7 ? ' he is hungry for company.' : br.appetite < 0.3 ? ' he has had his fill of company for now.' : ' he could take some company.';
  sentence(
    $('memory-sentence'),
    br.stm_depressed || br.ltm_depressed
      ? [b(fmt(br.stm_depressed)), ' short-term and ', b(fmt(br.ltm_depressed)), ' long-term traces across ',
        b(fmt(br.plastic_synapses)), ' plastic synapses. dust on his bristles ', b(pct(br.dust)), '.', hunger]
      : ['no traces yet across ', b(fmt(br.plastic_synapses)), ' plastic synapses. dust on his bristles ', b(pct(br.dust)), '.', hunger],
  );
  const list = $('people');
  list.replaceChildren(
    ...people.slice(0, 6).map((p) => {
      const li = el('li', 'person');
      const who = el('a', 'who');
      who.href = `https://bsky.app/profile/${p.did}`;
      nameInto(who, p.did);
      const bar = el('span', 'valence');
      const seg = el('i', p.learned >= 0 ? 'pos' : 'neg');
      const w = Math.min(50, Math.abs(p.learned) * 50);
      seg.style.width = `${w}%`;
      seg.style.left = p.learned >= 0 ? '50%' : `${50 - w}%`;
      bar.append(seg);
      li.append(who, bar, el('span', 'fam', `×${p.familiarity}`));
      return li;
    }),
  );
  if (!people.length) list.replaceChildren(el('li', 'empty', 'nobody yet. he has not met anyone he remembers.'));

  // his posts
  const uris = (st.posts || []).map((p) => p.uri).slice(0, 4);
  const got = await posts(uris);
  const byUri = new Map(got.map((p) => [p.uri, p]));
  const shown = (st.posts || []).slice(0, 4).filter((p) => byUri.has(p.uri));
  $('posts').replaceChildren(
    ...shown.map((p) => {
      const post = byUri.get(p.uri);
      const li = el('li');
      const link = el('a', 'post');
      link.href = postLink(p.uri);
      link.append(
        el('p', 't', post.record?.text || ''),
        el('span', 'm', `${p.kind.replace('_', ' ')} · ${ago(p.ts)} ago · ${plural(post.likeCount || 0, 'like', 'likes')}`),
      );
      li.append(link);
      return li;
    }),
  );
  if (!shown.length) $('posts').replaceChildren(el('li', 'empty', 'nothing yet.'));

  // the record
  $('recent').replaceChildren(
    ...(st.recent || []).slice(0, 12).map((w) => {
      const li = el('li', `win${w.action !== 'nothing' ? ' acted' : ''}`);
      const who = el('span', 'who');
      if (w.did) nameInto(who, w.did);
      else who.textContent = w.kind === 'spontaneous' ? 'a landing' : '–';
      li.append(
        el('span', null, ago(w.ts)),
        el('span', 'smell', w.kind === 'spontaneous' ? 'dust' : w.mentioned ? 'mention' : 'browse'),
        who,
        el('span', 'what', w.action === 'nothing' ? 'silence' : `${w.action.replace('_', ' ')}${w.acted ? '' : ' · withheld'}`),
      );
      return li;
    }),
  );

  $('fine').textContent =
    `memory ${br.digest.slice(0, 12)} · corpus ${st.corpus_digest.slice(0, 12)} · ` +
    `${fmt(br.neurons)} neurons · ${fmt(br.synapses)} synapses · status ${ago(st.generated)} ago`;
}

// ---- the rail's current row brightens as it passes the top ------------------------------------
const rows = [...document.querySelectorAll('.rail .row')];
const io = new IntersectionObserver(
  (entries) => {
    for (const e of entries) e.target.classList.toggle('is-current', e.isIntersecting);
  },
  { rootMargin: '-10% 0px -70% 0px' },
);
rows.forEach((r) => io.observe(r));

// ---- go ----------------------------------------------------------------------------------------
initBrain().catch((e) => {
  console.error(e);
  $('atlas-caption').textContent = 'the atlas did not load.';
});
pollStatus();
pollActivity();
setInterval(pollActivity, 1000);
setInterval(pollStatus, 20000);
