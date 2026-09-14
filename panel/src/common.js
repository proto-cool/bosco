// Shared by every page of bosco.proto.cool: formatting, the public API, the top line.
import '@arclux/arc-ui/base.css';
import './style.css';

export const DATA = './data';
export const PUBLIC_API = 'https://public.api.bsky.app/xrpc';
export const $ = (id) => document.getElementById(id);

// ---- formatting --------------------------------------------------------------------------------
export const fmt = (n) => (n == null ? '–' : Number(n).toLocaleString('en-US'));
export const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`);
export function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172800) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
}
export function alive(ms) {
  const h = ms / 3.6e6;
  const d = Math.floor(h / 24);
  return d ? `${d}d ${Math.floor(h % 24)}h` : `${Math.floor(h)}h ${Math.floor((h % 1) * 60)}m`;
}
// a clock time in his time zone
export function clockTime(ts, tz) {
  try {
    return new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: tz }).format(new Date(ts * 1000));
  } catch {
    return new Date(ts * 1000).toISOString().slice(11, 16);
  }
}
export function longDate(iso) {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', timeZone: 'UTC' }).toLowerCase();
}
export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
export const b = (text) => el('b', null, text);
// a sentence from parts: strings and nodes
export function sentence(node, parts) {
  node.replaceChildren(...parts.map((p) => (typeof p === 'string' ? document.createTextNode(p) : p)));
}
export const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : many}`;
// "a, b and c"
export function list(items) {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

// ---- fetching ------------------------------------------------------------------------------------
export async function getJSON(url, timeoutMs = 8000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { cache: 'no-store', signal: ctl.signal });
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return await r.json();
  } finally {
    clearTimeout(t);
  }
}
export async function getBytes(url, timeoutMs = 5000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { cache: 'no-store', signal: ctl.signal });
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return await r.arrayBuffer();
  } finally {
    clearTimeout(t);
  }
}

// ---- profiles and posts from the public API (nothing of other people's is stored server-side) ----
const profiles = new Map();
export async function profile(did) {
  if (profiles.has(did)) return profiles.get(did);
  const p = getJSON(`${PUBLIC_API}/app.bsky.actor.getProfile?actor=${encodeURIComponent(did)}`).catch(() => null);
  profiles.set(did, p);
  return p;
}
export async function posts(uris) {
  if (!uris.length) return [];
  const out = [];
  for (let i = 0; i < uris.length; i += 25) {
    const q = uris.slice(i, i + 25).map((u) => `uris=${encodeURIComponent(u)}`).join('&');
    const r = await getJSON(`${PUBLIC_API}/app.bsky.feed.getPosts?${q}`).catch(() => ({ posts: [] }));
    out.push(...(r.posts || []));
  }
  return out;
}
export const postLink = (uri) => {
  const m = uri.match(/^at:\/\/([^/]+)\/app\.bsky\.feed\.post\/(.+)$/);
  return m ? `https://bsky.app/profile/${m[1]}/post/${m[2]}` : '#';
};
export const short = (did) => did.replace('did:plc:', '').slice(0, 12);
export function nameInto(node, did) {
  node.textContent = short(did);
  profile(did).then((p) => p && (node.textContent = `@${p.handle}`));
}

// ---- his posts, as a list --------------------------------------------------------------------
export async function renderPosts(node, items, limit = 4) {
  const uris = items.map((p) => p.uri).slice(0, limit);
  const got = await posts(uris);
  const byUri = new Map(got.map((p) => [p.uri, p]));
  const shown = items.slice(0, limit).filter((p) => byUri.has(p.uri));
  node.replaceChildren(
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
  if (!shown.length) node.replaceChildren(el('li', 'empty', 'nothing yet.'));
  return shown.length;
}

// ---- people rows ------------------------------------------------------------------------------
// the valence mark: a segment from the centre, right and blue for sweet, left and violet for
// bitter, full at ±VALENCE_FULL (his verdicts live well inside ±1); the value beside it in mono.
export const VALENCE_FULL = 0.5;
export function personRow(p, extra) {
  const li = el('li', 'person');
  const who = el('a', 'who');
  who.href = `https://bsky.app/profile/${p.did}`;
  nameInto(who, p.did);
  const bar = el('span', 'valence');
  bar.title = `learned valence ${p.learned >= 0 ? '+' : ''}${p.learned.toFixed(2)}`;
  const seg = el('i', p.learned >= 0 ? 'pos' : 'neg');
  const w = Math.min(50, (Math.abs(p.learned) / VALENCE_FULL) * 50);
  seg.style.width = `${w}%`;
  seg.style.left = p.learned >= 0 ? '50%' : `${50 - w}%`;
  bar.append(seg);
  const val = el('span', 'val', `${p.learned >= 0 ? '+' : '−'}${Math.abs(p.learned).toFixed(2)}`);
  if (Math.abs(p.learned) < 0.005) val.textContent = '0';
  li.append(who, bar, val, el('span', 'fam', extra ?? `×${p.familiarity}`));
  return li;
}

// ---- the top line ----------------------------------------------------------------------------
// the live indicator: what the last activity file says about him
export function liveText(state, ageS) {
  if (state === 'live') return 'live';
  if (state === 'busy') return `between seconds · ${Math.round(ageS)}s`;
  if (state === 'stale') return `last second ${ago(Date.now() / 1000 - ageS)} ago`;
  return 'asleep or unreachable';
}
export function setLive(state, ageS) {
  const live = $('live');
  if (!live) return;
  live.classList.toggle('is-live', state === 'live');
  live.classList.toggle('is-busy', state === 'busy');
  $('live-text').textContent = liveText(state, ageS);
}
// the current page's word in the top line
export function markNav() {
  const here = location.pathname.split('/').pop() || 'index.html';
  for (const a of document.querySelectorAll('.nav a')) {
    const target = a.getAttribute('href').split('/').pop() || 'index.html';
    if (target === here || (here === '' && target === 'index.html')) a.setAttribute('aria-current', 'page');
  }
}
// the ladder on the left of a sheet lifts as its row passes the top (same rule as the rail)
export function trackCurrent(rows, onCurrent) {
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        e.target.classList.toggle('is-current', e.isIntersecting);
        if (e.isIntersecting && onCurrent) onCurrent(e.target);
      }
    },
    { rootMargin: '-10% 0px -70% 0px' },
  );
  rows.forEach((r) => io.observe(r));
  return io;
}
// the top line's handle and the "in development" word, from status.json when a page has it
export function applyIdentity(st) {
  if (st?.identity?.handle && $('handle')) {
    $('handle').textContent = `@${st.identity.handle}`;
    $('handle').href = `https://bsky.app/profile/${st.identity.handle}`;
  }
}
