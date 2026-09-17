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
  const p = getJSON(`${PUBLIC_API}/app.bsky.actor.getProfile?actor=${encodeURIComponent(did)}`, 20000).catch(() => null);
  profiles.set(did, p);
  return p;
}
export async function posts(uris) {
  if (!uris.length) return [];
  const out = [];
  for (let i = 0; i < uris.length; i += 25) {
    const q = uris.slice(i, i + 25).map((u) => `uris=${encodeURIComponent(u)}`).join('&');
    const r = await getJSON(`${PUBLIC_API}/app.bsky.feed.getPosts?${q}`, 20000).catch(() => ({ posts: [] }));
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

// ---- a post, the way bluesky shows it -----------------------------------------------------------
// Everything in the card comes from the public API at view time: the server holds the URI and
// nothing else, no text, no handle, no picture (PRODUCT.md, nothing of anyone else's is stored).
// The card is the page's one box, because what is inside it belongs to the network and not to us;
// his own note about it stays outside, on the ground (DESIGN.md, the quoted-specimen rule).

const SVG_NS = 'http://www.w3.org/2000/svg';
// One icon system, drawn on a 24 grid at a 2px stroke; `play` is the only solid one.
const ICONS = {
  reply: ['M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'],
  repost: ['M17 1l4 4-4 4', 'M3 11V9a4 4 0 0 1 4-4h14', 'M7 23l-4-4 4-4', 'M21 13v2a4 4 0 0 1-4 4H3'],
  like: ['M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1.1L12 21.2l7.8-7.8 1-1a5.5 5.5 0 0 0 0-7.8z'],
  out: ['M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6', 'M15 3h6v6', 'M10 14 21 3'],
  play: ['M8 5v14l11-7z'],
};
export function icon(name, cls) {
  const s = document.createElementNS(SVG_NS, 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('aria-hidden', 'true');
  s.setAttribute('focusable', 'false');
  if (cls) s.setAttribute('class', cls);
  for (const d of ICONS[name] || []) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    s.append(path);
  }
  return s;
}

// what the AppView hung on the post: pictures, a video, a link card, or a post he quoted
function mediaOf(embed) {
  if (!embed) return null;
  const t = embed.$type || '';
  if (t.startsWith('app.bsky.embed.recordWithMedia')) return mediaOf(embed.media) || mediaOf(embed.record);
  if (t.startsWith('app.bsky.embed.images')) return { kind: 'images', images: (embed.images || []).slice(0, 4) };
  if (t.startsWith('app.bsky.embed.video')) return { kind: 'video', thumb: embed.thumbnail, alt: embed.alt || '' };
  if (t.startsWith('app.bsky.embed.external')) return { kind: 'external', ...(embed.external || {}) };
  if (t.startsWith('app.bsky.embed.record')) return { kind: 'quote', handle: embed.record?.author?.handle };
  return null;
}

function frame(src, alt, cls) {
  const f = el('div', cls ? `frame ${cls}` : 'frame');
  if (src) {
    const img = el('img');
    img.src = src;
    img.alt = alt || '';
    img.loading = 'lazy';
    img.decoding = 'async';
    f.append(img);
  }
  // bluesky's own badge: this picture was described, and the description is the img's alt
  if (alt) f.append(el('span', 'alt-badge', 'alt'));
  return f;
}

function mediaNode(m) {
  if (!m) return null;
  if (m.kind === 'images' && m.images.length) {
    const n = Math.min(4, m.images.length);
    const grid = el('div', `shot n${n}`);
    for (const im of m.images) {
      const f = frame(im.thumb || im.fullsize, im.alt);
      const r = im.aspectRatio;
      // one picture keeps its own shape, clamped the way bluesky clamps it: a very tall or very
      // wide one is cropped to the frame rather than running the length of the rail
      if (n === 1 && r?.width && r?.height) f.style.aspectRatio = `${Math.min(Math.max(r.width / r.height, 0.8), 1.78)}`;
      grid.append(f);
    }
    return grid;
  }
  if (m.kind === 'video' && m.thumb) {
    const grid = el('div', 'shot n1');
    const f = frame(m.thumb, m.alt);
    const play = el('span', 'play');
    play.append(icon('play', 'solid'));
    f.append(play);
    grid.append(f);
    return grid;
  }
  if (m.kind === 'external' && (m.title || m.uri)) {
    const card = el('div', 'link-card');
    if (m.thumb) card.append(frame(m.thumb, '', 'thumb'));
    const t = el('div', 'link-text');
    t.append(el('span', 'link-title', m.title || m.uri), el('span', 'link-site', siteOf(m.uri)));
    card.append(t);
    return card;
  }
  if (m.kind === 'quote' && m.handle) return el('p', 'quote-note', `quoting @${m.handle}`);
  return null;
}

const siteOf = (uri) => {
  try {
    return new URL(uri).hostname.replace(/^www\./, '');
  } catch {
    return '';
  }
};

function count(name, n, one, many) {
  const c = el('span', 'count');
  c.append(icon(name), el('span', null, fmt(n || 0)));
  c.setAttribute('aria-label', plural(n || 0, one, many));
  return c;
}

// One post as bluesky draws it: who, what, what came with it, and how the network answered.
export function postCard(post) {
  const card = el('article', 'post-card');
  const author = post.author || {};
  const handle = author.handle || '';

  const who = el('header', 'who');
  const avatar = el('span', 'avatar');
  if (author.avatar) {
    const img = el('img');
    img.src = author.avatar;
    img.alt = '';
    img.loading = 'lazy';
    avatar.append(img);
  }
  const names = el('span', 'names');
  names.append(el('span', 'name', author.displayName || handle || 'someone'), el('span', 'handle', handle ? `@${handle}` : ''));
  const made = post.record?.createdAt || post.indexedAt;
  const when = el('time', 'when', made ? ago(new Date(made).getTime() / 1000) : '');
  if (made) when.dateTime = made;
  who.append(avatar, names, when);
  card.append(who);

  const text = post.record?.text || '';
  if (text) card.append(el('p', 'text', text));
  const media = mediaNode(mediaOf(post.embed));
  if (media) card.append(media);

  const foot = el('footer', 'foot');
  const counts = el('span', 'counts');
  counts.append(
    count('reply', post.replyCount, 'reply', 'replies'),
    count('repost', (post.repostCount || 0) + (post.quoteCount || 0), 'repost', 'reposts'),
    count('like', post.likeCount, 'like', 'likes'),
  );
  const open = el('a', 'open');
  open.href = postLink(post.uri);
  open.target = '_blank';
  open.rel = 'noopener noreferrer';
  open.setAttribute('aria-label', `open ${handle ? `@${handle}'s` : 'this'} post on bluesky`);
  open.append(el('span', null, 'bluesky'), icon('out'));
  foot.append(counts, open);
  card.append(foot);
  return card;
}

// ---- his posts, and the one he liked most, as a list ---------------------------------------------
// what the ledger says he did, in his voice; the card underneath is the network's own record, so
// the note never repeats what the card already shows (the author, the clock, how it was answered)
export const SAID = {
  spontaneous_post: 'he posted',
  reply: 'he answered',
  answer: 'he answered by reflex',
  identity: 'he said who he is',
  intro: 'he introduced himself',
  liked: 'he liked this',
};
export const saidNote = (kind) => SAID[kind] || kind.replaceAll('_', ' ');
export async function renderPosts(node, items, limit = 4) {
  const uris = items.map((p) => p.uri).slice(0, limit);
  const got = await posts(uris);
  const byUri = new Map(got.map((p) => [p.uri, p]));
  const shown = items.slice(0, limit).filter((p) => byUri.has(p.uri));
  node.replaceChildren(
    ...shown.map((p) => {
      const li = el('li', 'post-item');
      // the note is his record's; an item can pass label: '' where the sentence above already said it
      const note = p.label === undefined ? `${saidNote(p.kind)} · ${ago(p.ts)} ago` : p.label;
      if (note) li.append(el('p', 'label', note));
      li.append(postCard(byUri.get(p.uri)));
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

// ---- what came of a window --------------------------------------------------------------------
// the words of the record. a window carries the network's decision (`action`), the real action
// that went out (`done`, if any) and, when nothing did, the rail that stopped it (`why`).
const DONE = {
  like: 'like',
  follow: 'follow',
  reply: 'reply',
  answer: 'answered by reflex',
  identity: 'said who he is',
  walk: 'walk',
  leave: 'unfollow',
  spontaneous_post: 'a post',
  intro: 'introduced himself',
};
export const CHOSE = {
  like: 'like',
  follow: 'follow',
  reply: 'reply',
  walk: 'walk',
  leave: 'unfollow',
  spontaneous_post: 'post',
  nothing: 'silence',
};
// "like/hour" with the caps from status.json -> "cap 6 an hour"
export function capText(which, caps) {
  const [name, per] = (which || '').split('/');
  const perWord = per === 'day' ? 'a day' : 'an hour';
  if (name === 'global') {
    const n = caps?.global?.[per];
    return n ? `cap ${n} acts ${perWord}` : 'cap on everything';
  }
  if (name === 'thread') return 'cap for one thread';
  if (name === 'account-replies' || name === 'account-actions') return 'cap toward one account';
  const n = caps?.per_kind?.[name]?.[per];
  return n ? `cap ${n} ${perWord}` : 'a cap';
}
export function whyText(why, caps) {
  if (!why) return '';
  if (why.startsWith('cap:')) return capText(why.slice(4), caps);
  return {
    not_addressed: 'not addressed to him',
    answered: 'already answered',
    not_following: 'nobody to unfollow',
    asleep: 'asleep',
    cap: 'a cap was full',
    already_following: 'already follows them',
    labeled: 'a labeled post',
    unrecorded: 'reason not recorded',
  }[why] || why.replaceAll('_', ' ');
}
export function outcomeText(w, caps) {
  if (w.done) {
    if (w.action === 'follow' && w.done === 'walk') return 'already follows · walk';
    return DONE[w.done] || w.done.replace('_', ' ');
  }
  if (w.action === 'nothing') return w.labeled ? 'silence · labeled post' : 'silence';
  const what = CHOSE[w.action] || w.action.replace('_', ' ');
  return `${what} withheld: ${whyText(w.why, caps)}`;
}
// one row of the record: when, where it came from, who, what came of it
export function windowRow(w, when, caps) {
  const li = el('li', `win${w.acted ? ' acted' : ''}`);
  const who = el('span', 'who');
  if (w.did) nameInto(who, w.did);
  else who.textContent = w.kind === 'spontaneous' ? 'a landing' : '–';
  li.append(
    el('span', null, when),
    el('span', 'smell', w.kind === 'spontaneous' ? 'dust' : w.mentioned ? 'mention' : w.feed || 'browse'),
    who,
    el('span', 'what', outcomeText(w, caps)),
  );
  return li;
}
// the one-sentence day: what he read, what he did about it, what reached him
export function daySentence(node, c, tense = 'past') {
  const a = c.actions || {};
  if (!c.episodes) {
    node.textContent = tense === 'today' ? 'nothing read yet today.' : 'he read nothing that day.';
    return;
  }
  const replies = (a.reply || 0) + (a.answer || 0) + (a.identity || 0);
  const acts = [
    [a.like || 0, 'like', 'likes'],
    [a.follow || 0, 'follow', 'follows'],
    [a.walk || 0, 'walk', 'walks'],
    [replies, 'reply', 'replies'],
    [a.leave || 0, 'unfollow', 'unfollows'],
    [a.spontaneous_post || 0, 'post of his own', 'posts of his own'],
  ].filter(([n]) => n > 0);
  const parts = ['he read ', b(plural(c.episodes, 'post', 'posts')), ' and did nothing about ',
    b(fmt(Math.round(c.episodes * (c.silence || 0)))), ' of them. '];
  if (acts.length) acts.forEach(([n, one, many], i) => parts.push(b(plural(n, one, many)), i < acts.length - 1 ? ', ' : '. '));
  else parts.push('nothing else. ');
  if (c.rewards || c.punishments) {
    parts.push(b(plural(c.rewards, 'reward', 'rewards')), ' and ', b(plural(c.punishments, 'punishment', 'punishments')), ' reached him.');
  } else {
    parts.push(tense === 'today' ? 'nothing has rewarded or punished him yet.' : 'nothing rewarded or punished him.');
  }
  sentence(node, parts);
}
// what an outcome was, for "what stuck"
export const OUTCOME_SOURCE = {
  like: 'a like', repost: 'a repost', follow: 'a follow',
  like_on_post: 'a like on a post of his', repost_on_post: 'a repost of a post of his',
  kind_reply: 'a kind reply',
  known_account_reply: 'a known account replying',
  known_account_inbound: 'a known account coming back',
  known_account_follow: 'a known account following him',
  known_account_like: 'a known account liking a post of his',
  known_account_repost: 'a known account reposting him',
  block: 'a block',
  vader_negative_reply: 'an unkind reply',
};

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
  let atEnd = false; // at the end of the document the last row is current, whatever the band says
  const io = new IntersectionObserver(
    (entries) => {
      if (atEnd) return;
      for (const e of entries) {
        e.target.classList.toggle('is-current', e.isIntersecting);
        if (e.isIntersecting && onCurrent) onCurrent(e.target);
      }
    },
    { rootMargin: '-10% 0px -70% 0px' },
  );
  rows.forEach((r) => io.observe(r));
  // the last row can never reach the band once the page has no more to scroll: at the end of
  // the document it is the current one; scrolling back up hands the rows to the observer again
  const last = rows[rows.length - 1];
  const onScroll = () => {
    const end = innerHeight + scrollY >= document.documentElement.scrollHeight - 2;
    if (end === atEnd) return;
    atEnd = end;
    if (end) {
      rows.forEach((r) => r.classList.toggle('is-current', r === last));
      if (last && onCurrent) onCurrent(last);
    } else {
      rows.forEach((r) => {
        io.unobserve(r);
        io.observe(r);
      });
    }
  };
  if (last) addEventListener('scroll', onScroll, { passive: true });
  return io;
}
// the top line's handle and the "in development" word, from status.json when a page has it
export function applyIdentity(st) {
  if (st?.identity?.handle && $('handle')) {
    $('handle').textContent = `@${st.identity.handle}`;
    $('handle').href = `https://bsky.app/profile/${st.identity.handle}`;
  }
}
