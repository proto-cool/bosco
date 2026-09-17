import {
  $, DATA, fmt, pct, ago, alive, el, b, sentence, list, getJSON, getBytes, renderPosts, personRow,
  windowRow, daySentence, setLive, markNav, trackCurrent, applyIdentity,
} from './common.js';
import { BrainView, parseActivity, parseAtlas } from './brain.js';

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
  return { mb: rgbOf('--accent-primary'), dn: rgbOf('--accent-secondary'), rest: rgbOf('--text-secondary') };
}

// ---- brain ------------------------------------------------------------------------------------
let brain = null;
let popNames = [];

async function initBrain() {
  const [meta, buf] = await Promise.all([getJSON('./atlas.json'), getBytes('./atlas.bin', 20000)]);
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
  if (pending) {
    brain.push(pending);
    pending = null;
  }
}

// The five groups named by the act they make, not by the group: a reader cannot know that a groom
// is a post of his own.  Which neurons each one is stands in the caption underneath.
const ACT_OF = {
  engage: 'go to them',
  reply: 'answer',
  like: 'like',
  leave: 'turn away',
  groom: 'post',
};
// How close a group is to the point where he acts.  Hz is the ledger's business, not the page's;
// it stays on the row's title for anyone who wants it.
function nearness(ratio) {
  if (ratio >= 1) return 'over';
  if (ratio >= 0.75) return 'nearly';
  if (ratio >= 0.35) return 'stirring';
  return 'quiet';
}

function buildReadout() {
  const box = $('readout');
  box.replaceChildren();
  for (const p of popNames) {
    const row = el('div', 'ro');
    row.dataset.pop = p;
    row.append(el('span', 'k', ACT_OF[p] || p));
    const track = el('div', 'track');
    track.append(el('i', `fill${p === 'leave' ? ' avoid' : ''}`), el('i', 'tick'));
    if (p === 'engage') track.append(el('i', 'tick walk')); // the lower rung: a walk, not a follow
    row.append(track, el('span', 'v', '–'));
    box.append(row);
  }
}

// ---- the live second ----------------------------------------------------------------------------
// Polled every second.  One fetch in flight at a time, a short timeout, and a few misses before
// the page says he is unreachable; the file's own wall stamp says whether he is between seconds
// (a poll, a fetch, composing) or gone.
let thresholds = {};
let caps = null;
let lastT = -1;
let pending = null; // a second that arrived before the atlas did
let inFlight = false;
let misses = 0;
let lastAct = null;

function judge() {
  if (misses >= 3) return setLive('unreachable', 0);
  if (!lastAct) return;
  const age = Date.now() / 1000 - lastAct.wallTs;
  if (age < 20) setLive('live', age);
  else if (age < 240) setLive('busy', age);
  else setLive('stale', age);
}

async function pollActivity() {
  if (inFlight || document.hidden) return;
  inFlight = true;
  try {
    const act = parseActivity(await getBytes(`${DATA}/activity.bin`));
    misses = 0;
    lastAct = act;
    if (act.tMs !== lastT) {
      lastT = act.tMs;
      if (brain) brain.push(act);
      else pending = act;
      $('firing').textContent = fmt(act.idx.length);
      $('kc-active').textContent = fmt(act.kcActive);
      popNames.forEach((p, k) => {
        const row = document.querySelector(`.ro[data-pop="${p}"]`);
        if (!row) return;
        const rate = act.pops[k], th = thresholds[p];
        const ratio = th ? rate / th : 0;
        row.querySelector('.fill').style.transform = `scaleX(${Math.min(1, ratio / 1.6)})`;
        row.querySelector('.v').textContent = th ? nearness(ratio) : '–';
        row.title = `${p}: ${rate.toFixed(1)} Hz${th ? ` against ${th.toFixed(1)} Hz, where he acts` : ''}`;
        row.classList.toggle('over', th != null && rate > th);
      });
    }
  } catch {
    misses += 1;
  } finally {
    inFlight = false;
    judge();
  }
}

// ---- status ------------------------------------------------------------------------------------
let statusInFlight = false;
async function pollStatus() {
  if (statusInFlight || document.hidden) return;
  statusInFlight = true;
  let st;
  try {
    st = await getJSON(`${DATA}/status.json`);
  } catch {
    statusInFlight = false;
    return;
  }
  try {
    renderStatus(st);
  } finally {
    statusInFlight = false;
  }
}

async function renderStatus(st) {
  thresholds = st.readout?.thresholds || {};
  caps = st.caps || null;
  applyIdentity(st);
  const walkTick = document.querySelector('.ro[data-pop="engage"] .tick.walk');
  if (walkTick && thresholds.walk && thresholds.engage) {
    walkTick.style.left = `${(62.5 * thresholds.walk) / thresholds.engage}%`;
  }
  const br = st.brain;
  $('bio-time').textContent =
    `${alive(br.t_ms)} of life · ${br.lag_s > 120 ? `running ${Math.round(br.lag_s / 60)} min behind the clock` : 'keeping up with the clock'}`;

  // today: one figure, one sentence
  const t = st.today;
  $('silence').textContent = t.silence == null ? '–' : pct(t.silence);
  daySentence($('today-sentence'), t, 'today');
  const topics = Object.entries(st.topics_today || {}).sort((x, y) => y[1] - x[1]).slice(0, 6);
  const fd = (st.feeds && st.feeds.feeds) || [];
  const haunts = fd.filter((f) => f.reads > 0).sort((x, y) => y.share - x.share);
  const parts = [];
  if (topics.length) parts.push('smelled most: ', b(topics.map(([k, v]) => `${k} ×${v}`).join(', ')), '. ');
  if (haunts.length) {
    parts.push('he has been reading ', b(list(haunts.slice(0, 4).map((f) => f.name))), haunts.length > 4 ? ' and elsewhere' : '');
    parts.push('; he goes back most to ', b(haunts[0].name), '.');
  }
  sentence($('topics-sentence'), parts);

  // memory: one figure, one sentence, a short list
  const people = st.people || [];
  $('people-count').textContent = fmt(people.length);
  const hunger =
    br.appetite == null ? '' : br.appetite > 0.7 ? ' he is hungry for company.' : br.appetite < 0.3 ? ' he has had his fill of company for now.' : ' he could take some company.';
  const dust = br.dust
    ? [' he is carrying ', b(pct(br.dust)), ' of a load of dust; when enough lands he cleans himself off, and that is a post.']
    : [' no dust on him just now; when enough lands he cleans himself off, and that is a post.'];
  sentence(
    $('memory-sentence'),
    br.stm_depressed || br.ltm_depressed
      ? [b(fmt(br.stm_depressed)), ' synapses hold a short-term memory and ', b(fmt(br.ltm_depressed)),
        ' a long-term one, of ', b(fmt(br.plastic_synapses)), ' that can learn.', ...dust, hunger]
      : ['nothing learned yet, across ', b(fmt(br.plastic_synapses)), ' synapses that can.', ...dust, hunger],
  );
  const wd = st.words || {};
  const sweet = (wd.sweet || []).map(([w]) => w).slice(0, 6);
  const bitter = (wd.bitter || []).map(([w]) => w).slice(0, 6);
  const air = (wd.air || []).slice(0, 8);
  $('words-sentence').replaceChildren();
  if (sweet.length || bitter.length || air.length) {
    const ws = [];
    if (sweet.length) ws.push('words that have been sweet: ', b(sweet.join(', ')), '. ');
    if (bitter.length) ws.push('words that have been bitter: ', b(bitter.join(', ')), '. ');
    if (air.length) ws.push('on his antennae right now: ', b(air.join(', ')), '.');
    sentence($('words-sentence'), ws);
  }
  const plist = $('people');
  plist.replaceChildren(...people.slice(0, 6).map((p) => personRow(p)));
  if (!people.length) plist.replaceChildren(el('li', 'empty', 'nobody yet. he has not met anyone he remembers.'));

  // his posts
  await renderPosts($('posts'), st.posts || [], 4);

  // the record
  $('recent').replaceChildren(...(st.recent || []).slice(0, 12).map((w) => windowRow(w, ago(w.ts), caps)));

  // for anyone checking a replay against the published ledger
  $('fine').textContent =
    `fingerprints, so a replay can be checked against this: his brain ${br.digest.slice(0, 12)}, ` +
    `his corpus ${st.corpus_digest.slice(0, 12)}. ${fmt(br.neurons)} neurons, ${fmt(br.synapses)} synapses. ` +
    `this page was written ${ago(st.generated)} ago.`;
}

// ---- go ----------------------------------------------------------------------------------------
markNav();
trackCurrent([...document.querySelectorAll('.rail .row')]);
initBrain().catch((e) => {
  console.error(e);
  $('atlas-caption').textContent = 'the atlas did not load. reload the page.';
});
pollStatus();
pollActivity();
setInterval(pollActivity, 1000);
setInterval(pollStatus, 20000);
setInterval(judge, 5000); // the age of the last second keeps counting even when nothing new arrives
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    pollActivity();
    pollStatus();
  }
});
