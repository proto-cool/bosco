// days.html: one report per local day of his life, from data/days/*.json (the ledger alone).
import {
  $, DATA, fmt, pct, el, b, sentence, plural, list, longDate, clockTime, getJSON, renderPosts, personRow,
  windowRow, daySentence, whyText, saidNote, CHOSE, OUTCOME_SOURCE, markNav, trackCurrent, applyIdentity,
} from './common.js';

let index = null;
let current = null;
let caps = null; // the rails' numbers, from status.json

const wanted = () => new URLSearchParams(location.search).get('d');

function hourLabel(h) {
  return h === 0 ? 'midnight' : h === 12 ? 'noon' : h < 12 ? `${h} am` : `${h - 12} pm`;
}

function renderLadder() {
  const ol = $('ladder');
  ol.replaceChildren(
    ...index.days.map((d) => {
      const li = el('li');
      const a = el('a');
      a.href = `?d=${d.date}`;
      a.dataset.date = d.date;
      a.append(
        el('span', 'd', d.date),
        el('span', 'n', d.day_of_life ? `day ${d.day_of_life}` : ''),
        el('span', 's', d.silence == null ? '' : pct(d.silence)),
      );
      a.addEventListener('click', (e) => {
        e.preventDefault();
        history.pushState(null, '', `?d=${d.date}`);
        show(d.date);
      });
      li.append(a);
      return li;
    }),
  );
}

function markLadder(date) {
  for (const a of document.querySelectorAll('#ladder a')) a.setAttribute('aria-current', a.dataset.date === date ? 'page' : 'false');
}

function hourStrip(node, rep) {
  const max = Math.max(1, ...rep.hours.map((h) => h.episodes));
  node.replaceChildren(
    ...rep.hours.map((h, i) => {
      const col = el('span', 'hour');
      col.title = `${hourLabel(i)}: ${plural(h.episodes, 'post', 'posts')}, acted on ${h.acted}`;
      const all = el('i', 'all');
      all.style.height = `${(h.episodes / max) * 100}%`;
      const acted = el('i', 'acted');
      acted.style.height = `${(h.acted / max) * 100}%`;
      col.append(all, acted);
      if (h.landings) col.append(el('i', 'landing'));
      return col;
    }),
  );
}

async function show(date) {
  current = date;
  markLadder(date);
  let rep;
  try {
    rep = await getJSON(`${DATA}/days/${date}.json`);
  } catch {
    $('day-title').textContent = date;
    $('day-sentence').textContent = 'no report for that day.';
    return;
  }
  const c = rep.counts;
  const tz = rep.tz;
  document.title = `bosco · ${rep.date}`;
  $('day-title').textContent = longDate(rep.date);
  $('day-sub').textContent =
    `${rep.day_of_life ? `day ${rep.day_of_life} of his life · ` : ''}${rep.final ? 'the whole day' : 'so far'} · his time is ${tz.toLowerCase()}`;
  $('silence').textContent = c.silence == null ? '–' : pct(c.silence);
  daySentence($('day-sentence'), c, rep.final ? 'past' : 'today');

  // when
  hourStrip($('hours'), rep);
  const busiest = rep.hours.reduce((m, h, i) => (h.episodes > rep.hours[m].episodes ? i : m), 0);
  const app = rep.hours.map((h, i) => [i, h.appetite]).filter(([, v]) => v != null);
  const whenParts = [];
  if (c.episodes) whenParts.push('busiest around ', b(hourLabel(busiest)), '. ');
  if (rep.landings) whenParts.push('dust landed on him ', b(plural(rep.landings, 'time', 'times')), ' and he groomed ', b(plural(rep.grooms, 'time', 'times')), rep.grooms ? '; a groom is a post when his cap allows. ' : '. ');
  if (app.length > 1) {
    const lo = app.reduce((m, x) => (x[1] < m[1] ? x : m)), hi = app.reduce((m, x) => (x[1] > m[1] ? x : m));
    whenParts.push('his appetite for company was lowest around ', b(hourLabel(lo[0])), ' and highest around ', b(hourLabel(hi[0])), '.');
  }
  sentence($('when-sentence'), whenParts);

  // where
  const feeds = rep.feeds || [];
  if (feeds.length) {
    const parts = ['he read from ', b(list(feeds.slice(0, 5).map((f) => `${f.name} ×${f.reads}`))), '. '];
    const liked = feeds.filter((f) => f.approaches > 0).sort((x, y) => y.approaches - x.approaches);
    if (liked.length) parts.push('he approached something in ', b(list(liked.slice(0, 3).map((f) => f.name))), '.');
    else parts.push('nothing there drew him closer.');
    sentence($('where-sentence'), parts);
  } else {
    $('where-sentence').textContent = 'he did not go anywhere.';
  }

  // smelled
  const topics = Object.entries(rep.topics || {});
  const words = Object.entries(rep.words || {});
  const sm = [];
  if (topics.length) sm.push('topics: ', b(topics.slice(0, 8).map(([k, v]) => `${k} ×${v}`).join(', ')), '. ');
  if (words.length) sm.push('his own words in what he read: ', b(words.slice(0, 10).map(([k, v]) => `${k} ×${v}`).join(', ')), '. ');
  if (rep.words_unknown) sm.push('and ', b(plural(rep.words_unknown, 'smell', 'smells')), ' of words he has no word for, kept as hashes, never as words.');
  if (!sm.length) sm.push('nothing he has a word for.');
  sentence($('smell-sentence'), sm);

  // favorite and least favorite: the post by its smell, and the post itself only when he liked it in public
  const tasteOf = (p) => {
    const parts = [p.mentioned ? 'a mention' : p.feed ? `something in ${p.feed}` : 'something he browsed', ' around ', b(clockTime(p.ts, tz))];
    if (p.topics?.length) parts.push(', about ', b(list(p.topics)));
    if (p.words?.length) parts.push(', with ', b(list(p.words.slice(0, 4))), ' in it');
    if (p.vader != null && Math.abs(p.vader) >= 0.05) parts.push('; it tasted ', b(p.vader > 0 ? 'sweet' : 'bitter'));
    if (Math.abs(p.learned) >= 0.05) parts.push(p.vader != null && Math.abs(p.vader) >= 0.05 ? ' and ' : '; ', 'he had learned it was ', b(p.learned > 0 ? 'good' : 'bad'));
    return parts;
  };
  const didOf = (p) => {
    if (p.done) {
      return { like: 'he liked it', follow: 'he followed them', walk: 'he walked toward it and read more of them', reply: 'he answered', answer: 'he answered by reflex', leave: 'he unfollowed them', spontaneous_post: 'he posted' }[p.done] || `he did ${p.done.replace('_', ' ')}`;
    }
    if (p.action === 'nothing') return 'he did nothing';
    return `his neurons chose ${CHOSE[p.action] || p.action}, withheld: ${whyText(p.why, caps)}`;
  };
  const times = (x) => (x >= 10 ? Math.round(x) : x.toFixed(1));
  const fav = rep.favorite, least = rep.least;
  if (fav) {
    sentence($('favorite-sentence'), ['favorite: ', ...tasteOf(fav), '. ', didOf(fav), '; his like neurons ran at ', b(`${times(fav.ratio)}×`), ' their threshold.']);
    // the sentence above already says he liked it and when, so the card carries no note of its own
    if (fav.uri) await renderPosts($('favorite-post'), [{ uri: fav.uri, ts: fav.ts, kind: 'liked', label: '' }], 1);
    else $('favorite-post').replaceChildren();
  } else {
    $('favorite-sentence').textContent = 'nothing drew his tongue out that day.';
    $('favorite-post').replaceChildren();
  }
  if (least) sentence($('least-sentence'), ['least favorite: ', ...tasteOf(least), '. ', didOf(least), '; his leave neurons ran at ', b(`${times(least.ratio)}×`), ' their threshold.']);
  else $('least-sentence').textContent = 'nothing made him turn away that day.';

  // who
  const people = rep.people || [];
  const plist = $('people');
  plist.replaceChildren(
    ...people.slice(0, 10).map((p) => personRow(p, p.mentions ? `${p.mentions} to him` : `×${p.n}`)),
  );
  if (!people.length) plist.replaceChildren(el('li', 'empty', 'nobody.'));
  const came = people.filter((p) => p.mentions > 0);
  sentence(
    $('who-sentence'),
    came.length
      ? [b(plural(came.length, 'person', 'people')), ' spoke to him; he read ', b(plural(people.length, 'account', 'accounts')), ' in all.']
      : ['nobody spoke to him. he read ', b(plural(people.length, 'account', 'accounts')), '.'],
  );

  // said: the note carries his clock that day, since the card's own time is counted from now
  await renderPosts($('posts'), (rep.posts || []).map((p) => ({ ...p, label: `${saidNote(p.kind)} · ${clockTime(p.ts, tz)}` })), 12);

  // learned
  const oc = rep.outcomes || [];
  const bySource = {};
  for (const o of oc) bySource[`${o.valence}:${o.source}`] = (bySource[`${o.valence}:${o.source}`] || 0) + 1;
  const learned = [];
  if (oc.length) {
    learned.push(
      b(plural(c.rewards, 'reward', 'rewards')), ' and ', b(plural(c.punishments, 'punishment', 'punishments')), ': ',
      Object.entries(bySource).map(([k, v]) => `${OUTCOME_SOURCE[k.split(':')[1]] || k.split(':')[1].replaceAll('_', ' ')} ×${v}`).join(', '), '.',
    );
  } else {
    learned.push('no reward and no punishment reached him, so what he had learned only faded.');
  }
  sentence($('learned-sentence'), learned);

  // the record
  const rec = rep.record || [];
  $('record-list').replaceChildren(...rec.slice().reverse().slice(0, 80).map((w) => windowRow(w, clockTime(w.ts, tz), caps)));
  if (!rec.length) $('record-list').replaceChildren(el('li', 'empty', 'nothing crossed threshold and nobody spoke to him.'));
  $('record-caption').textContent =
    `every window that day where something crossed threshold, and every time someone spoke to him (${fmt(rec.length)}, newest first); ` +
    `the other ${fmt(rep.silent)} windows were silence.`;

  // fine print
  const ctl = rep.control || [];
  const ctlText = ctl.map((x) => {
    const at = clockTime(x.ts, tz);
    const span = /^(\d+)->(\d+)$/.exec(x.target || ''); // a span of his time, ms
    if (x.kind === 'downtime') {
      // the process was down; on restart the gap was skipped, not lived
      return span ? `off for ${Math.round((span[2] - span[1]) / 60000)} min at ${at}, skipped not lived` : `off for a while at ${at}, skipped not lived`;
    }
    if (x.kind === 'slow') {
      // he was awake for this, just running behind the wall clock; the time is his and he keeps it
      const m = /lag=(\d+)/.exec(x.target || '');
      return m ? `${Math.round(m[1] / 60)} min behind the clock at ${at}` : `behind the clock at ${at}`;
    }
    // the two boundaries: a replay before one runs under the old rule or numerics, after it under the new
    if (x.kind === 'numerics') return `cpu numerics set to ${(x.target || '').split('->').pop()} at ${at}`;
    if (x.kind === 'plasticity') return `learning rule ${(x.target || '').replace('->', ' → ')} at ${at}`;
    if (x.kind === 'kernel') return `kernel ${(x.target || '').replace('->', ' → ')} at ${at}`;
    if (x.kind === 'clock') return `his clock moved from ${(x.target || '').replace('->', ' to ')} at ${at}`;
    return {
      sleep: `put to sleep by the operator at ${at}`,
      wake: `woken by the operator at ${at}`,
      forget: `made to forget an account by the operator at ${at}`,
      ignored: `an account he was told to ignore spoke; not perceived, at ${at}`,
      deleted_in_app: `a post of his deleted from the app at ${at}`,
      unliked_in_app: `a like of his undone from the app at ${at}`,
      unfollowed_in_app: `a follow of his undone from the app at ${at}`,
    }[x.kind] || `${x.kind.replaceAll('_', ' ')} at ${at}`;
  });
  $('fine').textContent = [
    rep.digest ? `brain digest at close ${rep.digest.brain.slice(0, 12)} · weights ${rep.digest.weights.slice(0, 12)}` : '',
    ctlText.length ? ctlText.join(' · ') : 'no operator action and no downtime',
    `written ${new Date(rep.generated * 1000).toISOString().slice(0, 16).replace('T', ' ')} utc`,
  ].filter(Boolean).join(' · ');
}

async function main() {
  markNav();
  try {
    const st = await getJSON(`${DATA}/status.json`);
    applyIdentity(st);
    caps = st.caps || null;
  } catch {
    // the days stand on their own; a cap reads "a cap" without status.json
  }
  try {
    index = await getJSON(`${DATA}/days/index.json`);
  } catch {
    $('day-sentence').textContent = 'no days yet.';
    return;
  }
  if (!index.days.length) {
    $('day-sentence').textContent = 'no days yet.';
    return;
  }
  renderLadder();
  const d = wanted();
  show(index.days.some((x) => x.date === d) ? d : index.days[0].date);
  trackCurrent([...document.querySelectorAll('.sheet .row')]);
}
window.addEventListener('popstate', () => {
  const d = wanted();
  if (d && d !== current) show(d);
});
main();
