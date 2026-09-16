// days.html: one report per local day of his life, from data/days/*.json (the ledger alone).
import {
  $, DATA, fmt, pct, el, b, sentence, plural, list, longDate, clockTime, getJSON, nameInto, renderPosts, personRow,
  markNav, trackCurrent, applyIdentity,
} from './common.js';

let index = null;
let current = null;

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
  const c = rep.counts, a = c.actions || {};
  const tz = rep.tz;
  document.title = `bosco · ${rep.date}`;
  $('day-title').textContent = longDate(rep.date);
  $('day-sub').textContent =
    `${rep.day_of_life ? `day ${rep.day_of_life} of his life · ` : ''}${rep.final ? 'the whole day' : 'so far'} · his time is ${tz.toLowerCase()}`;
  $('silence').textContent = c.silence == null ? '–' : pct(c.silence);
  const replies = (a.reply || 0) + (a.answer || 0) + (a.identity || 0);
  if (c.episodes) {
    const acts = [
      [replies, 'reply', 'replies'],
      [a.like || 0, 'like', 'likes'],
      [a.follow || 0, 'follow', 'follows'],
      [a.spontaneous_post || 0, 'post of his own', 'posts of his own'],
    ].filter(([n]) => n > 0);
    const parts = ['he read ', b(plural(c.episodes, 'post', 'posts')), ' and said nothing to ',
      b(fmt(Math.round(c.episodes * (c.silence || 0)))), ' of them. '];
    if (acts.length) acts.forEach(([n, one, many], i) => parts.push(b(plural(n, one, many)), i < acts.length - 1 ? ', ' : '. '));
    else parts.push('he did nothing else. ');
    parts.push(
      c.rewards || c.punishments
        ? [b(plural(c.rewards, 'reward', 'rewards')), ' and ', b(plural(c.punishments, 'punishment', 'punishments')), '.']
        : ['nothing rewarded or punished him.'],
    );
    sentence($('day-sentence'), parts.flat());
  } else {
    $('day-sentence').textContent = 'he read nothing that day.';
  }

  // when
  hourStrip($('hours'), rep);
  const busiest = rep.hours.reduce((m, h, i) => (h.episodes > rep.hours[m].episodes ? i : m), 0);
  const app = rep.hours.map((h, i) => [i, h.appetite]).filter(([, v]) => v != null);
  const whenParts = [];
  if (c.episodes) whenParts.push('busiest around ', b(hourLabel(busiest)), '. ');
  if (rep.landings) whenParts.push('dust landed on him ', b(plural(rep.landings, 'time', 'times')), ' and he groomed ', b(plural(rep.grooms, 'time', 'times')), '. ');
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
  if (words.length) sm.push('his own words in what he read: ', b(words.slice(0, 10).map(([k, v]) => `${k} ×${v}`).join(', ')), '.');
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
    const did = { like: 'he liked it', follow: 'he followed', reply: 'he answered', walk: 'he walked toward it', leave: 'he left', nothing: 'he did nothing' }[p.action] || `he chose ${p.action}`;
    return p.action !== 'nothing' && !p.acted ? `${did}, withheld at the door` : did;
  };
  const fav = rep.favorite, least = rep.least;
  if (fav) {
    sentence($('favorite-sentence'), ['favorite: ', ...tasteOf(fav), '. ', didOf(fav), ` (${fav.ratio.toFixed(2)}× threshold).`]);
    if (fav.uri) await renderPosts($('favorite-post'), [{ uri: fav.uri, ts: fav.ts, kind: 'liked' }], 1);
    else $('favorite-post').replaceChildren();
  } else {
    $('favorite-sentence').textContent = 'nothing drew his tongue out that day.';
    $('favorite-post').replaceChildren();
  }
  if (least) sentence($('least-sentence'), ['least favorite: ', ...tasteOf(least), '. ', didOf(least), ` (${least.ratio.toFixed(2)}× threshold).`]);
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

  // said
  await renderPosts($('posts'), rep.posts || [], 12);

  // learned
  const oc = rep.outcomes || [];
  const bySource = {};
  for (const o of oc) bySource[`${o.valence}:${o.source}`] = (bySource[`${o.valence}:${o.source}`] || 0) + 1;
  const sourceName = {
    known_account_inbound: 'a known account coming back',
    like: 'a like', repost: 'a repost', follow: 'a follow', kind_reply: 'a kind reply',
    block: 'a block', vader_negative_reply: 'an unkind reply',
  };
  const learned = [];
  if (oc.length) {
    learned.push(
      b(plural(c.rewards, 'reward', 'rewards')), ' and ', b(plural(c.punishments, 'punishment', 'punishments')), ': ',
      Object.entries(bySource).map(([k, v]) => `${sourceName[k.split(':')[1]] || k.split(':')[1].replaceAll('_', ' ')} ×${v}`).join(', '), '. ',
    );
    learned.push('each one re-presented the post it answered and fired the dopamine neurons of that compartment.');
  } else {
    learned.push('no reward and no punishment reached him, so his weights only faded.');
  }
  sentence($('learned-sentence'), learned);

  // the record
  const rec = rep.record || [];
  $('record-list').replaceChildren(
    ...rec.slice().reverse().slice(0, 80).map((w) => {
      const li = el('li', `win${w.acted ? ' acted' : ''}`);
      const who = el('span', 'who');
      if (w.did) nameInto(who, w.did);
      else who.textContent = w.kind === 'spontaneous' ? 'a landing' : '–';
      li.append(
        el('span', null, clockTime(w.ts, tz)),
        el('span', 'smell', w.kind === 'spontaneous' ? 'dust' : w.mentioned ? 'mention' : w.feed || 'browse'),
        who,
        el('span', 'what', `${w.action.replace('_', ' ')}${w.acted ? '' : ' · withheld'}`),
      );
      return li;
    }),
  );
  if (!rec.length) $('record-list').replaceChildren(el('li', 'empty', 'he acted on nothing.'));
  $('record-caption').textContent =
    `every window that day where something crossed threshold (${fmt(rec.length)}); the other ${fmt(rep.silent)} were silence.`;

  // fine print
  const ctl = rep.control || [];
  const ctlText = ctl.map((x) => {
    if (x.kind === 'downtime') return `downtime skipped at ${clockTime(x.ts, tz)}`;
    if (x.kind === 'slow') {
      // he was awake for this, just running behind the wall clock; the time is his and he keeps it
      const m = /lag=(\d+)/.exec(x.target || '');
      const at = clockTime(x.ts, tz);
      return m ? `${Math.round(m[1] / 60)} min behind the clock at ${at}` : `behind the clock at ${at}`;
    }
    if (x.kind === 'numerics') {
      // the CPU code path numpy runs on changed here (a new machine): spans before this replay on
      // the old level, spans after on the new; a boundary, not a break
      return `numerics ${(x.target || '').replace('->', ' → ')} from ${clockTime(x.ts, tz)}`;
    }
    if (x.kind === 'plasticity') {
      // the learning rule changed here (README, Agent.PLASTICITY_VERSION): spans before this
      // replay under the old rule, spans after under the new; a boundary, not a break
      return `learning rule ${(x.target || '').replace('->', ' → ')} from ${clockTime(x.ts, tz)}`;
    }
    return `${x.kind.replaceAll('_', ' ')} at ${clockTime(x.ts, tz)}`;
  });
  $('fine').textContent = [
    rep.digest ? `memory at close ${rep.digest.brain.slice(0, 12)} · weights ${rep.digest.weights.slice(0, 12)}` : '',
    ctlText.length ? ctlText.join(' · ') : 'no operator action and no downtime',
    `written ${new Date(rep.generated * 1000).toISOString().slice(0, 16).replace('T', ' ')} utc`,
  ].filter(Boolean).join(' · ');
}

async function main() {
  markNav();
  getJSON(`${DATA}/status.json`).then(applyIdentity).catch(() => {});
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
