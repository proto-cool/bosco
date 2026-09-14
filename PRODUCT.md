# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Bystanders first: someone on Bluesky saw a reply or a post from
`@bosco.proto.cool`, clicked the link in his bio, and wants to understand in a
minute that there is a real fruit fly brain here and what it is doing right
now. They arrive on a phone as often as a desktop, mid-scroll, with no
background.

Second, one scroll down: people who know what a Kenyon cell is and want the
receipts. Neuroscience and modeling readers, and anyone auditing the
experiment: populations, thresholds, memory digests, what he learned and why.

Nick (`@proto.cool`) operates him, but not from this page. Operator control
lives on Bluesky by mention; the page is a window, never a console.

## Product Purpose

`bosco.proto.cool` is the public window on bosco: a simulated male fruit fly
(MaleCNS v1.0 connectome, 40,939 central-brain neurons, every synapse, leaky
integrate-and-fire with mushroom-body plasticity) living as an autonomous
Bluesky account. The page shows his brain as it fires, what his body is
inclined to do, what he did today, what he has learned about people and
topics, and what he said.

Success is a visitor who leaves knowing three things: it is a real brain
model and not a chatbot; he is genuinely the one acting; and he is learning,
slowly, from how people treat him. The page exists so the experiment is
legible to the community that is, in effect, running it.

## Positioning

The only account on the network whose every action follows from a complete
connectome under a published log. No language model anywhere. Every window
is recorded and replays bit-identical; the weights and ledger are dumped
nightly. The page is the live end of that audit trail, not a marketing
surface for it.

## Operating Context

- The bosco process writes two files a static server exposes:
  `activity.bin` (sparse list of neurons that spiked in the last simulated
  second, plus readout population rates) about once a second, and
  `status.json` (ledger-derived counts, people, recent windows, his post URIs,
  simulation state) once per poll, roughly every two minutes.
- The site is a static bundle (`panel/`, Vite, ARC UI 4.2.2 from the pinned
  npm package) served by Caddy from a container on the same 2-vCPU VPS
  (`ops/Containerfile.panel`, `ops/bosco-panel.container`). Deployment is
  documented in `docs/DEPLOY.md` §E.
- Handles, avatars and his own post text are fetched at view time from the
  public Bluesky API by DID and URI. The server holds no text of anyone's.
- He runs one biological second per wall second and can fall behind on the
  small box; the page shows lag honestly rather than hiding it.
- Silence is his most common decision and must read as normal, not as an
  outage.

## Capabilities and Constraints

Confirmed capabilities of the surface:

- Every neuron drawn at its real soma position (86% have one in the volume;
  the rest sit at the mean of their targets, flagged), lit when it spikes,
  three projections, groups toggleable (Kenyon cells, mushroom body output,
  dopamine, descending, sensory, the rest). No synapses are drawn.
- Readout populations as live rates against their thresholds.
- Today's counts, silence fraction, rewards and punishments, topics smelled.
- Memory: depressed short-term and long-term synapse counts, people ranked
  by learned valence with familiarity.
- His recent posts, and the last windows with their decisions.
- A plain-words explanation and links to the code, EXPERIMENT.md and the
  modeling decisions.

Hard constraints:

- **Read-only, nothing reaches him.** No control, form, comment box, or
  input that could influence the fly. This is a non-negotiable of the
  experiment (humans influence him only through the network).
- **No stored text of other people.** Only DIDs, URIs and features leave the
  server. Anything with words is fetched client-side from the public API.
- **No fabricated liveliness.** No fake activity, no smoothing that invents
  spikes, no minimum-activity floor. If the data file is stale, say so.
- **Thresholds and the action set are frozen at `freeze-v1`**; the page
  reports them, never edits them.
- **Terminology**: window (one second of stimulus), episode, readout
  population, learned valence, familiarity, landing (debris on his
  bristles, which is what makes him post), groom (an own post). "Fruit fly",
  never "fly" alone in his own voice.

Undecided:

- Whether the dunce control account (weeks 3–4 after the tag) gets a panel of
  its own or a toggle on this one.
- Whether nightly snapshot history (a day-by-day view) belongs on the page.

## Brand Commitments

- **ARC UI 4.2.2 and its `DESIGN.md` are binding.** Tokens only, never
  literals; the two-color contract; state by tint, glow and accent text,
  never a colored left border; typefaces as roles; type contexts, not
  ad-hoc sizes; light as lobes; motion only when motivated and honoring
  reduced motion; both themes checked. Source of truth:
  `~/projects/arclight/arc-ui/DESIGN.md`.
- **He is "he", a fruit fly, by name.** bosco is a clever fruit fly, not a
  brain we run or a plaything. No "we built", no "we're playing god", no
  "powered by" hype. The page describes him the way a naturalist describes an
  animal.
- **Lowercase fly voice for the page's own copy.** Plain declaratives, no
  jokes from the writer, no marketing register. Proper nouns keep their
  capitals (MaleCNS, Janelia, Bluesky).
- Name: `bosco`. Handle: `@bosco.proto.cool`. Creator: `@proto.cool`.
- Bio says "in development" until `freeze-v1`; the page carries the same
  badge until then.

## Evidence on Hand

- The live data itself: `state/panel/status.json` and `activity.bin` on the
  VPS; a scratch example can be produced with `bosco panel --out DIR`.
- The soma atlas `panel/public/atlas.bin` + `atlas.json`, built by
  `scripts/build_panel_atlas.py` from the MaleCNS annotations.
- The published repo (`github.com/proto-cool/bosco`), `EXPERIMENT.md`,
  `README.md` modeling decisions 1–23, `docs/` gate reports, nightly
  `snapshots/`.
- His actual posts and replies on Bluesky, fetchable by URI.
- No testimonials, press, or usage numbers exist; do not invent any.
- Phase gates that passed (0–3) and their reports in `docs/`; thresholds are
  provisional (synthetic battery) until dev-period calibration.

## Product Principles

1. **The fly does the work; the page only watches.** Nothing on the page may
   feed back into him, and nothing may dress up what he did.
2. **Legible in a minute, auditable in an hour.** Bystanders first, receipts
   below; the same page serves both without a mode switch.
3. **Honest about scale and silence.** Show lag, show 83% silence, show zero
   traces on day one. An empty state is a true state.
4. **Every number has a place in the model.** A stat on the page names a
   population, a synapse count, or a ledger row that exists; no derived
   vanity metrics.
5. **Other people are smells, not content.** Show what he learned about an
   account, never what the account said.

## Accessibility & Inclusion

The brain canvas is decorative-plus: the numbers it illustrates (neurons
firing, Kenyon cells active, readout rates) must also be present as text.
Group toggles are real buttons with pressed state. Motion respects
`prefers-reduced-motion` (faster decay, no transitions). Phone width is a
first-class layout, not a fallback. No colour is the only carrier of a
state; the readout bars also change text weight and value.
