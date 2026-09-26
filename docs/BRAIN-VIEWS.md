# Brain views for Ask Bosco (handoff from the brain session, 2026-09-25)

Three views of the same decision, drawn from one data source. Reference renders are in
`runs/brain-map/` (`anatomy-frames-mix0.3.png`, `anatomy-mix0.3.gif`, `frames-4.png`, `wave.gif`).
The reference code is `scripts/brain_map_anatomy.py`, `scripts/brain_map.py` and
`scripts/brain_render.py` (the shared look).

| view | job | default? |
|---|---|---|
| **Anatomical flat** (64 × 48 dots) | the "wow": a real brain lighting up | **yes**, the brain panel |
| **Wave** (64 × 48 dots, time left to right) | the explanation: what happened, in order, ending in the answer cells | an "explain this decision" toggle |
| **3D connectome** | the deep dive: every neuron at its real position | "advanced", full screen |

## The grid rules (both flat views)
- **Exactly 64 × 48 = 3,072 dots, and every dot carries real neurons,** about 16 each. Dots never move;
  only size, brightness and colour change.
- **Anatomical flat:** each neuron sits at its real soma position (MaleCNS `somaLocation`, front view:
  x left-right, y dorsal-ventral; missing positions take the cell type's median). The positions are
  spread so every dot is used, with the shape kept at `mix = 0.3` (0 = raw positions, 1 = fully
  evened out). Label it "front view · real positions".
- **Wave:**
  - **Rows:** the left hemisphere is the top 23 rows, the brain's unpaired midline neurons the middle
    2, and the right hemisphere is mirrored in the bottom 23. Each hemisphere has fixed bands: smell 3
    (receptors, projection neurons, local neurons), taste 1, vision 3, learning 6 (KC γ, KC α/β,
    KC α′/β′, approach MBONs, avoid and other MBONs, DANs), innate 2, navigation 2, output 3, other 3.
  - **Columns:** within each row, neurons are spread in the order they respond (measured on real
    sniffs), so each row reads early to late, left to right. In the output rows, columns 60–63 are
    the **named answer cells**: avoid DNs in 60–61 (including MDN, "the moonwalker"), approach DNs
    in 62–63 (the gnathal DNs).

## The look (both flat views)
- **The palette is calcium imaging, not a rainbow.**
  - Firing above rest glows **green (130, 255, 150), turning white-hot** at the top of its range.
  - Pushed below rest is a **subtler magenta dip (200, 80, 170)**, at about 55% of the excitation
    strength.
  - **Resting dots** carry a faint tint: per region in the anatomical view, per band in the wave.
- **Size and brightness vary for texture.**
  - At rest, dot radius and brightness grow with the log of the neurons under the dot, so denser
    regions look denser.
  - Active dots swell (radius + about 2 px at full activity) and get a soft blurred halo.
  - The answer cells render about 1.5 px larger.
- **Afterglow:** a dot's displayed level decays by ×0.9 per 5 ms step, so the wave shows motion,
  not just snapshots.
- **Brightness scale:**
  - anatomical: one scale for the whole brain (the 97th percentile of the dots' peaks);
  - wave: one scale **per band** (the 90th percentile within the band), so every system's activity
    shows.

## Recommended polish (the web app's call)
- **Wave:** a few pixels of extra spacing between bands (visual only; every dot is still used),
  band labels in the left margin, a thin time axis (0–400 ms) along the bottom, and "answer" over
  columns 60–63.
- **Both:** a soft vignette for depth. Autoplay the replay: motion is where it comes alive.
- **Narration:** name real cells as they fire ("antennal lobe", "5% of Kenyon cells recognised it",
  "MDN fired: he backed away"). The trace carries the cell-type names.
- **One time cursor across all views:** scrubbing one moves the others.

## Data from the service (so the app hard-codes nothing)
- **Per brain version (family):**
  - `brain-map/anatomy.json` and `brain-map/wave.json`: for each dot `{x, y, neurons: [indices], band
    or region}`, plus the named answer cells;
  - `positions.bin`: soma xyz for the 3D view (about 0.6 MB);
  - the neuron-order hash, the steps (v1: 80 × 5 ms = 400 ms), `dt_ms`, and the region and band
    names.
- **Per answer:** a slimmed trace. Per step, the **signed activity per dot** for each view (3,072
  numbers per step per view, from rest), plus the named cells' activity and the top neurons per step
  with their cell-type names. The app draws from this; it never needs all 50,000 neurons except in
  the 3D view, which gets per-step activity for the top neurons only.
- Rest is the brain's own state with nothing smelled (every glomerulus at its resting rate).
  Activity is shown relative to rest.

## Honesty in the labels
- "Front view · real positions (MaleCNS)" for the anatomical view; "each row reads early to late" for
  the wave.
- He is a computer model built from a real fly's map. Green means firing above his resting rate; it
  is not measured calcium.

---

## Reference renders (real data: the topic specialist deciding one item)

GitHub copies (branch `v2`), for a machine without the repo:

| view | still (40 / 100 / 200 / 400 ms) | animation |
|---|---|---|
| anatomical flat | ![anatomical](https://raw.githubusercontent.com/proto-cool/bosco/v2/runs/brain-map/anatomy-frames-mix0.3.png) | [anatomy-mix0.3.gif](https://raw.githubusercontent.com/proto-cool/bosco/v2/runs/brain-map/anatomy-mix0.3.gif) |
| wave | ![wave](https://raw.githubusercontent.com/proto-cool/bosco/v2/runs/brain-map/frames-4.png) | [wave.gif](https://raw.githubusercontent.com/proto-cool/bosco/v2/runs/brain-map/wave.gif) |

What to look for:
- **Anatomical:** the two hemispheres mirror each other. Activity starts low in the centre (the antennal
  lobes, where smell arrives) and spreads outward and up (the mushroom bodies and the lateral horn).
- **Wave:**
  - **40 ms:** a few smell dots on the left.
  - **100 ms:** the learning and innate rows light up.
  - **200 ms:** most of the middle is alive.
  - **400 ms:** the enlarged answer cells at the right edge flare as he commits.

## Map files (real, current brain: family v1, 50,140 neurons)

In the repo:
- `runs/brain-map/anatomy-map-mix0.3.json` is the anatomical view.
- `runs/brain-map/map.json` is the wave.

The shapes the service will publish:

```jsonc
// wave.json (today: runs/brain-map/map.json)
{
  "width": 64, "height": 48,
  "bands": [["smell", 3, [86,140,255]], ["taste", 1, [255,170,60]], ["vision", 3, [70,200,220]],
            ["learning", 6, [180,120,255]], ["innate", 2, [255,110,150]], ["navigation", 2, [120,220,120]],
            ["output", 3, [255,120,40]], ["other", 3, [150,150,160]]],   // name, rows per hemisphere, band colour
  "dots": [ { "x": 0, "y": 0, "band": "smell", "neurons": [1204, 1377, ...] }, ... ],  // exactly 3,072
  "named": { "avoid": ["DNa02","DNa03","DNa13","DNde002","DNde005","DNge127","DNp42","DNp52",
                       "DNpe021","DNpe023","DNpe026","DNpe028","MDN"],
             "approach": ["CB0429","DNg104","DNg13","DNg34","DNge138","DNge149","DNge150","DNge151","DNge152","DNp68"] }
}
// anatomy.json (today: runs/brain-map/anatomy-map-mix0.3.json)
{ "width": 64, "height": 48, "mix": 0.3,
  "dots": [ { "x": 0, "y": 0, "neurons": [...] }, ... ] }   // exactly 3,072; region tints: see below
```

- `neurons` are indices in the family's neuron order (0 … 50,139). A dot can share neurons with a
  neighbour in the anatomical view, where a sparse edge borrows its nearest neurons so no dot is
  empty.
- The band colours in `wave.json` are used only as **faint resting tints**, at about 28% of the
  listed colour. Active dots use the imaging palette.
- The band colours in the rendered wave are, for now, only the faint tints; everything bright is the
  imaging palette.

## Per-answer trace (what the service will send)

```jsonc
{
  "version": "topic-2026-10-xx", "family": "v1-<hash>",
  "steps": 80, "dt_ms": 5,
  "views": {
    "anatomy": { "activity": [[/* 3,072 signed floats, dot order as in anatomy.json */], ... ×80] },
    "wave":    { "activity": [[/* 3,072 */], ... ×80] }
  },
  "named": { "avoid": [/* per step, mean signed activity */], "approach": [/* per step */] },
  "top": [ [ { "neuron": 38121, "type": "MDN", "region": "dn", "rate": 0.41 }, ... up to 32 ], ... ×80 ],
  "answer": { "lean": -0.62, "read": "dn" }
}
```

- Activity is **signed, relative to rest** (his own state with nothing smelled, every glomerulus at
  its resting rate). Positive = firing above rest; negative = pushed below.
- Rounded to 3 significant digits and gzipped, two views × 80 steps × 3,072 dots is about 0.5–1 MB
  per sniff. If that's too heavy: send every 2nd step and interpolate, or quantise to int8 per view
  (scale sent alongside).

## Render algorithm (exactly what the reference does; `scripts/brain_render.py`)

```
inputs per view: dots[3072] with neuron count n_d and resting tint T_d (RGB); activity A[t][d] (signed)
dens_d  = (log1p(n_d) - min) / (max - min)                         // 0..1
peak_d  = max_t |A[t][d]|
scale_d = anatomical: percentile97(peak_*) for all d
          wave:       percentile90(peak_d over dots in d's band)   // per band
glow_d = 0, sign_d = 0
for each step t:
    v = clamp(|A[t][d]| / scale_d, 0, 1)
    if v >= glow_d:  glow_d = v; sign_d = sign(A[t][d])
    else:            glow_d = glow_d * 0.9                            // afterglow
    g = glow_d ^ 1.2;  if sign_d < 0: g = g * 0.55                    // inhibition is subtler
    rest   = T_d * (0.55 + 0.5 * dens_d)
    hot    = sign_d >= 0 ? (130,255,150) : (200,80,170)
    if sign_d >= 0 and g > 0.7: hot = lerp(hot, (255,255,255), (g - 0.7) / 0.3)   // white-hot
    colour = lerp(rest, hot, g)
    radius = 1.8 + 1.4 * dens_d + 2.2 * g (+1.5 for answer cells)     // at 14 px per grid cell
    draw circle(centre of cell, radius, colour)
    if g > 0.35 and sign_d >= 0: add a halo (radius + 4, colour hot * 0.35, blurred ~4 px)
background (6, 7, 10)
```

Anatomical resting tints by region: Kenyon cells, MBONs and DANs (70,55,95); smell (45,60,90); lateral
horn (85,50,65); central complex (50,80,60); visual projection neurons (40,70,80); descending neurons
(90,65,40); everything else (48,52,64).

## How the maps are built (in case the app wants to rebuild or explain them)
- **Anatomical:**
  1. Take the soma x, y (front view). Rank x into 64 equal-count columns; within each column, rank y
     into 48 equal-count rows.
  2. Blend each rank with the raw position (`mix` = 0.3 of rank, 0.7 of position), so the true shape
     survives.
  3. Empty dots take their nearest dot's neurons.
- **Wave:**
  1. Each neuron's response time is the step at which its distance from rest first reaches half its
     peak, over 16 unlabelled sniffs. Neurons that never react go last, ordered by synapse hops from
     the senses.
  2. Neurons are sorted into band rows (region, sub-row, hemisphere; midline neurons are those with no
     side).
  3. Each row's neurons are split evenly across its columns in response-time order.
  4. The named answer cells take columns 60–63 of the output rows.
- Both are rebuilt per brain version (family), not per specialist, since every specialist shares the
  family's neurons and layout.
