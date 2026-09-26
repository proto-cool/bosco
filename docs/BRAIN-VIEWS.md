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
