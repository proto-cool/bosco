# Gate C1 — the fly decides (pre-registered 2026-09-23)

Step 1 of `BRIEF.md`. Until now the decider's verdict was arithmetic on the
fly's synapse weights (`learned_valence`), trained offline from cached
Kenyon-cell codes. Here the fly decides with his output neurons, in the live
brain, with nothing cached. **Question: does a fly trained the decider's way
say sweet or bitter with his mushroom-body output neurons, and how well?**

## Harness check, before writing this (no labels used)

Six held-out sentences on the naive real brain, 4 presentations each:
approach-side and avoid-side MBONs fire 20–110 spikes per 500 ms
presentation, with large presentation-to-presentation spread; 2 of 6
sentences light ≤0.4% of Kenyon cells and leave the MBONs silent or near
it. 0.47 s per presentation. That is the whole of what was looked at.

## The fly

- Real MaleCNS central brain, `config/model_v1.yaml` as the decider runs it
  (MBON→DAN→KC feedback on), the B4 antenna (`data/cache/gate-b3/antenna.json`,
  unchanged), the rule as the decider runs it (mixture credit, contrast,
  per-item consolidation). **One fly.** No swarm.
- **Every presentation runs the kernel on the fly's current weights.** The
  Kenyon-cell code that is paired is the one his brain produced at that
  moment. No cache is read.

## Training (the decider's protocol)

- The 400 B3 training sentences (`sst_train`, 200 sweet, 200 bitter), one
  pass in a seeded order, 30 s apart (3.3 biological hours). Each: present
  (exposure), then pair with sugar or shock at full strength.
- Rehearsal: each item presented and paired again at +1 h, +3 h, +24 h after
  its first pairing (`REHEARSE_H`). All events run in clock order.
- The test happens at the last event + 30 s (~27.4 h), what a fly bootstrapped
  yesterday is like today. The clock does not move during the test, and test
  presentations record no exposure (reading must not change him).

## The read: approach vs avoid

- **Approach MBONs:** the MBON types in compartments of punishment DANs
  (PPL1): MBON11, 12, 13, 14, 18, 19, 23, 31, 32, 35 (26 neurons).
  **Avoid MBONs:** the MBON types in compartments of reward DANs (PAM):
  MBON01, 02, 03, 05, 06, 07, 09, 10, 21, 24, 26, 27, 29, 30, 33 (41
  neurons). Aso et al. 2014 sign: reward depresses avoidance-driving
  MBONs. MBON04 (both) and the α'3 exposure MBONs are left out. From
  `config/mb_compartments.yaml`, not chosen on data.
- One item = 8 presentations from rest (seeds fixed). A = approach spikes,
  V = avoid spikes, summed over the 8. **Score s = (A − V) / (A + V + 1)**,
  −1..1. A silent item scores 0.
- **Neutral, primary:** the fly's own recall midpoint (B4): the mean score of
  his last 30 trained sweet and last 30 trained bitter items, averaged.
  **Secondary:** s = 0, where approach and avoid balance.
- Beside it, on the same presentations: the old weight read
  (`learned_valence` on the live Kenyon-cell code), neutral by the same rule.

## Arms and seeds

real, hash, shuffle (the B-gate brains), training orders seed 1 and 2:
six flies, run in parallel, one process each.

## Measured

- **Held-out text:** the 400 `sst_heldout` sentences. Balanced accuracy at the
  primary neutral (the headline), at s = 0, and Spearman against the labels.
- **Pictures from text alone:** the 300 clear held-out OASIS pictures
  (`oasis_heldout[:300]`: 150 top, 150 bottom by valence). Balanced accuracy
  at the median of their own scores (the calibration rule for an untrained
  modality), and Spearman.
- **Nick's probes:** the five sentences and two pictures, score minus neutral.
- Recall gap (trained sweet − trained bitter), the share of held-out items
  with a silent read (A + V = 0), and the Kenyon-cell active fraction at test
  vs naive for the same items (the feedback's effect).

## Decision rule

On the real arm, headline = mean over the two seeds.

1. **The fly decides by his output neurons** (adopted into the decider) if
   the headline is **≥ 0.56**: chance plus twice the measurement floor.
2. Reported beside it, not gating adoption: the headline against the weight
   read on the same fly (a gap under 0.03 is "the same"), and against hash
   and shuffle.
3. If the headline is under 0.56, the output neurons do not carry what he
   learned under this read. That is written up as it stands. The decider
   does **not** quietly go back to the weight read. The next step is a
   separate, pre-registered diagnosis.

Nothing above is changed after the numbers are seen.

Runner: `scripts/gate_c1.py` (`run --arm --seed`, `report`). Results:
`runs/gate-c1/`, `docs/gate-c1-results.md`.
