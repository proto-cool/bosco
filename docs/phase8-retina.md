# Phase 8 gate: the retina

## Sparseness and reach

- KC fraction active, picture alone: 0.0182 (range at tag 0.0001 .. 0.1014)
- of the KCs active for the picture alone, visual KCs: 1.00
- KC fraction active, picture with an account odor: 0.0436
- mean MBON rate to the picture alone: 3.65 Hz; non-zero types: 10 of 37

**Check 1 (sparse): PASS**  **Check 2 (reaches the MB, carried by visual KCs): PASS**

## The same picture five times, thirty seconds apart

- familiarity before each: [0.423, 0.716, 0.876, 0.998, 0.999] (want rising)
- output resources used on a channel's visual KCs after: 0.125 (want > 0)
- a different picture right after: familiarity 0.581 (want below the fifth of the same)

**Check 3 (habituates, familiar): PASS**

## After the picture

- KC fraction active in the seconds after: 1 s 0.0000, 2 s 0.0000, 6 s 0.0000 (want falling to ~0)

**Check 4 (settles): PASS**

**GATE: PASS**

Notes (2026-09-15, rerun after familiarity was restricted to the cells a smell adds).  A
different picture right after five of the same reads as half familiar because they share
`img` and a brightness bin: channels are shared the way two feeds share a place, and the
picture-specific part (the hues) is what stays new.  Familiarity saturates by the fourth
exposure at 6 Hz because a driven visual KC fires past c_sat (3/s); that is the exposure
rule's ceiling, not the retina's.  Parameters were not moved.
