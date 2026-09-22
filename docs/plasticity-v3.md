# He can read an account as bitter

**The rule changes, and the measurement says why.**  Two earlier proposals against the one-way
ratchet were refused (`docs/plasticity-v2.md`).  Both were the same kind of thing — rescale one
compartment against the other — and both only moved where everybody sat together.  They were
aimed at the wrong defect.

## The defect is arithmetic, not sign

His verdict on an account is probed by presenting **the account's odor alone**
(`Agent.account_signature`: 500 ms, from rest).  Every pairing lands on **the whole mixture the
account arrived in** — its words, its topics, the feed it was read in, its pictures.

Measured on his live weights over his own logged windows (`scripts/kc_overlap.py`, 12 accounts):

| | share of an account's probe cells |
|---|---|
| lit by that account's **own** windows | **0.883** |
| lit by **any other** window | **0.475** |

The probe reads the right cells — that part was never broken.  But across 6,199 logged windows,
an account he has read 23 times supplies about **0.7%** of the depression its own verdict is read
from.  The other 99.3% is everybody else's posts coming through the same cells.

That is the r ≈ 0.  No rescaling of a quantity that is 99.3% background can put information into
it, which is why extinction and homeostasis were never going to work whatever their sign.

## What changes

Three parts, in `config/plasticity_v1.yaml` under `credit:`.

1. **`mode: share`** — a pairing about an account teaches the cells its odor owns, weighted by
   `1 / (number of accounts whose odor lights that cell)`.  A cell twenty accounts drive is
   nobody's, and learns almost nothing about any of them.  (`mode: own`, the flat signature, is
   the weaker variant; `mixture` is the v1 rule.)
2. **The verdict is read through the same weights**, so what taught a cell and what reads it
   agree (`MushroomBody.learned_valence(weights=)`).
3. **`contrast: true`** — the verdict is read against the compartment's own level rather than
   against nothing (`compartment_levels()`, over the cells that carry his traffic).  The standing
   offset was never in the weights; it was in reading a difference against zero.

The contrast level is **flat**, deliberately: it is a property of the compartment that every odor
inherits, not of the account.  Weighting it the way the read is weighted was tried and measured —
it moves verdicts *up* by about +0.065, the wrong way.

## First: the harness was lying

`scripts/plasticity_v2_gate.py` presented each logged window one simulated second after the last.
Short-term depression on the sensory afferents never recovers under that, and the mushroom body
goes nearly silent.  Over 250 of his real windows from a naive start:

| presentation | KCs active/window | spikes each (c_sat 3) | floor | LTM edges formed |
|---|---|---|---|---|
| back to back | 4 | 0.28 | +0.003 | **0** |
| from rest | **165** | **7.92** | **+0.047** | **6,904** |

About **2% of his real KC activity, no long-term memory at all, and a hundredfold too little
depression**.  His windows are half a minute apart with idle time between them, which is long
enough to recover; simulating those gaps costs a wall hour per hour of him, so `credit_gate.py`
presents each window from rest instead — fully recovered, which is what he is by the time the
next post arrives.

**Extinction and homeostasis were both refused on the old harness.  Those refusals are no longer
safe** and should be re-run before they are treated as settled.  They are still disabled.

## The result

6,199 logged windows, his own clock (98.1 h of his life), naive start, 103 accounts he read five
or more times, one process per arm.

| rule | mean | net-bitter (n=51) | net-sweet (n=52) | r | bitter below zero |
|---|---|---|---|---|---|
| `mixture` (what he runs) | +0.2806 | +0.2833 | +0.2779 | **+0.00** | 0 of 51 |
| **`share` + contrast** | +0.1840 | **+0.1604** | **+0.2073** | **+0.36** | 4 of 51 |

The control arm reproduces his live weights closely — live is mean +0.317, bitter +0.318, sweet
+0.318, r +0.081, 0 of 51 below zero — which is what says the harness now tells the truth.

**The two accounts the requirement named:**

| account | posts read | net VADER | `mixture` | `share` + contrast |
|---|---|---|---|---|
| `did:plc:6kv5inxd5x4tuydx` | 23 | −8.79 | +0.2207 | **−0.0158** |
| `did:plc:efh23t4xq6zcocom` | 155 | +102.55 | +0.3128 | **+0.5575** (103rd of 103) |

Under the rule he runs, the bitter account reads *sweeter* than the sweet one.  Under this rule
it reads negative, and the sweet one is the sweetest account he knows.

## What is still not true

Only **4 of 51** net-bitter accounts land below zero.  The ordering is right across the whole
population, but zero sits high, so an account whose posts were only mildly sour still reads
positive.  Where zero falls is a readout-calibration question rather than a learning one, and it
is now cheap to ask: `credit_gate.py` saves its final weights, so a different read can be probed
in seconds instead of by another two-hour replay.

Nothing here was tuned on his feed or on outcomes.  The taste gains are the published ones and
were not touched; `share` has no parameter to set, and the contrast has nothing in it to tune.

## Determinism

`Agent.ownership()` is a function of the cached signatures — of who he has met.  A span replayed
against a later `account_kcs.json` computes different shares and will not reproduce, so
**`account_kcs.json` must be snapshotted beside `brain_state.npz`**.

## Provenance

`scripts/kc_overlap.py` (the overlap measurement), `scripts/credit_gate.py` (both arms, the
harness fix, checkpoints, saved weights), `config/plasticity_v1.yaml` `credit:`, and the unit
test `test_credit_confinement_lets_a_sour_account_read_sour`.

This is a behaviour change after `freeze-v1`: it needs its own tag, an `EXPERIMENT.md` §2d note,
and a `plasticity` control row via `Agent.PLASTICITY_VERSION`.
