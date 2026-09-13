# Encoder v1 (frozen at `freeze-v1`)

The encoder turns an inbound event into sensory drive.  It sees three
things and nothing else: **who** (the account DID), **how toxic** (the VADER
compound score of the text, computed and discarded), and **whether Bosco
was addressed** (mention or reply).  Post text never reaches the encoder and
is never stored.

Config: `config/encoder_v1.yaml`.  Code: `src/bosco/encoder.py`.

## Account -> odor

`blake2b(DID)` seeds a deterministic draw of 12 glomeruli from the
valence-neutral list (53 annotated glomeruli minus 18 with documented innate
valence; the exclusions and citations are in the config).  All ORNs of the
chosen glomeruli are driven at 120 Hz for the episode.  This produces
2–5% Kenyon-cell activity, odor-specific (docs/phase2-stability.md).  The
same account is always the same odor; two accounts share on average
12·12/35 ≈ 4 glomeruli.

## Sentiment -> taste

VADER compound `c`.  `|c| <= 0.05`: no taste.  Otherwise rate =
150 Hz · min(1, |c| / 0.6); positive drives the 83 sugar/water labellar GRNs,
negative the 57 bitter GRNs.  Sugar at this rate produces the proboscis
extension reflex (MN9) in the model; bitter does not.

## Mention -> touch

If Bosco is mentioned or replied to, Johnston's-organ groups A and B
(sound / courtship-song sensitive; 138 neurons) are driven at 100 Hz.

## Not encoded

Topic, wording, images, links, follower counts, time of day (the clock
neurons get their own 24 h drive; see `docs/circadian.md`), and the
familiarity count.  Familiarity (number of prior direct interactions in the
ledger) is only a phrasebook key, chosen by the readout after the network
has acted.
