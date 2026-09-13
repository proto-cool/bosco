# Circadian drive

The 52 annotated clock neurons (s-LNv, l-LNv, 5th-LNv/LNd6, LNd_b/c, DN1a,
DN1pA/B, LPN_a/b) receive a Poisson drive whose rate follows a 24 h cosine,
rectified: morning cells peak at 07:00 local, evening cells at 19:00,
amplitude 8 Hz (`config/circadian_v1.yaml`).  The drive is part of every
episode's stimulus, so the same event at a different hour is a different
stimulus, and the seed plus the logged hour replays it exactly.

There is no posting-time gate.  Whatever daily rhythm Bosco's actions show
is produced by the network.
