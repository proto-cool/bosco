# Circadian drive

The 52 annotated clock neurons (s-LNv, l-LNv, 5th-LNv/LNd6, LNd_b/c, DN1a,
DN1pA/B, LPN_a/b) receive a Poisson drive whose rate follows a 24 h cosine,
rectified: morning cells peak at 07:00 local, evening cells at 19:00,
amplitude 8 Hz (`config/circadian_v1.yaml`).  The drive is part of every
episode's stimulus, so the same event at a different hour is a different
stimulus, and the seed plus the logged hour replays it exactly.

His time zone is `tz` in the config: America/New_York from his first day,
America/Denver from 2026-09-16. A replay recomputes the hour from the
current tz, so the change is a boundary like a change of learning rule:
the agent writes a `clock` control row and a snapshot on the first start
after it, and the day reports are re-bucketed once (`Panel.write_days`).

There is no posting-time gate.  Whatever daily rhythm Bosco's actions show
is produced by the network.
