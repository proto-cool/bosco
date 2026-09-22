#!/bin/sh
# Gate B2, T1, every arm x 5 seeds.
cd /Users/nickd/projects/bosco
(for arm in logistic real shuffle hash; do for s in 1 2 3 4 5; do echo "$arm $s"; done; done) \
 | xargs -P 8 -L 1 sh -c 'uv run python scripts/gate_b.py learn --task t1 --arm $0 --seed $1 > runs/gate-b2/logs/t1-$0-s$1.log 2>&1'
echo "T1 batch done $(date)" >> runs/gate-b2/logs/batch.log
