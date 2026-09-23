#!/bin/sh
# Gate B2: T2 (reversal) and T3 (pictures) for every arm x 5 seeds, then T4/T5/T6 off the T1 states.
cd /Users/nickd/projects/bosco
(for arm in logistic real shuffle hash; do for s in 1 2 3 4 5; do echo "t2 $arm $s"; done; done) \
 | xargs -P 8 -L 1 sh -c 'uv run python scripts/gate_b.py learn --task $0 --arm $1 --seed $2 > runs/gate-b2/logs/$0-$1-s$2.log 2>&1'
echo "T2 batch done $(date)" >> runs/gate-b2/logs/batch.log
(for arm in logistic real shuffle hash; do for s in 1 2 3 4 5; do echo "t3 $arm $s"; done; done) \
 | xargs -P 8 -L 1 sh -c 'uv run python scripts/gate_b.py learn --task $0 --arm $1 --seed $2 > runs/gate-b2/logs/$0-$1-s$2.log 2>&1'
echo "T3 batch done $(date)" >> runs/gate-b2/logs/batch.log
(for t in t4 t6 t5; do for arm in real shuffle hash; do for s in 1 2 3 4 5; do echo "$t $arm $s"; done; done; done) \
 | xargs -P 8 -L 1 sh -c 'uv run python scripts/gate_b.py after --task $0 --arm $1 --seed $2 > runs/gate-b2/logs/$0-$1-s$2.log 2>&1'
echo "after batch done $(date)" >> runs/gate-b2/logs/batch.log
