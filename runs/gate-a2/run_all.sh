#!/bin/sh
for s in 1 2 3; do for arm in real shuffle hash free; do
  uv run python scripts/gate_a2.py run --arm $arm --seed $s > runs/gate-a2/logs/$arm-s$s-joint.log 2>&1
done; done
for s in 1 2; do for q in sweet dangerous junk; do
  uv run python scripts/gate_a2.py run --arm real --seed $s --only $q > runs/gate-a2/logs/real-s$s-$q.log 2>&1
done; done
echo ALL DONE > runs/gate-a2/logs/finished
