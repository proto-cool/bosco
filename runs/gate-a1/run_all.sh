#!/bin/sh
for n in all 400; do for s in 1 2; do for arm in real shuffle hash free; do
  uv run python scripts/gate_a1.py run --arm $arm --seed $s --n-train $n > runs/gate-a1/logs/$arm-s$s-n$n.log 2>&1
done; done; done
echo ALL DONE > runs/gate-a1/logs/finished
