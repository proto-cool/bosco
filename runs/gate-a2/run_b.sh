#!/bin/sh
for e in all_kc antenna; do
  for s in 1 2; do uv run python scripts/gate_a2.py run --arm real --seed $s --eyes $e > runs/gate-a2/logs/b-real-s$s-$e.log 2>&1; done
  uv run python scripts/gate_a2.py run --arm shuffle --seed 1 --eyes $e > runs/gate-a2/logs/b-shuffle-s1-$e.log 2>&1
done
echo done > runs/gate-a2/logs/finished_b
