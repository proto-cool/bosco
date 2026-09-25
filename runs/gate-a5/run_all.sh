#!/bin/sh
cd /Users/nickd/projects/bosco
for s in 1 2 3; do
  for cfg in "real type" "layered type" "hash type" "free type" "real neuron"; do
    set -- $cfg
    uv run python scripts/gate_a5.py run --arm $1 --mode $2 --seed $s >> runs/gate-a5/logs/$1-$2-s$s.log 2>&1
    echo "[$1-$2-s$s] exit $? $(date +%H:%M:%S)" >> runs/gate-a5/logs/all.log
  done
done
echo done > runs/gate-a5/logs/finished
