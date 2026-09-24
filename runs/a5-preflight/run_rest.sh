#!/bin/sh
cd /Users/nickd/projects/bosco
for cfg in real-type-cut layered-type-cut hash-type-cut free-type-cut real-neuron-cut layered-neuron-cut hash-neuron-cut free-neuron-cut real-neuron-full layered-neuron-full hash-neuron-full free-neuron-full; do
  uv run python scripts/a5_preflight.py --only $cfg >> runs/a5-preflight/rest.log 2>&1
  echo "[$cfg] exit $? $(date +%H:%M:%S) free_mem_pages $(vm_stat | awk '/Pages free/ {print $3}')" >> runs/a5-preflight/rest.log
done
echo done > runs/a5-preflight/finished_rest
