#!/bin/sh
cd /Users/nickd/projects/bosco
for arm in real layered; do
  uv run python scripts/a4_pilot.py train --arm $arm >> runs/a4-pilot/$arm.log 2>&1
  echo "[$arm] exit $? $(date +%H:%M:%S)" >> runs/a4-pilot/all.log
done
uv run python scripts/a4_pilot.py report > runs/a4-pilot/report.log 2>&1
echo "[report] exit $? $(date +%H:%M:%S)" >> runs/a4-pilot/all.log
echo done > runs/a4-pilot/finished
