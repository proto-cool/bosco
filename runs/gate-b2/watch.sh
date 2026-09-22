#!/bin/sh
# One line per run every 5 minutes into runs/gate-b2/logs/progress.log, until no run is left.
cd /Users/nickd/projects/bosco
while pgrep -f "gate_b.py learn" > /dev/null; do
  {
    echo "== $(date '+%H:%M')  running: $(pgrep -f 'python scripts/gate_b.py learn' | wc -l | tr -d ' ')"
    for f in runs/gate-b2/logs/t*-*-s*.log; do
      n=$(basename "$f" .log)
      if [ -f "runs/gate-b2/$n/summary.json" ]; then echo "$n  done"; else
        grep -E "^\[.*item" "$f" | tail -1 | sed -E "s/^\[[^]]*\] /$n  /"; fi
    done
    echo
  } >> runs/gate-b2/logs/progress.log
  sleep 300
done
echo "== $(date '+%H:%M')  all runs finished" >> runs/gate-b2/logs/progress.log
