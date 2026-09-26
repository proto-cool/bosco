#!/bin/bash
# overnight queue on the 3080: preflight every run, then train only those that pass (docs/SPECIALIST-PILOT.md)
cd ~/bosco
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/specialist-pilot/logs; mkdir -p $L
RUNS="hate:real hatemoji:real topic:real intent_massive:real intent_clinc:real hate:layered hatemoji:layered topic:layered intent_massive:layered intent_clinc:layered all:real"
for r in $RUNS; do t=${r%%:*}; a=${r##*:}
  ~/.local/bin/uv run python scripts/v1_pilot.py preflight --task $t --arm $a > $L/preflight-$t-$a.log 2>&1
  echo "exit $?" >> $L/preflight-$t-$a.log
done
for r in $RUNS; do t=${r%%:*}; a=${r##*:}
  if grep -q '"pass": true' runs/specialist-pilot/preflight-$t-$a.json 2>/dev/null; then
    ~/.local/bin/uv run python scripts/v1_pilot.py train --task $t --arm $a > $L/train-$t-$a.log 2>&1
    echo "exit $?" >> $L/train-$t-$a.log
  else
    echo "skipped: preflight did not pass" > $L/train-$t-$a.log
  fi
done
echo done > runs/specialist-pilot/queue-finished
