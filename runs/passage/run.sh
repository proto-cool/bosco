#!/bin/sh
cd /Users/nickd/projects/bosco
for c in single sniff aware llm-ref; do
  uv run python scripts/passage_check.py embed --cond $c >> runs/passage/$c.log 2>&1
  code=$?
  if [ $code -ne 0 ] && [ $c != single ] && [ $c != sniff ]; then
    echo "[$c] retry in isolated env" >> runs/passage/$c.log
    uv run --with transformers==4.46.3 --with sentence-transformers==3.3.1 python scripts/passage_check.py embed --cond $c >> runs/passage/$c.log 2>&1
    code=$?
  fi
  echo "[$c] exit $code $(date +%H:%M:%S)" >> runs/passage/all.log
done
uv run python scripts/passage_check.py report > runs/passage/report.log 2>&1
echo done > runs/passage/finished
