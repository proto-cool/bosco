#!/bin/bash
cd /Users/nickd/projects/bosco
uv run python scripts/a4_broad.py baselines > runs/a4-broad/baselines.log 2>&1
echo "exit $?" >> runs/a4-broad/baselines.log
uv run python scripts/a4_broad.py train --arm real > runs/a4-broad/train-real.log 2>&1
echo "exit $?" >> runs/a4-broad/train-real.log
