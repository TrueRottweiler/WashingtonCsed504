#!/usr/bin/env bash
# Card 0's queue for the next two days: wait for study 1, then run half of study 4.
#
# Waits on a marker in the log rather than on the process, because pgrep cannot see Windows
# processes from Git-bash and a script that thinks a run has finished when it has not will
# happily start a second job on the same card.
#
# Written once and not edited while running: bash re-reads a script from a byte offset as it
# executes, so editing a running script runs the edit.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
STUDY1_LOG="$LOG_DIR/study1_correlation.log"
MARKER='DOES PRETRAINING LOSS PREDICT DOWNSTREAM SCORE'

echo "[chain0] waiting for study 1 to finish ($(date))"
while ! grep -q "$MARKER" "$STUDY1_LOG" 2>/dev/null; do
    sleep 60
done
echo "[chain0] study 1 finished ($(date))"

echo "[chain0] starting study 4, seeds 0-5, on card 0 ($(date))"
bash "$HERE/py.sh" study_clip_prevention.py --gpu 0 --seeds 0,1,2,3,4,5 \
    >> "$LOG_DIR/study4_clip_card0.log" 2>&1
echo "[chain0] study 4 half finished ($(date))"
