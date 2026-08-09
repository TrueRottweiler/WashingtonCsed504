#!/usr/bin/env bash
# Card 1's queue for the next two days: wait for study 2, then run the other half of study 4.
#
# Seeds 6-11 rather than a clip value or a data rung, so that if one card runs hotter or slower
# than the other, that difference lands on the seed axis instead of on the treatment being
# measured. Splitting an experiment by its own independent variable is how a hardware quirk
# becomes a finding.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
STUDY2_LOG="$LOG_DIR/study2_lr_transfer.log"
MARKER='DOES THE BEST LEARNING RATE TRANSFER'

echo "[chain1] waiting for study 2 to finish ($(date))"
while ! grep -q "$MARKER" "$STUDY2_LOG" 2>/dev/null; do
    sleep 60
done
echo "[chain1] study 2 finished ($(date))"

echo "[chain1] starting study 4, seeds 6-11, on card 1 ($(date))"
bash "$HERE/py.sh" study_clip_prevention.py --gpu 1 --seeds 6,7,8,9,10,11 \
    >> "$LOG_DIR/study4_clip_card1.log" 2>&1
echo "[chain1] study 4 half finished ($(date))"
