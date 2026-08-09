#!/usr/bin/env bash
# After study 4's half on card 0, extend study 2's learning-rate range.
#
# Three of five languages peaked at 7e-4, the top of the original sweep, so their optima lie
# outside it and their best-of-sweep numbers are not quotable. The extension adds 8.5e-4 and
# 1e-3 for all five languages -- not just the three that ran out of room, because extending only
# those recreates the asymmetry report 08 spent three passes removing.
#
# Guarded by a lock directory. This waits ten hours unattended, and it is easy to end up with two
# copies polling the same marker -- at which point both fire at once and two processes write the
# same checkpoint directory. mkdir is atomic on every filesystem we care about, which a
# "test -f then touch" pair is not.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
LOCK="$LOG_DIR/.chain_card0_after.lock"
MARKER='DOES TIGHTER CLIPPING PREVENT FAILURES'

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[after0] another copy already holds $LOCK -- exiting ($(date))"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

echo "[after0] holding the lock, waiting for study 4 on card 0 ($(date))"
while ! grep -q "$MARKER" "$LOG_DIR/study4_clip_card0.log" 2>/dev/null; do
    sleep 120
done
echo "[after0] study 4 card 0 finished ($(date))"

echo "[after0] extending the learning-rate sweep ($(date))"
bash "$HERE/py.sh" study_lr_transfer.py --gpu 0 >> "$LOG_DIR/study2_lr_transfer.log" 2>&1
echo "[after0] extension finished ($(date))"
