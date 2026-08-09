#!/usr/bin/env bash
# Last in card 0's queue: three more seeds per arm for the tokenizer penalty.
#
# claims_audit.py found the 0.144 bits/char headline does not clear its own null at n=3 (p=0.22).
# Six per arm does, at the observed effect size. The sample size was fixed before these runs
# started -- see the docstring in study_tokenizer_seeds.py -- so this is buying power, not
# running seeds until the p-value cooperates.
#
# Waits on the learning-rate extension, which itself waits on study 4. Same lock discipline as
# the other chains: mkdir is atomic, so a second copy exits instead of racing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$HERE/logs"
LOCK="$LOG_DIR/.chain_card0_seeds.lock"
MARKER='DOES THE BEST LEARNING RATE TRANSFER'

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[seeds] another copy holds $LOCK -- exiting ($(date))"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# The extension appends to study2's log, so wait for a SECOND occurrence of the marker: the
# first one is from the original sweep that already finished.
echo "[seeds] waiting for the learning-rate extension to finish ($(date))"
while [ "$(grep -c "$MARKER" "$LOG_DIR/study2_lr_transfer.log" 2>/dev/null || echo 0)" -lt 2 ]; do
    sleep 120
done
echo "[seeds] extension finished ($(date))"

echo "[seeds] starting three more seeds per tokenizer arm ($(date))"
bash "$HERE/py.sh" study_tokenizer_seeds.py --gpu 0 >> "$LOG_DIR/study5_tokenizer_seeds.log" 2>&1
echo "[seeds] finished ($(date))"
