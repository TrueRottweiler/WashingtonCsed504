#!/usr/bin/env bash
# Overnight: the downstream rows nobody has measured, then the swap at a serious budget.
#
# Waits for the swap/seeding batch already on the cards. Polls that driver's completion marker
# rather than the process table, because pgrep under Git-bash cannot see Windows processes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_FOR="$1"

while ! grep -q "all configurations done" "$WAIT_FOR" 2>/dev/null; do
    sleep 60
done
echo "earlier batch finished; starting"

# The best from-scratch Yoruba model on disk (val 2.315). Every downstream row below uses this
# one checkpoint, so the rows differ by task and budget rather than by which model they got.
CKPT="runs/yor_64M_62.5k_s0"

echo
echo "=== D. the from-scratch rows report 06 is missing ==="
# See downstream_rows.py -- the reasoning lives with the code that acts on it.
bash "$HERE/py.sh" downstream_rows.py || echo "  downstream block failed, continuing"

echo
echo "=== E. the swap at MATCHED COMPUTE, which the first pass did not have ==="
# The first swap matched STEPS, and that was a design error. A 250k output projection makes each
# step 5.1x more expensive, so the two arms did not get the same compute at all:
#
#     16k BPE    12,000 steps   8 min/seed    1.135 +-0.035 bits/char
#     250k      12,000 steps   41 min/seed    1.056 +-0.142 bits/char
#
# Read as matched steps that says the vocabularies are indistinguishable. Read honestly it says
# the 250k vocabulary needed FIVE TIMES THE COMPUTE to draw level, which is the penalty rather
# than its absence.
#
# The matched-compute comparison is 250k at 12k steps against 16k at ~61k steps -- same wall
# clock. Only the 16k arm is missing, and it is cheap. The 250k arm at 62.5k steps would cost
# ten hours to answer the confounded question again, so it is dropped.
printf 'swap62k' > "$HERE/runs/_fleet_queue"

bash "$HERE/py.sh" mlm_fleet.py --corpus yor      --data 69096452 --update-tokens 1024000000 --seeds 0 1 2 --preset poc      --tag-prefix swap62k || echo "  matched-compute arm failed, continuing"

echo
echo "all configurations done"
