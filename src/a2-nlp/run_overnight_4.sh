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
echo "=== E. the tokenizer swap at the ladder budget ==="
# The first swap ran at 196.6M tokens of updates, which matches multi_yor and leaves both arms
# well short of their asymptote. At 1.024B both get closer to what they can actually do, which is
# the fairer question: does the penalty still cost something once neither model is starved?
printf 'swap62k' > "$HERE/runs/_fleet_queue"

bash "$HERE/py.sh" mlm_fleet.py --corpus yor \
     --data 69096452 --update-tokens 1024000000 --seeds 0 1 2 --preset poc \
     --tag-prefix swap62k || echo "  own-BPE arm failed, continuing"

bash "$HERE/py.sh" mlm_fleet.py --corpus yor_xlmr \
     --data 121339416 --update-tokens 1024000000 --seeds 0 1 2 --preset poc \
     --tag-prefix swap62k || echo "  xlmr-vocab arm failed, continuing"

echo
echo "all configurations done"
