#!/usr/bin/env bash
# swh was skipped: its log file was still held open by a killed process when the fleet tried to
# clear it, so the cell failed before training. The queue continued, which is what it should do
# -- this picks the language back up once the cards are free.
#
# Polls the driver's own completion marker rather than the process table: pgrep under Git-bash
# cannot see Windows processes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$1"
while ! grep -q "all configurations done" "$LOG" 2>/dev/null; do
    sleep 60
done
echo "queue finished; retrying swh"
rm -f "$HERE/logs/grad_swh_50M_12k_s0.log"
bash "$HERE/py.sh" mlm_fleet.py --corpus swh \
    --data 50000000 --update-tokens 196608000 --seeds 0 --preset poc --tag-prefix grad
echo "swh done"
