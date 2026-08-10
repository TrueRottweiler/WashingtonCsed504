#!/usr/bin/env bash
# Card 0: finish Patrick's swap sweep, then pick my learning-rate extension back up.
#
# The extension was stopped mid-flight to give Patrick's blocked work the whole card. It writes a
# per-cell record and runs with reuse=True, so restarting it costs the one cell that was in
# progress and nothing else -- 55 of its 60 runs are already on disk.
set -u
cd "$(dirname "$0")"
. ./queue_lib.sh
hold_lock queue_card0

wait_for study_swap_downstream.py

echo "[queue] resuming the learning-rate extension on card 0"
bash py.sh study_lr_transfer.py --gpu 0
echo "[queue] card 0 clear at $(date -Is)"
