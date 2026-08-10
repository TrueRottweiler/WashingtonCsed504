#!/usr/bin/env bash
# Card 1: finish Patrick's tokenizer seeds, then pick the clipping study back up.
#
# The clipping study is the one that turned the three-seed ladder over: at fifteen seeds against
# thirteen it showed that tighter clipping does NOT prevent divergence (4/15 against 3/13, Fisher
# p = 1.00) while it does improve and tighten the runs that survive. The failure RATE is the part
# still barely resolved -- twelve seeds settles a spread and only gestures at a rate -- so the
# remaining cells are the ones that matter most for the one claim still open.
set -u
cd "$(dirname "$0")"
. ./queue_lib.sh
hold_lock queue_card1

wait_for study_tokenizer_seeds.py

echo "[queue] resuming the clipping study on card 1"
bash py.sh study_clip_prevention.py --gpu 1 --seeds 6,7,8,9,10,11
echo "[queue] card 1 clear at $(date -Is)"
