#!/usr/bin/env bash
# Redirected overnight queue.
#
# The schedule sweep was stopped early because it had already answered its question, in the
# opposite direction to the hypothesis. At the 64M cell:
#
#     warmup 0.06 (baseline)   0/3 diverged
#     warmup 0.15 (warm15)     2/2 diverged
#
# More warmup is worse, so warm25 -- which would have held the peak rate higher for longer still
# -- was dropped rather than run. The failure mode is not "never breaks through the plateau", as
# report 05 4 currently says. It is a run that learns normally and then loses it: warm15's two
# seeds peaked at 6.209 and 6.190 and fell back to 7.48, and English's unigram entropy is 7.491.
#
# What remains is ordered by what the poster needs, not by what is most interesting.
#
#   A. the twelve new languages, because the headline contrast rests on two per group and this
#      is the only work here that widens it. Cheap: ~8 min each.
#   B. lower peak learning rate, the one direction the divergence evidence actually points at
#   C. tighter gradient clipping, the other standard mitigation, and free -- --clip already
#      exists, so it needs no new code at midnight
set -uo pipefail          # not -e: one failed language or cell must not end the night
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf 'overnight3' > "$HERE/runs/_fleet_queue"      # one plan for the whole night; see write_plan

echo "=== A. from-scratch pretraining for the twelve new languages ==="
# The five-language budget -- 50M tokens, 196.6M tokens of updates -- so these rows drop straight
# into report 04's table rather than forming a separate experiment. Two of the twelve are much
# smaller than 50M (wol 4.8M, lug 20.1M); the trainer takes what exists, and the pass count on
# the dashboard shows them repeating more.
#
# Split across the two cards by hand. Each language is one cell, and a one-cell fleet uses one
# card -- so running them in sequence would leave half the machine idle for an hour and a half.
LANGS=$(bash "$HERE/py.sh" -c "
import json, os
try:
    rows = json.load(open(os.path.join('runs', 'gradient_languages.json'), encoding='utf-8'))
except Exception:
    rows = []
print(' '.join(r['corpus'] for r in rows if r['status'] in ('ok', 'existing')))
")

train_lang () {          # corpus, card
    bash "$HERE/py.sh" mlm_fleet.py --corpus "$1"         --data 50000000 --update-tokens 196608000 --seeds 0 --preset poc         --tag-prefix grad --n-gpu 1 --gpu-base "$2" || echo "  $1 failed, continuing"
}

EVEN=""; ODD=""          # set -u would abort on the first append otherwise
i=0
for corpus in $LANGS; do
    if [ $((i % 2)) -eq 0 ]; then EVEN="$EVEN $corpus"; else ODD="$ODD $corpus"; fi
    i=$((i + 1))
done
echo "  card 0:$EVEN"
echo "  card 1:$ODD"

( for c in $EVEN; do echo "--- $c (card 0) ---"; train_lang "$c" 0; done ) &
PID_A=$!
( for c in $ODD;  do echo "--- $c (card 1) ---"; train_lang "$c" 1; done ) &
PID_B=$!
wait $PID_A $PID_B

echo "=== B. lower peak learning rate at 86M ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b \
    --data 64000000 --update-tokens 1024000000 --seeds 0 1 2 \
    --preset afriberta --tag-prefix lr15 --lr 1.5e-4 --warmup 0.06 \
    || echo "  lr15 failed, continuing"

echo
echo "=== C. tighter gradient clipping at 86M ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b \
    --data 64000000 --update-tokens 1024000000 --seeds 0 1 2 \
    --preset afriberta --tag-prefix clip05 --lr 3e-4 --warmup 0.06 --clip 0.5 \
    || echo "  clip05 failed, continuing"

echo
echo "all configurations done"
