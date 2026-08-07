#!/usr/bin/env bash
# Overnight: find out whether the 86M model's failures are a schedule artifact.
#
# Thirteen 86M runs land as nine below 3.8 and four above 5.3 with nothing in between -- two
# populations, not a spread. Roughly a third of seeds never leave the unigram plateau inside
# their budget. Report 05 4 names two candidate explanations and tests neither:
#
#   1. the warmup floor is too short for this width at long budgets, so early training is
#      seed-dependent in a way that decides the whole run
#   2. 3e-4 sits near an instability edge at 86M, where 5e-4 and 1e-3 are known to collapse it
#
# This separates them. One rung (64M tokens), one budget (1.024B tokens of updates), three seeds
# per configuration, and exactly one thing changed at a time from the baseline we already have
# three seeds of:
#
#   baseline    lr 3e-4   warmup 0.06   -- already measured: 3.278 / 2.818 / 5.376, 1 of 3 failed
#   warm15      lr 3e-4   warmup 0.15
#   warm25      lr 3e-4   warmup 0.25
#   lr15        lr 1.5e-4 warmup 0.06
#
# The measure is NOT the mean. Averaging a collapsed run with a converged one describes neither,
# which is the mistake this project has already made twice. What matters is how many of three
# seeds clear the plateau, and a configuration going 3/3 where the baseline went 2/3 is the
# result worth having.
#
# Nine runs at ~1.57 h each, longest-first across two cards: about 7 hours.
#
# Every configuration is namespaced with --tag-prefix. The cell tag carries only
# (corpus, tokens, steps, seed, preset), so without it all four configurations would name their
# cells identically and overwrite one another.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA=64000000
UPD=1024000000

run_config () {          # prefix, lr, warmup
    echo
    echo "=== $1  (lr $2, warmup $3) ==="
    bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b \
        --data "$DATA" --update-tokens "$UPD" --seeds 0 1 2 \
        --preset afriberta --tag-prefix "$1" --lr "$2" --warmup "$3"
}

run_config warm15 3e-4   0.15
run_config warm25 3e-4   0.25
run_config lr15   1.5e-4 0.06

echo
echo "all configurations done"
