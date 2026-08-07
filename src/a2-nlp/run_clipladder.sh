#!/usr/bin/env bash
# The 86M ladder re-run at clip 0.5.
#
# Report 05's 86M column measures a badly-clipped model. At the 64M cell, clipping at 0.5 instead
# of 1.0 moved the mean from 3.824 to 2.829 and the seed spread from 1.363 to 0.520 -- so every
# rung in that column is a measurement of a configuration we now know to be wrong, and the
# report's "read this column as anecdote" is doing a lot of work.
#
# Twelve cells: four data rungs, three seeds each -- 64M is already done at this
# clipping, three seeds of it, from the mitigation sweep, everything else identical to the original
# ladder so the two are directly comparable. About ten hours across both cards.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf 'clipladder' > "$HERE/runs/_fleet_queue"

bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b --queue clipladder \
     --lr 3e-4 --warmup 0.06 --clip 0.5 --tag-prefix clip05

echo
echo "all configurations done"
