#!/usr/bin/env bash
# Measure the seed spread at the budget the ladders actually ran at.
#
# Reports 04 and 05 judge every difference against a spread of 0.049, and that number was
# measured on the 33.8M model, on Yoruba, at 196.6M tokens of updates -- a different corpus and
# a fifth of the compute of any rung it is used to judge. Two load-bearing claims rest on it:
# that the data axis is flat past 64M (+0.007 and +0.017, "inside the spread"), and that Yoruba
# and English decelerate alike (0.025 apart, "half the spread"). Neither can be falsified until
# the spread is measured here.
#
# Ordered cheapest-and-most-valuable first. Within a fleet the scheduler runs the longest cells
# first, which is right for card utilisation and wrong for what a person wants to see when they
# check back in two hours -- so the ordering that matters is done here, between fleets.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1/3  33.8M seeds, English (the spread reports 04 and 05 lean on) ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b --queue seedcheck

echo
echo "=== 2/3  33.8M seeds, Yoruba (the transfer claim in report 05 section 3) ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus yor --data 64000000 --update-tokens 1024000000 \
     --seeds 1 2 --preset poc

echo
echo "=== 3/3  86M seeds at the rungs never repeated ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b --queue seedcheck86

echo
echo "all three fleets done"
