#!/usr/bin/env bash
# The tokenizer swap, and the two single-seed cells Patrick flagged.
#
# THE SWAP. Everything in this project says the tokenizer penalty EXISTS and tracks XLM-R's
# coverage. Nothing shows it CAUSES anything, and the XLM-R downstream comparison that might have
# stood in for it is gone -- not withdrawn as a bug, just genuinely uninformative once it turned
# out not to clear its own control.
#
# Same Yoruba text, same architecture, same compute. The only difference is which vocabulary
# turned characters into tokens:
#
#     yor        16,000 tokens,   3.734 chars/token,   69.1M tokens of text
#     yor_xlmr  250,002 tokens,   2.133 chars/token,  121.3M tokens of text
#
# Both use their whole corpus, so both models see the same 260M characters. At a fixed number of
# optimizer updates the XLM-R-vocabulary model therefore covers LESS TEXT for the same compute --
# which is the penalty, expressed as the thing it actually costs.
#
# Read in bits per character. Nats per token cannot compare two vocabularies, and this project
# has been caught by that twice already.
#
# The budget matches multi_yor (196.6M tokens of updates), so the existing run is a free fourth
# point rather than a separate experiment.
#
# THE SEEDS. Patrick's guard on report 07: the 4M and 1024M cells of the 33.8M ladder are single
# draws, and the +0.351 at 1024M rests on one seed against a spread of ~0.149. Both get two more.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf 'swap' > "$HERE/runs/_fleet_queue"

echo "=== A. tokenizer swap: our 16k BPE ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus yor \
     --data 69096452 --update-tokens 196608000 --seeds 0 1 2 --preset poc \
     --tag-prefix swap || echo "  own-BPE arm failed, continuing"

echo
echo "=== B. tokenizer swap: XLM-R's 250k vocabulary ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus yor_xlmr \
     --data 121339416 --update-tokens 196608000 --seeds 0 1 2 --preset poc \
     --tag-prefix swap || echo "  xlmr-vocab arm failed, continuing"

echo
echo "=== C. the two single-seed cells Patrick flagged ==="
bash "$HERE/py.sh" mlm_fleet.py --corpus eng_1b \
     --data 4000000 1024000000 --update-tokens 1024000000 --seeds 1 2 --preset poc \
     || echo "  seeding failed"

echo
echo "all configurations done"
