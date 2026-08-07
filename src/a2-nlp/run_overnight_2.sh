#!/usr/bin/env bash
# Second overnight queue: widen the language gradient.
#
# The study's headline contrast is computed from two languages in each group, and the POC's own
# output says it has no degrees of freedom behind it. This adds up to twelve more, split by
# whether XLM-R pretrained on them.
#
# Phase A and B need no GPU at all -- streaming, tokenizing and counting -- so they run
# alongside the schedule sweep already on the cards. Phase C waits, because it does not.
#
# Nothing here is gated on Patrick's XLM-R configuration. The tokenizer-fit gradient and the
# from-scratch pretraining loss are both measurable without a working XLM-R fine-tune; only the
# downstream contrast needs that, and it is not attempted here.
set -uo pipefail          # deliberately NOT -e: a language that fails must not end the night
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_LOG="$1"

echo "=== A. preparing corpora (no GPU; runs beside the sweep) ==="
bash "$HERE/py.sh" prepare_gradient_languages.py

echo
echo "=== B. the tokenizer-fit gradient (no GPU) ==="
bash "$HERE/py.sh" gradient_table.py

echo
echo "=== C. waiting for the schedule sweep to release the cards ==="
# Poll the sweep's own completion marker rather than the process table: pgrep under Git-bash
# cannot see Windows processes, which is why an earlier queue script launched immediately
# instead of waiting and shared the cards with the fleet it was meant to follow.
while ! grep -q "all configurations done" "$SWEEP_LOG" 2>/dev/null; do
    sleep 60
done
echo "cards free"

echo
echo "=== D. from-scratch pretraining for each new language ==="
# The same budget the five-language comparison used -- 50M tokens, 196.6M tokens of updates --
# so the new rows drop straight into report 04's table instead of forming a separate experiment.
for corpus in $(bash "$HERE/py.sh" -c "
import json, os
try:
    rows = json.load(open(os.path.join('runs', 'gradient_languages.json'), encoding='utf-8'))
except Exception:
    rows = []
print(' '.join(r['corpus'] for r in rows if r['status'] in ('ok', 'existing')))
"); do
    echo
    echo "--- $corpus ---"
    bash "$HERE/py.sh" mlm_fleet.py --corpus "$corpus" \
        --data 50000000 --update-tokens 196608000 --seeds 0 --preset poc \
        --tag-prefix grad || echo "  $corpus failed, continuing"
done

echo
echo "second queue done"
