#!/usr/bin/env bash
# Card 1 runs Patrick's work in order, without a human between the jobs.
#
# The lock is not decoration. Relaunching this script while an earlier copy was still waiting
# started a SECOND study_ner_control_sweep against the same runs/ner_control_sweep.json, and the
# newcomer's four-row summary overwrote the incumbent's fourteen. Nothing was actually lost --
# the per-cell ft_*.json records are the data and the summary is derived from them -- but for a
# few minutes the dashboard and every reader of that file saw ten finished cells disappear. A
# study that writes a shared file must be single-instance, and the cheap way to guarantee that is
# to refuse to start rather than to remember not to.
set -u
cd "$(dirname "$0")"

LOCK=runs/.queue_card1.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[queue] another copy holds $LOCK -- refusing to start a second one"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# pgrep is the obvious way to write this and it does not work here. Git Bash's pgrep only sees
# processes in its own tree, so a study launched from another shell -- which is every study, since
# they are started detached -- reads as "not running" and the wait falls straight through. That is
# what started the duplicate sweep the lock above now catches: the lock was the second line of
# defence for a bug that was really in this function. Ask Windows instead.
running() {                        # $1 = script name
    local n
    n=$(powershell -NoProfile -Command \
        "@(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" |
           Where-Object { \$_.CommandLine -match '$1' }).Count" 2>/dev/null | tr -d '\r ')
    [ "${n:-0}" -gt 0 ]
}

wait_for() {                       # $1 = script name to wait out
    while running "$1"; do sleep 60; done
    echo "[queue] $1 finished at $(date -Is)"
}

wait_for study_ner_control_sweep.py

# The first pass rose monotonically across all eight rates and peaked at 0.594 on the highest one,
# so it measured the edge of its range rather than the floor. Re-running with the range extended
# costs only the new rates: the finished cells come back from reuse in seconds.
echo "[queue] extending the NER control sweep upward"
bash py.sh study_ner_control_sweep.py --gpu 1
echo "[queue] NER control sweep finished at $(date -Is)"

# The tokenizer seeds were already started by an older chain script, so wait them out rather than
# launching a competing copy.
wait_for study_tokenizer_seeds.py
echo "[queue] card 1 is clear at $(date -Is)"
