# Shared helpers for the per-card queues. Source this; do not run it.
#
# The reason this file exists is one bug, worth stating so nobody reintroduces it. The obvious way
# to wait for a study to finish is `pgrep -f study_foo.py`, and under Git Bash on Windows that only
# sees processes in its own tree. Every study here is launched detached, so pgrep reported "not
# running" for a study that was very much running, the wait fell straight through, and a second
# copy started against the same output file. Ten finished cells appeared to vanish from the
# summary. Nothing was actually lost -- the per-cell records are the data -- but the panel and
# anyone reading it were wrong for several minutes.
#
# So: ask Windows, and hold a lock as a second line of defence rather than a first.

running() {                        # $1 = script name, e.g. study_lr_transfer.py
    local n
    n=$(powershell -NoProfile -Command \
        "@(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" |
           Where-Object { \$_.CommandLine -match '$1' }).Count" 2>/dev/null | tr -d '\r ')
    [ "${n:-0}" -gt 0 ]
}

wait_for() {                       # $1 = script name to wait out
    if running "$1"; then
        echo "[queue] waiting for $1 ($(date '+%H:%M'))"
        while running "$1"; do sleep 60; done
    fi
    echo "[queue] $1 is not running ($(date '+%H:%M'))"
}

hold_lock() {                      # $1 = lock name; exits 0 if another copy holds it
    LOCK="runs/.$1.lock"
    if ! mkdir "$LOCK" 2>/dev/null; then
        echo "[queue] another copy holds $LOCK -- refusing to start a second one"
        exit 0
    fi
    trap 'rmdir "$LOCK" 2>/dev/null' EXIT
}
