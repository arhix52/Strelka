#!/usr/bin/env bash
# Sample Apple's device-wide GPU hardware counters against a running renderer.
#
#   tools/gpu_counters.sh <pid|--idle|--run "<command>"> [seconds]
#
# `--run` is the one to use on a renderer: it launches the command, waits for it
# to print STRELKA_RENDER_BEGIN -- which HeadlessApp emits once the first sample
# has completed -- and only then starts sampling. Counters are a time average,
# and on a heavy scene the load, the texture cache and the acceleration
# structure build are most of a short run; sample across those and the numbers
# describe a BVH build rather than a render.
#
# Instruments templates do not collect performance counters by default. The
# working setting is Recording Options -> Counter Set -> Performance Limiters.
# Xcode 26 serializes the M4-compatible version of that preset as profile 13;
# `xctrace --instrument "Metal GPU Counters"` otherwise selects legacy profile 3
# and the trace contains "Selected counter profile is not supported". This
# script patches Xcode's Blank template with the correct option before recording.
#
# And they are device-wide. Attaching to a pid does not isolate them -- an idle
# desktop can show substantial ALU utilization and fragment occupancy from the
# compositor. Take an `--idle` control under the same desktop conditions. Treat
# it as a contamination baseline; percentage/limiter counters are ratios and
# must not be arithmetically subtracted.
#
# The trace is deleted on the way out unless STRELKA_KEEP_TRACE is set. XML is
# streamed into the summarizer because a short export can be hundreds of MB.
# See docs/gpu-counters.md for the GUI workflow, overrides, and interpretation.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
LOG=""
CHILD=""
KEEP="${STRELKA_KEEP_TRACE:-}"
SKIP_EXPORT="${STRELKA_SKIP_EXPORT:-}"
TRACE="$WORK/counters.trace"
RECORD_LOG="$WORK/record.log"
EXPORT_LOG="$WORK/export.log"
TEMPLATE="$WORK/Strelka GPU Counters.tracetemplate"

if [ -n "$SKIP_EXPORT" ] && [ -z "$KEEP" ]; then
    echo "STRELKA_SKIP_EXPORT requires STRELKA_KEEP_TRACE=<path>" >&2
    exit 1
fi

# Instruments writes a multi-gigabyte kernel trace next to the recording and
# leaves it behind whenever a recording is interrupted rather than ending on its
# own. Report old files rather than deleting a shared scratch path automatically.
warn_orphans() {
    local n size
    n=$(find "${TMPDIR:-/tmp}" -maxdepth 1 -name '*.ktrace' -mmin +10 2>/dev/null | wc -l | tr -d ' ')
    [ "${n:-0}" -eq 0 ] && return 0
    size=$(find "${TMPDIR:-/tmp}" -maxdepth 1 -name '*.ktrace' -mmin +10 -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
    echo "note: $n abandoned Instruments kernel traces (${size:-?}) in ${TMPDIR:-/tmp}" >&2
    echo "      rm -rf \"\${TMPDIR:-/tmp}\"*.ktrace   # when nothing is recording" >&2
    return 0
}

cleanup() {
    [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
    warn_orphans
    rm -rf "$WORK"
    [ -n "$KEEP" ] && [ -e "$KEEP" ] && echo "trace kept at $KEEP" >&2
    return 0
}
trap cleanup EXIT

# STRELKA_KEEP_TRACE=<path> keeps the recording instead of deleting it, for
# opening in the Instruments UI.
if [ -n "$KEEP" ]; then
    if [ -e "$KEEP" ]; then
        echo "refusing to overwrite existing trace: $KEEP" >&2
        exit 1
    fi
    TRACE="$KEEP"
fi

TARGET="${1:?usage: gpu_counters.sh <pid|--idle|--run \"cmd\"> [seconds]}"
if [ "$TARGET" = "--run" ]; then
    CMD="${2:?--run needs a command}"
    SECS="${3:-2}"
    LOG="$WORK/renderer.log"
    eval "$CMD" > "$LOG" 2>&1 &
    CHILD=$!
    for _ in $(seq 1 600); do
        grep -q STRELKA_RENDER_BEGIN "$LOG" 2>/dev/null && break
        kill -0 "$CHILD" 2>/dev/null || { echo "target exited before rendering:" >&2; tail -3 "$LOG" >&2; exit 1; }
        sleep 1
    done
    grep -q STRELKA_RENDER_BEGIN "$LOG" 2>/dev/null || { echo "timed out waiting for STRELKA_RENDER_BEGIN" >&2; exit 1; }
    # By executable name, not by command line: this script's own arguments
    # contain the renderer's, so a -f match finds the script itself.
    TARGET="$(pgrep -x StrelkaCLI | tail -1)"
    [ -n "$TARGET" ] || TARGET="$CHILD"
else
    SECS="${2:-2}"
fi
case "$SECS" in
    ''|*[!0-9]*) echo "seconds must be a positive integer: $SECS" >&2; exit 1 ;;
esac
[ "$SECS" -gt 0 ] || { echo "seconds must be greater than zero" >&2; exit 1; }

if [ "$TARGET" = "--idle" ]; then
    ATTACH=(--all-processes)
else
    ATTACH=(--attach "$TARGET")
fi

CUSTOM_TEMPLATE="${STRELKA_GPU_COUNTER_TEMPLATE:-}"
if [ -n "$CUSTOM_TEMPLATE" ]; then
    [ -f "$CUSTOM_TEMPLATE" ] || { echo "GPU counter template not found: $CUSTOM_TEMPLATE" >&2; exit 1; }
    RECORD_TEMPLATE=(--template "$CUSTOM_TEMPLATE")
    PROFILE_DESCRIPTION="custom template"
else
    DEVELOPER_DIR="$(xcode-select -p)"
    BLANK_TEMPLATE="$DEVELOPER_DIR/../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate"
    [ -f "$BLANK_TEMPLATE" ] || { echo "Instruments Blank template not found under $DEVELOPER_DIR" >&2; exit 1; }

    PROFILE_ID="${STRELKA_GPU_COUNTER_PROFILE:-}"
    if [ -z "$PROFILE_ID" ]; then
        XCODE_MAJOR="$(xcodebuild -version | awk 'NR == 1 { split($2, version, "."); print version[1] }')"
        if [ "${XCODE_MAJOR:-0}" -ge 26 ]; then
            PROFILE_ID=13
        else
            PROFILE_ID=3
        fi
    fi
    python3 "$SCRIPT_DIR/gpu_counter_tools.py" make-template \
        --profile-id "$PROFILE_ID" "$BLANK_TEMPLATE" "$TEMPLATE" || exit 1
    RECORD_TEMPLATE=(--template "$TEMPLATE" --instrument "Metal GPU Counters")
    PROFILE_DESCRIPTION="Performance Limiters (profile $PROFILE_ID)"
fi

echo "recording GPU counters: $PROFILE_DESCRIPTION, ${SECS}s" >&2
if ! xcrun xctrace record "${RECORD_TEMPLATE[@]}" "${ATTACH[@]}" \
    --time-limit "${SECS}s" --output "$TRACE" >"$RECORD_LOG" 2>&1; then
    tail -20 "$RECORD_LOG" >&2
    echo "xctrace record failed" >&2
    exit 1
fi
if grep -q "Selected counter profile is not supported" "$RECORD_LOG"; then
    tail -20 "$RECORD_LOG" >&2
    echo "GPU counter profile is unsupported; see docs/gpu-counters.md" >&2
    exit 1
fi

# Keeping the native trace is much faster than exporting its potentially
# multi-gigabyte XML. Require an explicit destination: otherwise cleanup would
# delete the temporary trace and a successful capture would leave no output.
if [ -n "$SKIP_EXPORT" ]; then
    echo "skipping XML export; trace kept at $TRACE" >&2
    exit 0
fi

if ! xcrun xctrace export --input "$TRACE" \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-counter-intervals"]' \
    2>"$EXPORT_LOG" | python3 "$SCRIPT_DIR/gpu_counter_tools.py" summarize; then
    tail -20 "$EXPORT_LOG" >&2
    echo "xctrace counter export failed" >&2
    exit 1
fi
