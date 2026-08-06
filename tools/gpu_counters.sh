#!/usr/bin/env bash
# Sample Apple's GPU hardware counters against a running renderer.
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
# Two things about these counters decide how they have to be used.
#
# They are not in the public Metal API. `MTLDevice.counterSets` on an M1 Pro
# offers one set -- `timestamp`, one counter -- so nothing here is reachable
# through MTLCounterSampleBuffer. They come from Instruments, and only if the
# *instrument* is named: recording with the "Metal System Trace" or "Game
# Performance" templates produces zero counter rows and looks like a permissions
# problem. It is not one; no entitlement or signature is needed.
#
# Record them *alone*. Adding `--instrument "Metal Application"` alongside, to
# get the encoder and frame tracks in the same document, leaves
# `metal-gpu-counter-intervals` empty -- the raw `gpu-counter-value` samples are
# still there, but the aggregated table the UI draws its tracks from is not, and
# the counters simply do not appear. Take two recordings instead.
#
# And they are device-wide. Attaching to a pid does not isolate them -- an idle
# desktop reads ALU Utilization 56% and Fragment Occupancy 48%, which is the
# compositor. Only the difference against an idle control means anything, so
# take one with `--idle` and subtract.
#
# The trace and its XML are deleted on the way out. A two-second recording is
# ~70 MB and its export ~600 MB; leaving those around fills a disk in a few
# runs, which is how this script came to exist.
set -u
TARGET="${1:?usage: gpu_counters.sh <pid|--idle|--run \"cmd\"> [seconds]}"
CHILD=""
if [ "$TARGET" = "--run" ]; then
    CMD="${2:?--run needs a command}"
    SECS="${3:-2}"
    LOG="$(mktemp)"
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
# STRELKA_KEEP_TRACE=<path> keeps the recording instead of deleting it, for
# opening in the Instruments UI.
KEEP="${STRELKA_KEEP_TRACE:-}"
if [ -n "$KEEP" ]; then
    rm -rf "$KEEP"
    TRACE="$KEEP"
else
    TRACE="$(mktemp -d)/counters.trace"
fi
XML="${TRACE%.trace}.xml"
cleanup() {
    [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
    rm -f "${XML:-}" "${LOG:-}"
    [ -z "$KEEP" ] && rm -rf "$(dirname "$TRACE")"
    [ -n "$KEEP" ] && echo "trace kept at $KEEP" >&2
    return 0
}
trap cleanup EXIT

if [ "$TARGET" = "--idle" ]; then
    ATTACH=(--all-processes)
else
    ATTACH=(--attach "$TARGET")
fi

xcrun xctrace record --instrument "Metal GPU Counters" "${ATTACH[@]}" \
    --time-limit "${SECS}s" --output "$TRACE" >/dev/null 2>&1 || {
    echo "xctrace record failed" >&2; exit 1; }

xcrun xctrace export --input "$TRACE" \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-counter-intervals"]' \
    > "$XML" 2>/dev/null

python3 - "$XML" <<'PY'
import re, sys, collections
names = {}
tot = collections.defaultdict(lambda: [0.0, 0])
buf = ""
with open(sys.argv[1]) as f:
    for chunk in iter(lambda: f.read(1 << 20), ""):
        buf += chunk
        while True:
            i = buf.find("<row>")
            j = buf.find("</row>", i + 1)
            if i < 0 or j < 0:
                break
            row, buf = buf[i + 5:j], buf[j + 6:]
            m = re.search(r'<gpu-counter-name id="(\d+)" fmt="([^"]*)"', row)
            if m:
                names[m.group(1)] = m.group(2)
                nm = m.group(2)
            else:
                r = re.search(r'<gpu-counter-name ref="(\d+)"', row)
                nm = names.get(r.group(1), "?") if r else "?"
            v = re.search(r'<(?:percent|fixed-decimal)[^>]*>([0-9.eE+-]+)</', row)
            d = re.search(r'<duration[^>]*>(\d+)</duration>', row)
            if v:
                w = int(d.group(1)) if d else 1
                t = tot[nm]
                t[0] += float(v.group(1)) * w
                t[1] += w
for k in sorted(tot):
    print("%-44s %8.2f" % (k[:44], tot[k][0] / max(tot[k][1], 1)))
PY
