#!/usr/bin/env bash
# Sample Apple's GPU hardware counters against a running renderer.
#
#   tools/gpu_counters.sh <pid|--idle> [seconds]
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
# And they are device-wide. Attaching to a pid does not isolate them -- an idle
# desktop reads ALU Utilization 56% and Fragment Occupancy 48%, which is the
# compositor. Only the difference against an idle control means anything, so
# take one with `--idle` and subtract.
#
# The trace and its XML are deleted on the way out. A two-second recording is
# ~70 MB and its export ~600 MB; leaving those around fills a disk in a few
# runs, which is how this script came to exist.
set -u
TARGET="${1:?usage: gpu_counters.sh <pid|--idle> [seconds]}"
SECS="${2:-2}"
TRACE="$(mktemp -d)/counters.trace"
XML="${TRACE%.trace}.xml"
trap 'rm -rf "$(dirname "$TRACE")"' EXIT

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
