#!/usr/bin/env python3
"""Per-pipeline compiler statistics out of a .gputrace, with no replay.

Xcode's capture already stores what the GPU driver's compiler reported for every
pipeline it built: register counts, spilled bytes, an instruction mix and the
optimizer's remarks. It sits in `store0` as zlib-compressed NSKeyedArchiver
plists. Replay only adds the *timing* — everything below is static.

  tools/metal_trace_stats.py foo.gputrace
  tools/metal_trace_stats.py foo.gputrace --remarks wavefrontShadeBase
"""
import argparse, os, plistlib, re, zlib
from plistlib import UID

COLUMNS = [("Temporary register count", "regs"), ("Uniform register count", "uniform"),
           ("Spilled bytes", "spill"), ("Thread invariant spilled bytes", "ti_spill"),
           ("Threadgroup memory", "tgmem"), ("Instruction count", "instr"),
           ("ALU instruction count", "alu"), ("Branch instruction count", "branch"),
           ("Device load instruction count", "ld"), ("Device store instruction count", "st"),
           ("Texture reads instruction count", "tex")]


def store(bundle):
    raw = open(os.path.join(bundle, "store0"), "rb").read()
    out, pos = bytearray(), 0
    while pos < len(raw):
        z = zlib.decompressobj()
        try:
            chunk = z.decompress(raw[pos:])
        except zlib.error:
            pos += 1
            continue
        if not chunk and z.unused_data == raw[pos:]:
            pos += 1
            continue
        out += chunk
        pos += max(len(raw) - pos - len(z.unused_data), 1)
    return bytes(out)


def unarchive(plist):
    objs = plist["$objects"]

    def resolve(o):
        if isinstance(o, UID):
            return resolve(objs[o.data])
        if isinstance(o, dict):
            if "NS.keys" in o:
                return {resolve(k): resolve(v) for k, v in zip(o["NS.keys"], o["NS.objects"])}
            if "NS.objects" in o:
                return [resolve(x) for x in o["NS.objects"]]
            return {k: resolve(v) for k, v in o.items() if k != "$class"}
        return o

    return resolve(plist["$top"])["root"]


def plist_end(buf):
    """Length of the binary plist at the start of `buf`.

    The stream stores no length, so find the 32-byte trailer by its own
    consistency: the offset table has to end exactly where the trailer begins.
    """
    for end in range(len(buf), 40, -1):
        off_size, ref_size = buf[end - 26], buf[end - 25]
        if not (1 <= off_size <= 8 and 1 <= ref_size <= 8):
            continue
        count = int.from_bytes(buf[end - 24:end - 16], "big")
        top = int.from_bytes(buf[end - 16:end - 8], "big")
        table = int.from_bytes(buf[end - 8:end], "big")
        if top < count and table + count * off_size == end - 32:
            return end
    return 0


def sources(bundle):
    """Every byte range in the bundle that can hold a stats archive.

    Most live zlib-packed in `store0`; a pipeline compiled outside the captured
    frame gets its own plain file next to it.
    """
    yield store(bundle)
    for name in sorted(os.listdir(bundle)):
        if name.startswith(("MTLAcceleration", "MTLBuffer", "MTLTexture")) or name == "store0":
            continue
        path = os.path.join(bundle, name)
        if os.path.isfile(path):
            yield open(path, "rb").read()


def blocks(bundle):
    for s in sources(bundle):
        yield from _blocks(s)


def _blocks(s):
    starts = [m.start() for m in re.finditer(b"bplist00", s)]
    for a, b in zip(starts, starts[1:] + [len(s)]):
        end = plist_end(s[a:b])
        if not end:
            continue
        try:
            plist = plistlib.loads(s[a:a + end])
        except Exception:
            continue
        if not isinstance(plist, dict) or "$objects" not in plist:
            continue
        stats = unarchive(plist)
        if isinstance(stats, dict):
            # "Telemetry Statistics" is archived empty; the dictionary naming the
            # function is left unreferenced in the object table, so go find it.
            for o in plist["$objects"]:
                if isinstance(o, dict) and "NS.keys" in o:
                    keys = [plist["$objects"][u.data] for u in o["NS.keys"]]
                    if "Function Name" in keys:
                        vals = [plist["$objects"][u.data] for u in o["NS.objects"]]
                        stats["function"] = vals[keys.index("Function Name")]
        yield stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gputrace")
    ap.add_argument("--remarks", metavar="FUNCTION", help="print the optimizer remarks for one function")
    args = ap.parse_args()

    rows = []
    for stats in blocks(args.gputrace):
        if not isinstance(stats, dict) or "Instruction count" not in stats:
            continue
        name = stats.get("function", "?")
        if args.remarks:
            if name == args.remarks:
                print(stats.get("Remarks", "(no remarks)"))
                return
            continue
        rows.append((name, stats))

    if args.remarks:
        print(f"no pipeline named {args.remarks}")
        return
    width = max((len(n) for n, _ in rows), default=8)
    print(f"{'function':<{width}}  " + "  ".join(f"{short:>8}" for _, short in COLUMNS))
    for name, s in sorted(rows, key=lambda r: -r[1]["Instruction count"]):
        print(f"{name:<{width}}  " + "  ".join(f"{s.get(key, 0):>8}" for key, _ in COLUMNS))


if __name__ == "__main__":
    main()
