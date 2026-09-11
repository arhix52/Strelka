#!/usr/bin/env python3
"""Print AGX register/spill stats for Metal kernels without Xcode or a GPU replay.

Translates AIR -> native AGX offline with metal-nt, then reads the counters the
compiler leaves in __GPU_METADATA of the resulting MH_GPU_EXECUTE. Same numbers
Xcode's shader profiler shows, minus the capture and the replay.

  tools/metal_regs.py build/Release/metal/shaders/wavefront.metallib wavefrontShade \
      -c 0=1 -c 1=1 -c 11=1 -c 17=1 -c 23=1

Constants are function-constant index (or name) = value, bool unless -t is given.
"""
import argparse, glob, json, os, struct, subprocess, tempfile, zlib

# ponytail: slot numbers are AGCCodeGenerator.ShaderInfo field order, read off the
# flatbuffer schema embedded in libapplegpu-nt.dylib. A toolchain that renumbers them
# makes the output obviously wrong (registers in the thousands), not quietly off.
SHADER_INFO = {
    0: ("regs", 4),          # temporary_register_count
    1: ("shared", 4),        # shared_register_count (uniform registers)
    14: ("spill", 4),        # spill_buffer_bytes
    31: ("ti_spill", 4),     # thread_invariant_spill_buffer_bytes
    32: ("complexity", 1),
    36: ("tls_spill", 4),    # per_thread_local_memory_spill_bytes
    38: ("ipr", 4),          # largest_ipr_bytes
}


def trace_inputs(bundle, kernel):
    """Metallib + the exact function constants a .gputrace recorded for `kernel`.

    Xcode stores the libraries as plain MetalLib files and the pipeline stream
    zlib-compressed in `store0`, where each constant is six little-endian u64s:
    index, 0, MTLDataType, 0, size, value.
    """
    libs = [(os.path.getsize(f), f) for f in glob.glob(os.path.join(bundle, "*"))
            if os.path.isfile(f) and open(f, "rb").read(4) == b"MTLB"]
    lib = max(libs)[1]

    raw = open(os.path.join(bundle, "store0"), "rb").read()
    store, pos = bytearray(), 0
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
        store += chunk
        pos += max(len(raw) - pos - len(z.unused_data), 1)

    best = {}
    name = kernel.encode()
    at = store.find(name)
    while at >= 0:
        v = [struct.unpack_from("<Q", store, o)[0] for o in range(max(at - 1400, 0), at - 8, 8)]
        found = {}
        for k in range(len(v) - 5):
            idx, z1, ty, z2, size, val = v[k:k + 6]
            if z1 == 0 and z2 == 0 and ty in (33, 53) and size in (1, 4) and idx < 64:
                found[idx] = ("ConstantBool" if ty == 53 else "ConstantUInt", val)
        # A captured pipeline stores the function name once in its function
        # descriptor and once again after the specialized pipeline record. The
        # constants immediately preceding the latter are the ones used by that
        # PSO. On ties keep the later occurrence; keeping the first silently
        # associates adjacent shade-bucket constants with one another.
        if len(found) >= len(best):
            best = found
        at = store.find(name, at + 1)
    return lib, best


def sections(blob):
    ncmds = struct.unpack_from("<I", blob, 16)[0]
    out, off = {}, 32
    for _ in range(ncmds):
        cmd, csz = struct.unpack_from("<II", blob, off)
        if cmd == 0x19:  # LC_SEGMENT_64
            nsects = struct.unpack_from("<I", blob, off + 64)[0]
            so = off + 72
            for _ in range(nsects):
                sect = blob[so:so + 16].rstrip(b"\0").decode()
                seg = blob[so + 16:so + 32].rstrip(b"\0").decode()
                _, size, offs = struct.unpack_from("<QQI", blob, so + 32)
                out[f"{seg},{sect}"] = (offs, size)
                so += 80
        off += csz
    return out


def read_stats(md):
    u32 = lambda o: struct.unpack_from("<I", md, o)[0]
    i32 = lambda o: struct.unpack_from("<i", md, o)[0]
    u16 = lambda o: struct.unpack_from("<H", md, o)[0]

    def slot(table, i):
        vt = table - i32(table)
        return u16(vt + 4 + 2 * i) if 4 + 2 * i < u16(vt) else 0

    root = u32(0)
    common = slot(root, 0)                       # CompileReply.common_info
    info = root + common + u32(root + common)    # -> ShaderInfo
    out = {}
    for i, (name, width) in SHADER_INFO.items():
        off = slot(info, i)
        out[name] = 0 if not off else (md[info + off] if width == 1 else u32(info + off))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metallib", help=".metallib, or a .gputrace bundle with --trace")
    ap.add_argument("kernels", nargs="+")
    ap.add_argument("--trace", action="store_true",
                    help="read the metallib and the recorded function constants out of a .gputrace")
    ap.add_argument("--lib", help="with --trace: use this metallib instead of the bundle's")
    ap.add_argument("--set", action="append", default=[], metavar="ID=VAL",
                    help="with --trace: override one recorded constant, keeping its recorded type")
    ap.add_argument("-c", "--constant", action="append", default=[], metavar="ID=VAL",
                    help="add/override a function constant (also overrides the recorded value with --trace)")
    ap.add_argument("-t", "--type", default="ConstantBool", help="constant type (default ConstantBool)")
    ap.add_argument("-a", "--arch", default="applegpu_g16s",
                    help="M1 g13g/g13s, M2 g14g/g14s, M3 g15g/g15s, M4 g16g/g16s (default)")
    args = ap.parse_args()

    tc = os.path.dirname(subprocess.check_output(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], text=True).strip())
    keep = tempfile.TemporaryDirectory(prefix="metal_regs_")
    tmp = keep.name

    cli = []
    for c in args.constant:
        k, _, v = c.partition("=")
        cli.append({"id_type": "FunctionConstantIndex" if k.isdigit() else "FunctionConstantName",
                    "id": {"data": int(k)} if k.isdigit() else k,
                    "value_type": args.type,
                    "value": {"data": v not in ("0", "false") if args.type == "ConstantBool"
                              else json.loads(v)}})

    source = args.metallib
    stripped = None
    print(f"{args.arch}  {os.path.basename(args.metallib)}")
    for kernel in args.kernels:
        cv = cli
        if args.trace:
            source, recorded = trace_inputs(args.metallib, kernel)
            for o in args.set:
                k, _, v = o.partition("=")
                recorded[int(k)] = (recorded[int(k)][0], int(v))
            source = args.lib or source
            stripped = None
            override_indices = {c["id"]["data"] for c in cli if c["id_type"] == "FunctionConstantIndex"}
            cv = [{"id_type": "FunctionConstantIndex", "id": {"data": i},
                   "value_type": t, "value": {"data": bool(val) if t == "ConstantBool" else val}}
                  for i, (t, val) in sorted(recorded.items()) if i not in override_indices]
            cv.extend(cli)
        if stripped is None:
            # metal-nt rejects the private metadata -frecord-sources embeds, so strip it first.
            stripped = os.path.join(tmp, "stripped.metallib")
            subprocess.run([f"{tc}/metal-strip", "-o", stripped, source], check=True)
        lib = stripped
        ref = "fnd:fn"
        doc = {"function_descriptors": {"library_function_descriptors": [{"label": "fn", "name": kernel}]}}
        if cv:
            doc["function_descriptors"]["specialized_function_descriptors"] = [
                {"label": "sp", "function_descriptor": ref, "constant_values": cv}]
            ref = "fnd:sp"
        doc["pipeline_descriptors"] = {"compute_pipeline_descriptors": [{"compute_function_descriptor": ref}]}
        script = os.path.join(tmp, "pipelines.mtl4-json")
        with open(script, "w") as f:
            json.dump(doc, f)

        out = os.path.join(tmp, "gpu.bin")
        subprocess.run([f"{tc}/metal-nt", "-arch", args.arch, "-platform_version", "macos", "26.0", "26.0",
                        "-N", script, "-o", out, lib], check=True)
        blob = open(out, "rb").read()
        off, _ = sections(blob)["__TEXT,__compute"]
        obj = blob[off:]
        secs = sections(obj)
        md_off, md_size = secs["__GPU_METADATA,__compute"]
        s = read_stats(obj[md_off:md_off + md_size])
        print(f"  {kernel}: regs={s['regs']} shared={s['shared']} spill={s['spill']}B "
              f"ti_spill={s['ti_spill']}B tls_spill={s['tls_spill']}B ipr={s['ipr']}B "
              f"complexity={s['complexity']} code={secs['__TEXT,__text'][1]}B")


if __name__ == "__main__":
    main()
