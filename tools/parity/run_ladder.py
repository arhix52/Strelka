#!/usr/bin/env python3
"""Grade a backend against the Cycles feature-test ladder. No Blender required.

    tools/parity/run_ladder.py --cli build/Release/StrelkaCLI
    tools/parity/run_ladder.py --only 00 --only 19
    tools/parity/run_ladder.py --skip-render          # re-grade existing EXRs

What it does, per scene in the ladder:

  1. Rewrites the scene's .toml. The shipped ones carry absolute macOS paths
     (`/Users/ikryukov/Strelka/scenes/feature_tests/...`) baked in by the Blender
     exporter, so they cannot be used as-is anywhere else. The rewrite goes to a
     temp directory; the assets are never modified.
  2. Runs StrelkaCLI on it.
  3. Compares the result against two targets:
       * `<scene>_cycles.exr`  -- the reference. tools/feature_tests/README.md
         records the rel/ratio each row is expected to reach, and that table is
         the pass bar.
       * `<scene>_strelka.exr` -- what the Metal backend produced. Two backends
         of one renderer should agree far more tightly than either agrees with
         Cycles, so a large number here is a backend-difference signal that the
         Cycles column can hide.

Read 00_calibration first. It is a grey sphere under the key light and nothing
else; if its ratio is not ~1.0 then light units or exposure disagree and every
other row is re-measuring that same offset rather than the feature it names.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_tests"))

import numpy as np  # noqa: E402
from exr_io import load_exr  # noqa: E402

DEFAULT_ASSETS = os.path.expanduser("~/strelka_assets/feature_tests")

# From tools/feature_tests/README.md -- what each row measured when it was last
# recorded, against Cycles. Not a tolerance to pass; a number to move toward.
EXPECTED = {
    "00_calibration": (0.021, 1.010), "01_srgb_texture": (0.026, 1.004),
    "02_basecolor": (0.022, 1.003), "03_roughness": (0.024, 1.015),
    "04_metal": (0.055, 0.986), "05_anisotropy": (0.061, 1.007),
    "06_normalmap": (0.071, 1.061), "07_alpha_clip": (0.025, 1.018),
    "08_alpha_blend": (0.024, 1.016), "09_glass_ior": (0.059, 1.012),
    "10_glass_absorption": (0.038, 1.012), "11_emission": (0.017, 1.004),
    "12_lights_punctual": (0.031, 1.002), "13_uv2_vcol": (0.024, 1.010),
    "14_sheen": (0.053, 1.016), "15_clearcoat": (0.037, 1.002),
    "16_iridescence": (0.023, 1.010), "17_coated_glass": (0.067, 0.994),
    "18_bounded_volume": (0.027, 1.000), "19_env_and_light": (0.026, 1.000),
    "20_mirror_and_floor": (0.034, 1.038), "21_specular_color": (0.030, 1.018),
    "22_thin_walled": (0.094, 0.966), "23_diffuse_transmission": (0.012, 1.000),
    "24_orthographic": (0.024, 1.009), "25_subsurface": (0.056, 1.009),
    "26_dof": (0.024, 1.015), "27_ies": (0.028, 1.021), "28_hair": (0.033, 1.012),
}


def rewrite_toml(src_toml, scene_dir, name, out_exr, spp_override):
    """Point [scene].path and [output].path at this machine, in a temp copy."""
    text = open(src_toml).read()
    gltf = None
    for candidate in (name + ".gltf", name + ".glb"):
        if os.path.exists(os.path.join(scene_dir, candidate)):
            gltf = os.path.join(scene_dir, candidate)
            break
    if gltf is None:
        raise FileNotFoundError("no .gltf/.glb beside " + src_toml)

    text = re.sub(r'(?m)^(\s*path\s*=\s*)"[^"]*\.(gltf|glb)"', r'\1"%s"' % gltf, text, count=1)
    text = re.sub(r'(?m)^(\s*path\s*=\s*)"[^"]*\.exr"', r'\1"%s"' % out_exr, text, count=1)
    if spp_override:
        text = re.sub(r"(?m)^(\s*spp\s*=\s*)\d+", r"\g<1>%d" % spp_override, text, count=1)
    return text


def stats(test, ref):
    """rel / rmse / ratio, defined as compare.py defines them."""
    if test.shape != ref.shape:
        return None
    ref_mean = float(ref.mean())
    if ref_mean <= 0.0:
        return None
    return (
        float(np.abs(test - ref).mean()) / ref_mean,
        float(np.sqrt(((test - ref) ** 2).mean())),
        float(test.mean()) / ref_mean,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build/Release/StrelkaCLI")
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--suffix", default="optix", help="written as <scene>_<suffix>.exr")
    ap.add_argument("--only", action="append", default=[], help="substring filter, repeatable")
    ap.add_argument("--spp", type=int, default=0, help="override spp (faster smoke runs)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    cli = os.path.abspath(args.cli)
    if not args.skip_render and not os.access(cli, os.X_OK):
        sys.exit("not executable: %s" % cli)
    # StrelkaCLI resolves optix/ and metal/shaders/ next to the executable, but
    # run from its own directory anyway so any CWD-relative fallback also works.
    cli_dir = os.path.dirname(cli)

    scenes = sorted(
        d for d in os.listdir(args.assets)
        if os.path.isdir(os.path.join(args.assets, d)) and not d.startswith("_")
    )
    if args.only:
        scenes = [s for s in scenes if any(f in s for f in args.only)]

    tmp = tempfile.mkdtemp(prefix="strelka_parity_")
    rows, failures = [], []

    for name in scenes:
        scene_dir = os.path.join(args.assets, name)
        src_toml = os.path.join(scene_dir, name + ".toml")
        if not os.path.exists(src_toml):
            continue
        out_exr = os.path.join(scene_dir, "%s_%s.exr" % (name, args.suffix))

        if not args.skip_render:
            toml_path = os.path.join(tmp, name + ".toml")
            try:
                open(toml_path, "w").write(rewrite_toml(src_toml, scene_dir, name, out_exr, args.spp))
            except FileNotFoundError as exc:
                failures.append((name, str(exc)))
                continue
            proc = subprocess.run([cli, "-c", toml_path], cwd=cli_dir,
                                  capture_output=True, text=True, timeout=args.timeout)
            if proc.returncode != 0 or not os.path.exists(out_exr):
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
                failures.append((name, "exit %d: %s" % (proc.returncode, tail[0][:110])))
                continue

        ref_path = os.path.join(scene_dir, name + "_cycles.exr")
        metal_path = os.path.join(scene_dir, name + "_strelka.exr")
        try:
            test = load_exr(out_exr)
        except Exception as exc:
            failures.append((name, "unreadable output: %s" % exc))
            continue

        vs_cycles = stats(test, load_exr(ref_path)) if os.path.exists(ref_path) else None
        vs_metal = stats(test, load_exr(metal_path)) if os.path.exists(metal_path) else None
        rows.append((name, vs_cycles, vs_metal))

    shutil.rmtree(tmp, ignore_errors=True)

    print("%-24s %-21s %-16s %s" % ("scene", "vs Cycles rel/ratio", "vs Metal rel", "recorded"))
    print("-" * 78)
    for name, cyc, met in rows:
        cyc_s = "%.3f / %.3f" % (cyc[0], cyc[2]) if cyc else "n/a"
        met_s = "%.3f" % met[0] if met else "n/a"
        exp = EXPECTED.get(name)
        exp_s = "%.3f / %.3f" % exp if exp else ""
        print("%-24s %-21s %-16s %s" % (name, cyc_s, met_s, exp_s))

    if failures:
        print("\nfailed to render or read (%d):" % len(failures))
        for name, why in failures:
            print("  %-24s %s" % (name, why))

    print("\n%d graded, %d failed" % (len(rows), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
