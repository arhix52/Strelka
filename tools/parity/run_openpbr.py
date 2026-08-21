#!/usr/bin/env python3
"""Grade OpenPBR against standard_pbr on the feature-test ladder's own scenes.

Why this is not a row in run_ladder.py: Cycles is the wrong oracle for OpenPBR.
Blender has no OpenPBR, so grading one against it compares two material models
and charges the difference to us -- and this tree already carries a measured
+3.3% offset against Cycles on 00_calibration alone, which is larger than
compare.py's tolerance before a single OpenPBR lobe is involved.

So this harness has no external reference at all. It renders each ladder scene
twice from one asset -- once through the glTF model it was authored for, once
with `[render] material_model = "openpbr"`, which converts the glTF *factors*
(not the maps) into OpenPBRParams -- and reports how far apart the two models
land. That number is the measurement:

  * Where the two specifications agree -- a Lambert base, a smooth metal, a
    single-specular dielectric -- they must produce the same image, and a row
    that drifts says the bridge broke.
  * Where they legitimately disagree -- coat darkening, the fuzz fit, the
    subsurface entry -- the number is not a defect. It is recorded so that a
    *change* in it is visible, which is the only thing a model with no
    independent implementation can be held to.

RECORDED below is what each row measured when it was last taken, on Metal.
Not a tolerance: a number to notice moving.

    tools/parity/run_openpbr.py --cli build/Release/StrelkaCLI
    tools/parity/run_openpbr.py --only 14_sheen --spp 64
    tools/parity/run_openpbr.py --record        # print a fresh table to paste

Compare the *ratio* column at any sample count -- it is a brightness offset and
barely moves with noise. `rel` is a per-pixel average and rises as the sample
count falls, so it is only comparable against RECORDED when run without --spp,
at the sample count each scene's own .toml asks for, which is how the table was
taken. Needs numpy: Blender's bundled interpreter has one
(/Applications/Blender.app/Contents/Resources/*/python/bin/python3*).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_tests"))

from exr_io import load_exr  # noqa: E402
from run_ladder import DEFAULT_ASSETS, rewrite_toml, stats  # noqa: E402

# scene -> (rel, ratio) of the OpenPBR render against the glTF one, on Metal.
#
# Read the ratio column first. Most rows sit at 0.96-0.98: OpenPBR is a few per
# cent darker than standard_pbr across the board, which is the two models'
# diffuse and specular layers differing and not a defect.
#
# Four rows are far off that, and all four are the *conversion's* limitation
# rather than the model's: `material_model = "openpbr"` translates the glTF
# factors and not the maps, so a textured material renders with its factor alone
# and a white factor over a dark map reads much brighter. 01_srgb_texture,
# 13_uv2_vcol, 20_mirror_and_floor and 07_alpha_clip are the ladder's textured
# scenes, and they are the four above 1.2. They stay in the table because the
# number still has to be stable; they are not evidence about the BSDF.
#
# 25_subsurface and 18_bounded_volume carry no maps at all. Their offsets are
# real model differences -- the subsurface entry and the volume boundary -- and
# those are the rows to watch when the OpenPBR side changes.
RECORDED = {
    "00_calibration": (0.025, 0.975), "01_srgb_texture": (0.681, 1.677),
    "02_basecolor": (0.022, 0.980), "03_roughness": (0.030, 0.972),
    "04_metal": (0.046, 0.982), "05_anisotropy": (0.073, 0.984),
    "06_normalmap": (0.051, 0.964), "07_alpha_clip": (0.245, 1.221),
    "08_alpha_blend": (0.043, 0.959), "09_glass_ior": (0.048, 0.975),
    "10_glass_absorption": (0.035, 0.976), "11_emission": (0.017, 0.992),
    "12_lights_punctual": (0.033, 0.968), "13_uv2_vcol": (0.308, 1.299),
    "14_sheen": (0.056, 0.972), "15_clearcoat": (0.109, 0.892),
    "16_iridescence": (0.031, 0.978), "17_coated_glass": (0.114, 0.903),
    "18_bounded_volume": (0.279, 0.905), "19_env_and_light": (0.036, 0.964),
    "20_mirror_and_floor": (0.536, 1.529), "21_specular_color": (0.038, 0.965),
    "22_thin_walled": (0.078, 0.996), "23_diffuse_transmission": (0.070, 0.935),
    "24_orthographic": (0.027, 0.973), "25_subsurface": (0.203, 1.182),
    "26_dof": (0.030, 0.972), "27_ies": (0.047, 0.969), "28_hair": (0.029, 0.975),
}


def with_material_model(text, model):
    """Set [render] material_model, adding the key if the config omits it."""
    if re.search(r"(?m)^\s*material_model\s*=", text):
        return re.sub(r'(?m)^(\s*material_model\s*=\s*)"[^"]*"', r'\1"%s"' % model, text, count=1)
    return re.sub(r"(?m)^(\[render\]\s*)$", r'\1\nmaterial_model = "%s"' % model, text, count=1)


def render(cli, cli_dir, toml_text, toml_path, out_exr, timeout):
    open(toml_path, "w").write(toml_text)
    proc = subprocess.run([cli, "-c", toml_path], cwd=cli_dir, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(out_exr):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return "exit %d: %s" % (proc.returncode, tail[0][:110])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build/Release/StrelkaCLI")
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--only", action="append", default=[], help="substring filter, repeatable")
    ap.add_argument("--spp", type=int, default=0, help="override spp (faster smoke runs)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--record", action="store_true", help="print a RECORDED table instead of grading")
    args = ap.parse_args()

    cli = os.path.abspath(args.cli)
    if not os.access(cli, os.X_OK):
        sys.exit("not executable: %s" % cli)
    cli_dir = os.path.dirname(cli)

    scenes = sorted(
        d for d in os.listdir(args.assets)
        if os.path.isdir(os.path.join(args.assets, d)) and not d.startswith("_")
    )
    if args.only:
        scenes = [s for s in scenes if any(f in s for f in args.only)]

    tmp = tempfile.mkdtemp(prefix="strelka_openpbr_")
    rows, failures = [], []

    for name in scenes:
        scene_dir = os.path.join(args.assets, name)
        src_toml = os.path.join(scene_dir, name + ".toml")
        if not os.path.exists(src_toml):
            continue

        pbr_exr = os.path.join(scene_dir, "%s_model_pbr.exr" % name)
        opbr_exr = os.path.join(scene_dir, "%s_model_openpbr.exr" % name)
        try:
            base = rewrite_toml(src_toml, scene_dir, name, pbr_exr, args.spp)
        except FileNotFoundError as exc:
            failures.append((name, str(exc)))
            continue

        why = render(cli, cli_dir, with_material_model(base, "standard_pbr"),
                     os.path.join(tmp, name + "_pbr.toml"), pbr_exr, args.timeout)
        if why:
            failures.append((name, "standard_pbr " + why))
            continue

        opbr_text = with_material_model(base.replace(pbr_exr, opbr_exr), "openpbr")
        why = render(cli, cli_dir, opbr_text, os.path.join(tmp, name + "_opbr.toml"), opbr_exr, args.timeout)
        if why:
            failures.append((name, "openpbr " + why))
            continue

        try:
            measured = stats(load_exr(opbr_exr), load_exr(pbr_exr))
        except Exception as exc:
            failures.append((name, "unreadable output: %s" % exc))
            continue
        if measured is None:
            failures.append((name, "shape or reference mismatch"))
            continue
        rows.append((name, measured))

    shutil.rmtree(tmp, ignore_errors=True)

    if args.record:
        print("RECORDED = {")
        for name, m in rows:
            print('    "%s": (%.3f, %.3f),' % (name, m[0], m[2]))
        print("}")
        # To stderr, so the table on stdout stays paste-ready -- and printed at
        # all, because a recording run that silently drops a row produces a
        # table that looks complete and is not.
        for name, why in failures:
            sys.stderr.write("failed: %-24s %s\n" % (name, why))
        return 1 if failures else 0

    print("%-24s %-22s %s" % ("scene", "openpbr vs gltf rel/ratio", "recorded"))
    print("-" * 66)
    for name, m in rows:
        exp = RECORDED.get(name)
        print("%-24s %-22s %s" % (name, "%.3f / %.3f" % (m[0], m[2]),
                                  "%.3f / %.3f" % exp if exp else ""))

    if failures:
        print("\nfailed to render or read (%d):" % len(failures))
        for name, why in failures:
            print("  %-24s %s" % (name, why))

    print("\n%d graded, %d failed" % (len(rows), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
