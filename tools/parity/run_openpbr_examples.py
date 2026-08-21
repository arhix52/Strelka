#!/usr/bin/env python3
"""Render the OpenPBR showcase materials MaterialX ships, and record what they look like.

Nothing in this tree rendered them. They are the model's own parameter coverage
-- glass with dispersion, a coated car paint, a pearl's thin film, a lightbulb
at ten thousand nits, ketchup's subsurface -- and every one exercises a lobe the
glTF ladder never reaches, because glTF cannot express it.

There is no reference to compare them against: no second OpenPBR implementation
is available here, and Cycles has no OpenPBR at all. So each material is graded
against open_pbr_default rendered in the same box, and the recorded number is
what that difference was. A number moving is the signal; the numbers themselves
say only that the lobe ran and by how much it changed the image.

Against the default rather than in absolute terms because the blocks are a small
part of a Cornell box. A whole-frame mean is dominated by its walls: brushed
aluminium moves it by 0.1% while changing the blocks completely, which reads as
a material that did nothing. Differencing cancels the box. The scene renders
deterministically -- two runs of one material agree to rel 0.0000 -- so every
digit is the material.

Each material is put on the two blocks of scenes/validation/openpbr through a
<look> that names them, so the box stays a constant; the default's own mean and
p99 are recorded as well, because a change in the walls would otherwise cancel
out of every row at once and read as nothing moving. The scene's _openpbr.json
is deliberately left behind: this measures the document, not the sidecar.

    tools/parity/run_openpbr_examples.py --cli build/Release/StrelkaCLI
    tools/parity/run_openpbr_examples.py --record
    tools/parity/run_openpbr_examples.py --only glass --spp 512

Needs numpy: Blender's bundled interpreter has one.
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCENE_DIR = os.path.join(ROOT, "scenes", "validation", "openpbr")
DEFAULT_EXAMPLES = os.path.join(ROOT, "third_party", "materialx", "resources", "Materials", "Examples", "OpenPbr")
BLOCKS = ("TallBlock", "ShortBlock")

# material -> (rel, ratio) against the open_pbr_default render of the same box.
#
# Against the default rather than in absolute terms, because the blocks are a
# small part of a Cornell box and its walls dominate any whole-frame average: a
# white metal in a white box moves the mean by 0.1% while changing the blocks
# completely. Differencing against a common reference cancels the box, and this
# scene renders deterministically -- two runs of one material agree to rel
# 0.0000 -- so every digit here is the material and none of it is noise.
RECORDED = {
    "open_pbr_aluminum_brushed": (0.036, 1.001),
    "open_pbr_carpaint": (0.028, 0.919),
    "open_pbr_glass": (0.051, 0.985),
    "open_pbr_honey": (0.040, 0.944),
    "open_pbr_ketchup": (0.037, 0.903),
    "open_pbr_lightbulb": (86.008, 260.540),
    "open_pbr_pearl": (0.033, 0.976),
    "open_pbr_soapbubble": (0.054, 1.040),
    "open_pbr_velvet": (0.039, 0.858),
}

# The reference's own numbers, so a change in the *box* is caught too -- it
# would otherwise cancel out of every row at once and read as nothing moving.
RECORDED_REFERENCE = (4.8182, 11.1345)


def wrapper_document(example_path, material_name):
    """The example's own nodes, plus a look that puts them on the blocks.

    Inlined rather than <xi:include>d: an include resolves against a search path
    the renderer sets for its own reasons, and a document that silently includes
    nothing would grade as a material that renders like the box it sits in.
    """
    body = open(example_path).read()
    body = re.sub(r"(?s)^.*?<materialx[^>]*>", "", body)
    body = re.sub(r"(?s)</materialx>\s*$", "", body)
    assigns = "\n".join(
        '    <materialassign name="a_%s" geom="%s" material="%s" />' % (geom, geom, material_name)
        for geom in BLOCKS
    )
    return ('<?xml version="1.0"?>\n<materialx version="1.39" colorspace="lin_rec709">\n'
            + body
            + '  <look name="L_example">\n' + assigns + "\n  </look>\n</materialx>\n")


def material_name_of(example_path):
    match = re.search(r'<surfacematerial\s+name="([^"]+)"', open(example_path).read())
    return match.group(1) if match else None


REFERENCE = "open_pbr_default"


def luminance_stats(path):
    img = load_exr(path)[..., :3]
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    lum = lum[np.isfinite(lum)]
    return float(lum.mean()), float(np.percentile(lum, 99))


def against(path, reference_path):
    """rel / ratio against the reference render, as compare.py defines them."""
    test = load_exr(path)[..., :3]
    ref = load_exr(reference_path)[..., :3]
    if test.shape != ref.shape:
        return None
    ref_mean = float(ref.mean())
    if ref_mean <= 0.0:
        return None
    rel = float(np.sqrt(((test - ref) ** 2).mean())) / float(np.sqrt((ref ** 2).mean()))
    return rel, float(test.mean()) / ref_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build/Release/StrelkaCLI")
    ap.add_argument("--examples", default=DEFAULT_EXAMPLES)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--spp", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--keep", default="", help="directory to leave the rendered EXRs in")
    args = ap.parse_args()

    cli = os.path.abspath(args.cli)
    if not os.access(cli, os.X_OK):
        sys.exit("not executable: %s" % cli)
    cli_dir = os.path.dirname(cli)

    documents = sorted(f for f in os.listdir(args.examples) if f.endswith(".mtlx"))
    if args.only:
        documents = [d for d in documents if any(f in d for f in args.only)]
    # The reference is rendered whatever the filter says: every other row is
    # expressed against it, and a --only run that dropped it would have nothing
    # to be expressed against.
    reference_doc = REFERENCE + ".mtlx"
    if reference_doc not in documents and os.path.exists(os.path.join(args.examples, reference_doc)):
        documents.insert(0, reference_doc)

    tmp = tempfile.mkdtemp(prefix="strelka_openpbr_examples_")
    rendered, rows, failures = [], [], []

    for doc in documents:
        label = os.path.splitext(doc)[0]
        example = os.path.join(args.examples, doc)
        material = material_name_of(example)
        if not material:
            failures.append((label, "no <surfacematerial> in the document"))
            continue

        work = os.path.join(tmp, label)
        os.makedirs(work, exist_ok=True)
        # Only the geometry and its lights: the scene's _openpbr.json would
        # author the walls as OpenPBR too, which is a second variable.
        for name in ("openpbr_cornell.glb", "openpbr_cornell_light.json"):
            shutil.copy(os.path.join(SCENE_DIR, name), work)
        open(os.path.join(work, "openpbr_cornell.mtlx"), "w").write(wrapper_document(example, material))

        out_exr = os.path.join(args.keep or work, label + ".exr")
        if args.keep:
            os.makedirs(args.keep, exist_ok=True)
        text = open(os.path.join(SCENE_DIR, "openpbr_cornell.toml")).read()
        text = re.sub(r'(?m)^(\s*path\s*=\s*)"[^"]*\.glb"', r'\1"%s"' % os.path.join(work, "openpbr_cornell.glb"),
                      text, count=1)
        text = re.sub(r'(?m)^(\s*path\s*=\s*)"[^"]*\.(exr|png)"', r'\1"%s"' % out_exr, text, count=1)
        if args.spp:
            text = re.sub(r"(?m)^(\s*spp\s*=\s*)\d+", r"\g<1>%d" % args.spp, text, count=1)
        toml_path = os.path.join(work, "run.toml")
        open(toml_path, "w").write(text)

        proc = subprocess.run([cli, "-c", toml_path], cwd=cli_dir, capture_output=True, text=True,
                              timeout=args.timeout)
        if proc.returncode != 0 or not os.path.exists(out_exr):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
            failures.append((label, "exit %d: %s" % (proc.returncode, tail[0][:110])))
            continue
        # A document that bound nothing renders the box and grades as a
        # material, so the binding is checked rather than assumed.
        if "binding(s) applied" in proc.stdout and re.search(r"0 binding\(s\) applied", proc.stdout):
            failures.append((label, "the look bound nothing"))
            continue
        rendered.append((label, out_exr))

    reference_path = next((path for label, path in rendered if label == REFERENCE), None)
    reference = luminance_stats(reference_path) if reference_path else None
    for label, path in rendered:
        if label == REFERENCE:
            continue
        if reference_path is None:
            failures.append((label, "no %s render to express this against" % REFERENCE))
            continue
        measured = against(path, reference_path)
        if measured is None:
            failures.append((label, "shape or reference mismatch"))
            continue
        rows.append((label, measured))

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)

    if args.record:
        print("RECORDED = {")
        for label, m in rows:
            print('    "%s": (%.3f, %.3f),' % (label, m[0], m[1]))
        print("}")
        if reference:
            print("RECORDED_REFERENCE = (%.4f, %.4f)" % reference)
        for label, why in failures:
            sys.stderr.write("failed: %-30s %s\n" % (label, why))
        return 1 if failures else 0

    if reference:
        print("%s: mean %.4f, p99 %.4f%s" % (
            REFERENCE, reference[0], reference[1],
            "   recorded %.4f / %.4f" % RECORDED_REFERENCE if RECORDED_REFERENCE else ""))
    print("%-30s %-22s %s" % ("material vs default", "rel / ratio", "recorded"))
    print("-" * 74)
    for label, m in rows:
        exp = RECORDED.get(label)
        print("%-30s %-22s %s" % (label, "%.3f / %.3f" % (m[0], m[1]),
                                  "%.3f / %.3f" % exp if exp else ""))

    if failures:
        print("\nfailed (%d):" % len(failures))
        for label, why in failures:
            print("  %-30s %s" % (label, why))

    print("\n%d rendered, %d failed" % (len(rows), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
