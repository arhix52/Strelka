#!/usr/bin/python3
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "feature_tests"))
from exr_io import load_exr  # noqa: E402


MANIFEST_SHA256 = "cec235160109b775b4492b3a62879e22ff75b064d2ae18777e0ed4181870309f"


def restir(candidates, temporal, spatial, neighbors=0):
    return ["--restir-di", "--restir-candidates", str(candidates),
            f"--restir-temporal={'true' if temporal else 'false'}",
            f"--restir-spatial={'true' if spatial else 'false'}", "--restir-neighbors", str(neighbors),
            "--restir-bias-correction", "basic"]


VARIANTS = {"initial": restir(1, False, False), "temporal": restir(1, True, False),
            "spatial_s1": restir(1, False, True, 1), "temporal_s1": restir(1, True, True, 1),
            "temporal_s2": restir(1, True, True, 2)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"render failed: {' '.join(map(str, command))}\n{completed.stdout[-2000:]}")
    return completed.stdout


def sequence_command(binary, scene, output, prefix, width, height, spp, lights, sequence, variant=(), audit=False):
    command = [str(binary), str(scene), "-o", str(output), "--width", str(width), "--height", str(height),
               "--spp", str(spp), "--depth", "4", "--tonemap", "none", "--clamp", "0",
               "--audit-frames", "16", "--audit-moving-lights", str(lights),
               "--audit-motion-sequence", str(sequence)]
    if prefix:
        command += ["--audit-frame-prefix", str(prefix)]
    if spp > 1:
        command.append("--audit-freeze")
    if audit:
        command.append("--audit-render-work")
    return command + list(variant)


def load_frames(prefix):
    return [load_exr(Path(f"{prefix}-{frame:02}.exr"))[..., :3].astype(np.float64) for frame in range(8, 24)]


def luminance(image):
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def frame_metrics(images, references):
    rmses = []
    ratios = []
    for image, reference in zip(images, references):
        rmses.append(float(np.sqrt(np.mean((image - reference) ** 2) / np.mean(reference ** 2))))
        ratios.append(float(np.mean(luminance(image)) / np.mean(luminance(reference))))
    stacked_error = np.stack(images) - np.stack(references)
    stacked_reference = np.stack(references)
    accumulated = np.mean(images, axis=0)
    accumulated_reference = np.mean(references, axis=0)
    return {"single_frame_rmse": rmses, "mean_single_frame_rmse": float(np.mean(rmses)),
            "paired_sequence_rmse": float(np.sqrt(np.mean(stacked_error ** 2) / np.mean(stacked_reference ** 2))),
            "accumulated_sequence_rmse": float(np.sqrt(np.mean((accumulated - accumulated_reference) ** 2) /
                                                          np.mean(accumulated_reference ** 2))),
            "mean_ratio": float(np.mean(ratios)), "mean_ratio_spread": float(max(ratios) - min(ratios))}


def bilinear(image, x, y):
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    valid = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < w) & (y0 + 1 < h)
    x0 = np.clip(x0, 0, w - 2)
    y0 = np.clip(y0, 0, h - 2)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    value = ((1 - fx) * (1 - fy) * image[y0, x0] + fx * (1 - fy) * image[y0, x0 + 1] +
             (1 - fx) * fy * image[y0 + 1, x0] + fx * fy * image[y0 + 1, x0 + 1])
    return value, valid


def temporal_error(images, references, motion_images):
    errors = [image - reference for image, reference in zip(images, references)]
    values = []
    h, w = images[0].shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    for frame in range(1, len(images)):
        motion = (motion_images[frame][..., :2] - 0.5) / 0.05
        previous, valid = bilinear(errors[frame - 1], x + motion[..., 0], y + motion[..., 1])
        delta = errors[frame] - previous
        values.append(float(np.sqrt(np.mean(delta[valid] ** 2) / np.mean(references[frame][valid] ** 2))))
    return float(np.mean(values))


def audit_json(text):
    records = [json.loads(line) for line in text.splitlines() if line.startswith('{"frames"')]
    return records[-1]


def gpu_times(text):
    return [float(value) for value in re.findall(r"STRELKA_AUDIT_FRAME \d+ GPU=([0-9.]+) ms", text)]


def render_set(binary, scene, directory, width, height, spp, lights, sequence, name, variant=(), audit=False):
    prefix = directory / name
    text = run(sequence_command(binary, scene, directory / f"{name}.exr", prefix, width, height, spp, lights,
                                sequence, variant, audit), directory / f"{name}.log")
    return load_frames(prefix), text


def main():
    parser = argparse.ArgumentParser(description="ReSTIR DI motion utility audit")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--debug-binary", type=Path, default=Path("build/Debug/StrelkaCLI"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--lights", type=int, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--reference-spp", type=int, default=1024)
    parser.add_argument("--equal-time-ms", type=float, default=50.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs/restir-bias-benchmark.json"
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("restir benchmark manifest hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    scene = args.scene.resolve()
    release = args.binary.resolve()
    debug = args.debug_binary.resolve()

    references, _ = render_set(release, scene, args.output, args.width, args.height, args.reference_spp,
                               args.lights, args.sequence, "reference")
    motion_config = args.output / "motion.toml"
    motion_config.write_text("[render]\ndebug = 8\n")
    motion_prefix = args.output / "motion"
    command = sequence_command(release, scene, args.output / "motion.exr", motion_prefix, args.width, args.height, 1,
                               args.lights, args.sequence)
    command[2:2] = ["-c", str(motion_config)]
    run(command, args.output / "motion.log")
    motions = load_frames(motion_prefix)

    result = {"manifest_sha256": MANIFEST_SHA256, "scene_sha256": sha256(scene), "frames": list(range(8, 24)),
              "resolution": [args.width, args.height], "reference_spp_per_frame": args.reference_spp,
              "equal_time_budget_ms": args.equal_time_ms, "variants": {}}
    timings = {}

    def equal_time(name, variant, timing):
        spp = max(1, round(args.equal_time_ms / timing))
        equal_frames, text = render_set(release, scene, args.output, args.width, args.height, spp, args.lights,
                                        args.sequence, f"equal-{name}", variant)
        measured = float(np.median(gpu_times(text)))
        corrected_spp = max(1, round(spp * args.equal_time_ms / measured))
        if corrected_spp != spp:
            spp = corrected_spp
            equal_frames, text = render_set(release, scene, args.output, args.width, args.height, spp, args.lights,
                                            args.sequence, f"equal-{name}", variant)
            measured = float(np.median(gpu_times(text)))
        equal = frame_metrics(equal_frames, references)
        equal["spp"] = spp
        equal["measured_gpu_ms"] = measured
        equal["temporal_error"] = temporal_error(equal_frames, references, motions)
        return equal

    for name, variant in VARIANTS.items():
        frames, text = render_set(release, scene, args.output, args.width, args.height, 1, args.lights, args.sequence,
                                  name, variant)
        times = gpu_times(text)
        timings[name] = float(np.median(times))
        debug_text = run(sequence_command(debug, scene, args.output / f"audit-{name}.exr", None, args.width,
                                          args.height, 1, args.lights, args.sequence, variant, True),
                         args.output / f"audit-{name}.log")
        utility = audit_json(debug_text)
        metric = frame_metrics(frames, references)
        metric["gpu_ms"] = timings[name]
        metric["temporal_error"] = temporal_error(frames, references, motions)
        metric["utility"] = utility["restirUtility"]
        metric["work"] = {key: utility[key] for key in
                          ("restirInitialCandidates", "temporalReservoirMerges", "spatialReservoirMerges",
                           "restirEffectiveM", "temporalRejects", "restirCandidateQueries", "restirReuseQueries",
                           "restirDiagnosticQueries", "finalRestirVisibilityRays", "dispatchCount")}
        result["variants"][name] = metric

    for name, variant in VARIANTS.items():
        result["variants"][name]["equal_time"] = equal_time(name, variant, timings[name])

    best = min(VARIANTS, key=lambda name: result["variants"][name]["equal_time"]["mean_single_frame_rmse"])
    c2_name = f"c2_{best}"
    c2_variant = list(VARIANTS[best])
    c2_variant[c2_variant.index("--restir-candidates") + 1] = "2"
    frames, text = render_set(release, scene, args.output, args.width, args.height, 1, args.lights, args.sequence,
                              c2_name, c2_variant)
    times = gpu_times(text)
    timing = float(np.median(times))
    debug_text = run(sequence_command(debug, scene, args.output / f"audit-{c2_name}.exr", None, args.width,
                                      args.height, 1, args.lights, args.sequence, c2_variant, True),
                     args.output / f"audit-{c2_name}.log")
    utility = audit_json(debug_text)
    metric = frame_metrics(frames, references)
    metric["gpu_ms"] = timing
    metric["temporal_error"] = temporal_error(frames, references, motions)
    metric["utility"] = utility["restirUtility"]
    metric["work"] = {key: utility[key] for key in
                      ("restirInitialCandidates", "temporalReservoirMerges", "spatialReservoirMerges",
                       "restirEffectiveM", "temporalRejects", "restirCandidateQueries", "restirReuseQueries",
                       "restirDiagnosticQueries", "finalRestirVisibilityRays", "dispatchCount")}
    metric["equal_time"] = equal_time(c2_name, c2_variant, timing)
    result["best_c1_variant"] = best
    result["variants"][c2_name] = metric

    result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
