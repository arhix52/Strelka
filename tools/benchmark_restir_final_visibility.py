#!/usr/bin/python3
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from audit_restir_utility import (MANIFEST_SHA256, audit_json, frame_metrics, gpu_times, load_frames,
                                  render_set, sha256, temporal_error)


def restir(initial_visibility, temporal, spatial, reuse):
    return ["--restir-di", "--restir-candidates", "1",
            f"--restir-temporal={'true' if temporal else 'false'}",
            f"--restir-spatial={'true' if spatial else 'false'}",
            "--restir-neighbors", "1" if spatial else "0", "--restir-bias-correction", "basic",
            "--restir-initial-visibility", "on" if initial_visibility else "off",
            "--restir-final-visibility-reuse", reuse, "--restir-final-visibility-max-age", "4"]


PRESETS = {
    "Initial": (False, False, False),
    "InitialVis+T": (True, True, False),
    "InitialVis+S1": (True, False, True),
    "InitialVis+T+S1": (True, True, True),
}
VARIANTS = {f"{name}/{reuse}": restir(*preset, reuse)
            for name, preset in PRESETS.items() for reuse in ("off", "conservative")}


def require_finite(name, frames):
    for frame, image in enumerate(frames, 8):
        if not np.isfinite(image).all():
            raise RuntimeError(f"{name} frame {frame} contains non-finite HDR values")


def main():
    parser = argparse.ArgumentParser(description="ReSTIR conservative final-visibility benchmark")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--debug-binary", type=Path, default=Path("build/Debug/StrelkaCLI"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--lights", type=int, default=0)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--reference-spp", type=int, default=256)
    parser.add_argument("--equal-time-ms", type=float, default=50.0)
    parser.add_argument("--timing-runs", type=int, default=5)
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
    require_finite("reference", references)
    if args.lights:
        motion_config = args.output / "motion.toml"
        motion_config.write_text("[render]\ndebug = 8\n")
        from audit_restir_utility import run, sequence_command
        command = sequence_command(release, scene, args.output / "motion.exr", args.output / "motion",
                                   args.width, args.height, 1, args.lights, args.sequence)
        command[2:2] = ["-c", str(motion_config)]
        run(command, args.output / "motion.log")
        motions = load_frames(args.output / "motion")
    else:
        motions = [np.full_like(references[0], 0.5) for _ in references]

    timings = {name: [] for name in VARIANTS}
    quality = {}
    names = list(VARIANTS)
    for repetition in range(args.timing_runs):
        for name in names[repetition:] + names[:repetition]:
            slug = name.replace("/", "-")
            frames, text = render_set(release, scene, args.output, args.width, args.height, 1, args.lights,
                                      args.sequence, f"timing-{repetition}-{slug}", VARIANTS[name])
            require_finite(name, frames)
            timings[name].append(float(np.median(gpu_times(text)[3:])))
            if repetition == 0:
                quality[name] = frames

    result = {
        "schema": 1, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                          text=True).strip(),
        "manifest_sha256": MANIFEST_SHA256, "scene": str(scene), "scene_sha256": sha256(scene),
        "device": "Apple M4 Pro", "resolution": [args.width, args.height], "depth": 4,
        "frames": list(range(8, 24)), "tonemap": "none", "timing": "rotating order, median run medians",
        "variants": {},
    }
    for name, flags in VARIANTS.items():
        slug = name.replace("/", "-")
        frames = quality[name]
        metric = frame_metrics(frames, references)
        metric["flicker"] = temporal_error(frames, references, motions)
        metric["gpu_ms"] = float(np.median(timings[name]))
        metric["gpu_ms_runs"] = timings[name]
        _, text = render_set(debug, scene, args.output, args.width, args.height, 1, args.lights, args.sequence,
                             f"audit-{slug}", flags, True)
        audit = audit_json(text)
        cache = audit["restirFinalVisibilityCache"]
        frames_count = audit["frames"]
        metric.update({
            "initial_rays": audit["restirInitialVisibilityQueries"] / frames_count,
            "final_rays": audit["finalRestirVisibilityRays"] / frames_count,
            "cache_hit_pct": 100.0 * cache["hits"] / max(cache["attempts"], 1),
            "visible_visible_pct": 100.0 * cache["visibleVisible"] / max(cache["oracleQueries"], 1),
            "visible_occluded_pct": 100.0 * cache["visibleOccluded"] / max(cache["oracleQueries"], 1),
            "cache_rejects": cache["rejects"], "dispatches": audit["dispatchCount"],
            "restir_bytes_per_pixel": audit["memory"]["restirAllocatedBytes"] / audit["pixels"],
        })
        spp = max(1, round(args.equal_time_ms / metric["gpu_ms"]))
        equal, equal_text = render_set(release, scene, args.output, args.width, args.height, spp, args.lights,
                                       args.sequence, f"equal-{slug}", flags)
        require_finite(f"equal-{name}", equal)
        measured = float(np.median(gpu_times(equal_text)[3:]))
        corrected_spp = max(1, round(spp * args.equal_time_ms / measured))
        if corrected_spp != spp:
            spp = corrected_spp
            equal, equal_text = render_set(release, scene, args.output, args.width, args.height, spp, args.lights,
                                           args.sequence, f"equal-{slug}", flags)
            require_finite(f"equal-{name}", equal)
        equal_metric = frame_metrics(equal, references)
        equal_metric["gpu_ms"] = float(np.median(gpu_times(equal_text)[3:]))
        equal_metric["spp"] = spp
        metric["equal_time"] = equal_metric
        result["variants"][name] = metric

    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
