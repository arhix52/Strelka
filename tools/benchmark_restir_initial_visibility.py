#!/usr/bin/python3
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from audit_restir_utility import (MANIFEST_SHA256, audit_json, frame_metrics, gpu_times, load_frames,
                                  sequence_command, sha256, temporal_error)


def restir(visible, temporal, spatial, neighbors=0):
    return ["--restir-di", "--restir-candidates", "1",
            f"--restir-temporal={'true' if temporal else 'false'}",
            f"--restir-spatial={'true' if spatial else 'false'}", "--restir-neighbors", str(neighbors),
            "--restir-bias-correction", "basic", "--restir-initial-visibility", "on" if visible else "off"]


VARIANTS = {
    "Initial": restir(False, False, False),
    "InitialVis": restir(True, False, False),
    "InitialVis+T": restir(True, True, False),
    "InitialVis+S1": restir(True, False, True, 1),
    "InitialVis+T+S1": restir(True, True, True, 1),
    "InitialVis+T+S2": restir(True, True, True, 2),
}


def run(command, log, env=None):
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, timeout=1800)
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:])
    return completed.stdout


def reference_hash(directory):
    digest = hashlib.sha256()
    for frame in range(8, 24):
        digest.update((directory / f"reference-{frame:02}.exr").read_bytes())
    return digest.hexdigest()


def render_frames(binary, scene, output, width, height, spp, lights, sequence, name, flags):
    prefix = output / name
    command = sequence_command(binary, scene, output / f"{name}.exr", prefix, width, height, spp, lights,
                               sequence, flags)
    text = run(command, output / f"{name}.log")
    return load_frames(prefix), text


def equal_time(binary, scene, output, references, width, height, lights, sequence, name, flags, timing, budget):
    spp = max(1, round(budget / timing))
    frames, text = render_frames(binary, scene, output, width, height, spp, lights, sequence,
                                 f"equal-{name}", flags)
    measured = float(np.median(gpu_times(text)[3:]))
    corrected = max(1, round(spp * budget / measured))
    if corrected != spp:
        spp = corrected
        frames, text = render_frames(binary, scene, output, width, height, spp, lights, sequence,
                                     f"equal-{name}", flags)
        measured = float(np.median(gpu_times(text)[3:]))
    result = frame_metrics(frames, references)
    result.update({"spp": spp, "gpu_ms": measured})
    return result


def main():
    parser = argparse.ArgumentParser(description="ReSTIR initial-visibility preset benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moving-reference", type=Path, required=True)
    parser.add_argument("--local-reference", type=Path, required=True)
    parser.add_argument("--moving-scene", type=Path,
                        default=Path("scenes/validation/cornell_box/cornell_box.glb"))
    parser.add_argument("--local-scene", type=Path,
                        default=Path("scenes/validation/restir_local_many/restir_local_many.gltf"))
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--debug-binary", type=Path, default=Path("build/Debug/StrelkaCLI"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--equal-time-ms", type=float, default=50.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs/restir-bias-benchmark.json"
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("restir benchmark manifest hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = {
        "moving_balanced": (args.moving_scene.resolve(), args.moving_reference.resolve(), 512, 4),
        "local_many": (args.local_scene.resolve(), args.local_reference.resolve(), 4096, 3),
    }
    result = {"schema": 1, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                               text=True).strip(),
              "manifest_sha256": MANIFEST_SHA256, "device": "Apple M4 Pro",
              "protocol": {"resolution": [args.width, args.height], "depth": 4, "spp_per_frame": 1,
                           "frames": list(range(8, 24)), "tonemap": "none", "clamp": 0,
                           "timing": "rotating order; median of 5 run medians",
                           "equal_time_ms": args.equal_time_ms}, "scenes": {}}
    audit_env = os.environ.copy()
    audit_env["STRELKA_RESTIR_DIAGNOSTIC_PIXELS"] = "18240,18248,18256,20080,20088,20096,21920,21928"

    for scene_name, (scene, reference_dir, lights, sequence) in scenes.items():
        references = load_frames(reference_dir / "reference")
        motions = load_frames(reference_dir / "motion")
        output = args.output / scene_name
        output.mkdir(exist_ok=True)
        timings = {name: [] for name in VARIANTS}
        quality = {}
        names = list(VARIANTS)
        for repetition in range(5):
            for name in names[repetition:] + names[:repetition]:
                frames, text = render_frames(args.binary.resolve(), scene, output, args.width, args.height, 1,
                                             lights, sequence, f"timing-{repetition}-{name}", VARIANTS[name])
                timings[name].append(float(np.median(gpu_times(text)[3:])))
                if repetition == 0:
                    quality[name] = frames

        scene_result = {"scene_sha256": sha256(scene), "reference_set_sha256": reference_hash(reference_dir),
                        "lights": lights, "motion_sequence": sequence, "variants": {}}
        for name, flags in VARIANTS.items():
            debug_command = sequence_command(args.debug_binary.resolve(), scene, output / f"audit-{name}.exr",
                                             None, args.width, args.height, 1, lights, sequence, flags, True)
            audit = audit_json(run(debug_command, output / f"audit-{name}.log", audit_env))
            metric = frame_metrics(quality[name], references)
            metric["gpu_ms"] = float(np.median(timings[name]))
            metric["gpu_ms_runs"] = timings[name]
            metric["flicker"] = temporal_error(quality[name], references, motions)
            history = audit["restirUtility"]["finalHistory"]
            initial_rays = audit["restirInitialVisibilityQueries"] / audit["frames"]
            final_rays = audit["finalRestirVisibilityRays"] / audit["frames"]
            reused = audit["restirInitialVisibilityReused"] / audit["frames"]
            records = audit["candidateIndependence"]["records"]
            metric.update({
                "history_to_visible_pct": 100.0 * audit["restirUtility"]["finalHistoryVisible"] / max(history, 1),
                "initial_visibility_rays": initial_rays, "final_visibility_rays": final_rays,
                "total_visibility_rays": initial_rays + final_rays,
                "same_frame_visibility_reuse_pct": 100.0 * reused / max(reused + final_rays, 1),
                "candidate_exact_collisions": audit["candidateIndependence"]["exactSample"],
                "effective_light_count": 1.0 / records[0]["pmfCollision"] if records else None,
            })
            metric["equal_time"] = equal_time(args.binary.resolve(), scene, output, references, args.width,
                                                args.height, lights, sequence, name, flags, metric["gpu_ms"],
                                                args.equal_time_ms)
            scene_result["variants"][name] = metric
        result["scenes"][scene_name] = scene_result

    local = result["scenes"]["local_many"]["variants"]
    moving = result["scenes"]["moving_balanced"]["variants"]
    accepted = []
    for name in list(VARIANTS)[1:]:
        if (local[name]["equal_time"]["mean_single_frame_rmse"] <=
                0.9 * local["Initial"]["equal_time"]["mean_single_frame_rmse"] and
                abs(local[name]["mean_ratio"] - 1.0) <= 0.02 and
                local[name]["history_to_visible_pct"] >= 80.0 and
                local[name]["total_visibility_rays"] <= 1.5 * local["Initial"]["total_visibility_rays"] and
                moving[name]["equal_time"]["mean_single_frame_rmse"] <=
                1.03 * moving["Initial"]["equal_time"]["mean_single_frame_rmse"]):
            accepted.append(name)
    result["accepted_presets"] = accepted
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
