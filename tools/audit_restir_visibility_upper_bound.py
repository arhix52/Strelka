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


def restir(candidates, temporal=True, spatial=True, bias="basic", initial_visibility="off"):
    return ["--restir-di", "--restir-candidates", str(candidates),
            f"--restir-temporal={'true' if temporal else 'false'}",
            f"--restir-spatial={'true' if spatial else 'false'}", "--restir-neighbors", "2",
            "--restir-bias-correction", bias, "--restir-initial-visibility", initial_visibility]


VARIANTS = {
    "current": restir(1),
    "initial_visibility": restir(1, initial_visibility="on"),
    "reuse_visibility": restir(1, bias="raytraced-diagnostic"),
    "both_visibility": restir(1, bias="raytraced-diagnostic", initial_visibility="on"),
    "initial_c1": restir(1, False, False, "off"),
    "initial_c2": restir(2, False, False, "off"),
    "initial_c8": restir(8, False, False, "off"),
}


def run(command, env=None):
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, timeout=1800)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-4000:])
    return completed.stdout


def reference_hash(directory):
    digest = hashlib.sha256()
    for frame in range(8, 24):
        digest.update((directory / f"reference-{frame:02}.exr").read_bytes())
    return digest.hexdigest()


def render_frames(binary, scene, output, width, height, spp, lights, sequence, name, flags, freeze=False):
    prefix = output / name
    command = sequence_command(binary, scene, output / f"{name}.exr", prefix, width, height, spp, lights,
                               sequence, flags)
    if freeze:
        command.append("--audit-freeze")
    text = run(command)
    return load_frames(prefix), text


def equal_time(binary, scene, output, references, width, height, lights, sequence, name, flags, timing):
    spp = max(1, round(50.0 / timing))
    frames, text = render_frames(binary, scene, output, width, height, spp, lights, sequence,
                                 f"equal-{name}", flags, True)
    measured = float(np.median(gpu_times(text)[3:]))
    corrected = max(1, round(spp * 50.0 / measured))
    if corrected != spp:
        spp = corrected
        frames, text = render_frames(binary, scene, output, width, height, spp, lights, sequence,
                                     f"equal-{name}", flags, True)
        measured = float(np.median(gpu_times(text)[3:]))
    result = frame_metrics(frames, references)
    result.update({"spp": spp, "gpu_ms": measured})
    return result


def main():
    parser = argparse.ArgumentParser(description="Focused ReSTIR visibility upper-bound audit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--moving-scene", type=Path, required=True)
    parser.add_argument("--moving-reference", type=Path, required=True)
    parser.add_argument("--local-scene", type=Path, default=Path("scenes/validation/restir_local_many/restir_local_many.gltf"))
    parser.add_argument("--local-reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--debug-binary", type=Path, default=Path("build/Debug/StrelkaCLI"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--reference-spp", type=int, default=2048)
    parser.add_argument("--only", choices=("moving_512", "local_many"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs/restir-bias-benchmark.json"
    if sha256(manifest) != MANIFEST_SHA256:
        raise RuntimeError("restir benchmark manifest hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = {
        "moving_512": (args.moving_scene.resolve(), args.moving_reference.resolve(), 512, 1),
        "local_many": (args.local_scene.resolve(), args.local_reference.resolve(), 4096, 3),
    }
    result = {"schema": 1, "manifest_sha256": MANIFEST_SHA256, "resolution": [args.width, args.height],
              "frames": list(range(8, 24)), "timing": "rotating order; median of 5 run medians",
              "reference_spp_per_frame": args.reference_spp, "tonemap": "none", "clamp": 0, "scenes": {}}

    for scene_name, (scene, reference_dir, lights, sequence) in scenes.items():
        if args.only and scene_name != args.only:
            continue
        references = load_frames(reference_dir / "reference")
        motions = load_frames(reference_dir / "motion")
        scene_output = args.output / scene_name
        scene_output.mkdir(exist_ok=True)
        timing_runs = {name: [] for name in VARIANTS}
        quality_frames = {}
        names = list(VARIANTS)
        for repetition in range(5):
            order = names[repetition:] + names[:repetition]
            for name in order:
                frames, text = render_frames(args.binary.resolve(), scene, scene_output, args.width, args.height, 1,
                                             lights, sequence, f"timing-{repetition}-{name}", VARIANTS[name])
                timing_runs[name].append(float(np.median(gpu_times(text)[3:])))
                if repetition == 0:
                    quality_frames[name] = frames

        light_config = scene.with_name(f"{scene.stem}_light.json")
        scene_result = {"scene_sha256": sha256(scene),
                        "light_config_sha256": sha256(light_config) if light_config.exists() else None,
                        "reference_set_sha256": reference_hash(reference_dir), "lights": lights,
                        "motion_sequence": sequence, "variants": {}}
        for name, flags in VARIANTS.items():
            metric = frame_metrics(quality_frames[name], references)
            timing = float(np.median(timing_runs[name]))
            debug_command = sequence_command(args.debug_binary.resolve(), scene, scene_output / f"audit-{name}.exr",
                                             None, args.width, args.height, 1, lights, sequence, flags, True)
            env = os.environ.copy()
            env["STRELKA_RESTIR_DIAGNOSTIC_PIXELS"] = "18240,18248,18256,20080,20088,20096,21920,21928"
            audit = audit_json(run(debug_command, env))
            utility = audit["restirUtility"]
            source_total = sum(utility["finalSources"])
            history_rays = utility["finalHistoryRays"]
            metric.update({
                "gpu_ms": timing,
                "gpu_ms_runs": timing_runs[name],
                "flicker": temporal_error(quality_frames[name], references, motions),
                "history_selected_pct": 100.0 * utility["finalHistory"] / max(source_total, 1),
                "history_visible_pct": 100.0 * utility["finalHistoryVisible"] / max(history_rays, 1),
                "candidate_exact_collision_pct":
                    100.0 * audit["candidateIndependence"]["exactSample"] /
                    max(audit["temporalReservoirMerges"], 1),
                "visibility_rays_per_frame":
                    (audit["finalRestirVisibilityRays"] + audit["restirDiagnosticQueries"] +
                     audit["restirInitialVisibilityQueries"]) / audit["frames"],
            })
            metric["equal_time"] = equal_time(args.binary.resolve(), scene, scene_output, references, args.width,
                                                args.height, lights, sequence, name, flags, timing)
            scene_result["variants"][name] = metric
        result["scenes"][scene_name] = scene_result

    result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
