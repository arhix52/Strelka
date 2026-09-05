#!/usr/bin/python3
import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "feature_tests"))
from exr_io import load_exr  # noqa: E402


SCENES = ("uniform", "distributed", "occluded")
VARIANTS = {
    "nee": [],
    "off_c1": ["--restir-di", "--restir-candidates", "1", "--restir-temporal=true",
                "--restir-spatial=true", "--restir-neighbors", "2", "--restir-bias-correction", "off"],
    "basic_c1": ["--restir-di", "--restir-candidates", "1", "--restir-temporal=true",
                  "--restir-spatial=true", "--restir-neighbors", "2", "--restir-bias-correction", "basic"],
    "basic_c2": ["--restir-di", "--restir-candidates", "2", "--restir-temporal=true",
                  "--restir-spatial=true", "--restir-neighbors", "2", "--restir-bias-correction", "basic"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def glb_json(path: Path) -> dict:
    with path.open("rb") as source:
        source.read(12)
        length, kind = struct.unpack("<II", source.read(8))
        if kind != 0x4E4F534A:
            raise RuntimeError(f"{path}: first GLB chunk is not JSON")
        return json.loads(source.read(length).rstrip(b"\0 \t\r\n"))


def camera_description(path: Path) -> dict:
    document = glb_json(path)
    nodes = []
    for index, node in enumerate(document.get("nodes", [])):
        if "camera" in node:
            nodes.append({"node": index, "camera": node["camera"],
                          "matrix": node.get("matrix"), "translation": node.get("translation"),
                          "rotation": node.get("rotation"), "scale": node.get("scale")})
    cameras = document.get("cameras", [])
    if not cameras:
        return {"selected": 0, "source": "HeadlessApp AABB auto-fit", "fov_degrees": 45.0,
                "orientation": [1.0, 0.0, 0.0, 0.0]}
    return {"cameras": cameras, "nodes": nodes, "selected": 0}


def run_render(binary: Path, scene: Path, output: Path, width: int, height: int, spp: int,
               variant: list[str], profile: bool = False) -> str:
    command = [str(binary), "-o", str(output), "-w", str(width), "--height", str(height),
               "--spp", str(spp), "--depth", "4", "--tonemap", "none"]
    if profile:
        command.append("--profile-stages")
    command.extend(variant)
    command.append(str(scene))
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    if completed.returncode:
        raise RuntimeError(f"render failed: {' '.join(command)}\n{completed.stdout[-3000:]}")
    return completed.stdout


def metrics(image: np.ndarray, reference: np.ndarray) -> dict:
    luminance = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    reference_luminance = (reference[..., 0] * 0.2126 + reference[..., 1] * 0.7152 +
                           reference[..., 2] * 0.0722)
    return {
        "mean_ratio": float(np.mean(luminance, dtype=np.float64) /
                            np.mean(reference_luminance, dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(image - reference), dtype=np.float64) /
                              np.mean(np.square(reference), dtype=np.float64))),
    }


def summarize(values: list[dict]) -> dict:
    result = {"windows": values}
    for key in ("mean_ratio", "rmse"):
        samples = [value[key] for value in values]
        result[key] = float(np.mean(samples))
        result[f"{key}_spread"] = float(max(samples) - min(samples))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical linear-HDR ReSTIR OFF/BASIC benchmark")
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--scenes", type=Path, default=Path("/private/tmp/strelka-restir-triage-20260904"))
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/restir-bias-canonical"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/restir-bias-benchmark.json"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--window-spp", type=int, default=128)
    parser.add_argument("--reference-spp", type=int, default=1024)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = (root / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    manifest_path = (root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {
        "schema": 1, "commit": commit, "device": "Apple M4 Pro", "build": "Release",
        "resolution": [args.width, args.height], "depth": 4, "spp_per_frame": 1,
        "restir": {"temporal": True, "spatial_neighbors": 2, "max_history_length": 20,
                   "variants": {name: values for name, values in VARIANTS.items()}},
        "sampling": {"sampler": "Sobol/Owen", "window_spp": args.window_spp,
                     "windows": [[0, args.window_spp], [args.window_spp, 2 * args.window_spp],
                                 [2 * args.window_spp, 3 * args.window_spp]]},
        "image": {"encoding": "linear RGB float EXR", "tonemap": "none", "roi": "all pixels",
                  "rmse": "sqrt(mean((rgb-ref)^2)/mean(ref^2))"},
        "timing": {"warmup_frames": 3, "measured_frames": 5, "runs": 5,
                   "aggregation": "median frames per run, then median runs",
                   "ordering": "rotating round-robin variants within each run", "validation": "disabled"},
        "scenes": {},
    }
    for scene_name in SCENES:
        scene_dir = args.scenes / scene_name
        scene = scene_dir / f"{scene_name}.glb"
        sidecar = scene_dir / f"{scene_name}_light.json"
        reference = args.output / f"{scene_name}-reference-{args.reference_spp}.exr"
        if not args.reuse_existing or not reference.exists():
            run_render(binary, scene, reference, args.width, args.height, args.reference_spp, [])
        reference_image = load_exr(reference)
        if not np.isfinite(reference_image).all():
            raise RuntimeError(f"{reference}: non-finite linear reference")
        scene_result = {
            "scene": str(scene), "scene_sha256": sha256(scene),
            "config": str(sidecar), "config_sha256": sha256(sidecar), "camera": camera_description(scene),
            "reference": str(reference), "reference_sha256": sha256(reference),
            "reference_spp": args.reference_spp, "actual_frames": args.reference_spp,
            "results": {},
        }
        for variant_name, variant in VARIANTS.items():
            cumulative_images = []
            for window in range(1, 4):
                spp = window * args.window_spp
                output = args.output / f"{scene_name}-{variant_name}-{spp}.exr"
                if not args.reuse_existing or not output.exists():
                    run_render(binary, scene, output, args.width, args.height, spp, variant)
                cumulative_images.append(load_exr(output))
            windows = []
            previous = np.zeros_like(cumulative_images[0])
            previous_spp = 0
            for index, cumulative in enumerate(cumulative_images, 1):
                spp = index * args.window_spp
                image = (cumulative * spp - previous * previous_spp) / args.window_spp
                windows.append(metrics(image, reference_image))
                previous, previous_spp = cumulative, spp
            summary = summarize(windows)
            summary["actual_frames"] = 3 * args.window_spp
            scene_result["results"][variant_name] = summary
        timing_runs = {name: [] for name in VARIANTS}
        names = list(VARIANTS)
        for run in range(5):
            order = names[run % len(names):] + names[:run % len(names)]
            for variant_name in order:
                timing_output = args.output / f"timing-{scene_name}-{variant_name}-{run}.exr"
                log = run_render(binary, scene, timing_output, args.width, args.height, 8,
                                 VARIANTS[variant_name], True)
                frame_times = [float(value) for value in re.findall(r"CMDBUF gpu ([0-9.]+) ms", log)]
                if len(frame_times) != 8:
                    raise RuntimeError(f"missing GPU timestamps for {scene_name}/{variant_name}")
                timing_runs[variant_name].append(float(np.median(frame_times[3:])))
        for variant_name, runs in timing_runs.items():
            scene_result["results"][variant_name]["gpu_ms"] = float(np.median(runs))
            scene_result["results"][variant_name]["gpu_ms_runs"] = runs
        manifest["scenes"][scene_name] = scene_result
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
