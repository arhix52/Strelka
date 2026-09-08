#!/usr/bin/env python3
"""Metal Sobol sampler profiling: timing ablations and image-quality deltas.

Mirrors the OptiX methodology in docs/open-perf.md (1280x720, depth 4, spp 24,
median ms/sample over samples 8..24). Writes results under ~/strelka_metal_profile/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLI = ROOT / "build" / "Release" / "StrelkaCLI"
OUT_ROOT = Path.home() / "strelka_metal_profile"

SCENES = {
    "iso_bathroom": {
        "path": ROOT / "scenes/iso_bathroom/iso_bathroom.gltf",
        "camera": 1,
    },
    "kids_room": {
        "path": ROOT / "scenes/kids_room/kids_room.gltf",
        "camera": 0,
    },
    "pine_scene": {
        "path": Path.home() / "pine_scene/polyhaven_pine_fir_forest.gltf",
        "camera": 0,
    },
}

SAMPLERS = ("sobol", "pcg", "sobol_notable")

MS_RE = re.compile(r"(\d+\.\d+)\s+ms/sample")


@dataclass
class TimingResult:
    scene: str
    sampler: str
    median_ms: float
    samples: list[float]
    elapsed_s: float


def parse_ms_sample(stdout: str) -> list[float]:
    return [float(m.group(1)) for m in MS_RE.finditer(stdout)]


def run_render(cli: Path, scene: str, sampler: str, spp: int, width: int, height: int,
               depth: int, camera: int, out_path: Path) -> tuple[list[float], str]:
    cmd = [
        str(cli),
        str(SCENES[scene]["path"]),
        "-o", str(out_path),
        "-w", str(width),
        "--height", str(height),
        "--spp", str(spp),
        "--depth", str(depth),
        "--camera", str(camera),
        "--sampler", sampler,
        "--tonemap", "none",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cli.parent)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"render failed ({scene}/{sampler}):\n{proc.stderr}\n{proc.stdout}")
    samples = parse_ms_sample(proc.stdout + proc.stderr)
    return samples, proc.stdout + proc.stderr


def median_window(samples: list[float], lo: int, hi: int) -> float:
    window = samples[lo:hi]
    if not window:
        raise ValueError("empty timing window")
    return statistics.median(window)


def try_load_exr(path: Path):
    try:
        import numpy as np
        import OpenEXR
        import Imath
    except ImportError:
        return None
    if not path.exists():
        return None
    exr = OpenEXR.InputFile(str(path))
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r = np.frombuffer(exr.channel("R", pt), dtype=np.float32).reshape(h, w)
    g = np.frombuffer(exr.channel("G", pt), dtype=np.float32).reshape(h, w)
    b = np.frombuffer(exr.channel("B", pt), dtype=np.float32).reshape(h, w)
    return np.stack([r, g, b], axis=-1)


def image_metrics(ref, img):
    import numpy as np
    diff = img - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mean_ref = float(np.mean(ref))
    rel_rmse = rmse / max(mean_ref, 1e-6)
    moved = float(np.mean(np.any(np.abs(diff) > 1e-3, axis=-1)))
    return {"rmse": rmse, "rel_rmse": rel_rmse, "frac_moved_gt_1e-3": moved}


def run_timings(cli: Path, width: int, height: int, depth: int, spp: int,
                warmup_lo: int, measure_lo: int, measure_hi: int) -> list[TimingResult]:
    results: list[TimingResult] = []
    for scene, meta in SCENES.items():
        if not meta["path"].exists():
            print(f"skip {scene}: missing {meta['path']}", file=sys.stderr)
            continue
        for sampler in SAMPLERS:
            out = OUT_ROOT / "images" / f"{scene}_{sampler}.exr"
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"timing {scene} sampler={sampler} ...", flush=True)
            samples, _ = run_render(cli, scene, sampler, spp, width, height, depth,
                                    meta["camera"], out)
            if len(samples) < measure_hi:
                raise RuntimeError(f"{scene}/{sampler}: expected >={measure_hi} samples, got {len(samples)}")
            med = median_window(samples, measure_lo, measure_hi)
            results.append(TimingResult(scene, sampler, med, samples, 0.0))
            print(f"  median ms/sample [{measure_lo}:{measure_hi}] = {med:.2f}", flush=True)
    return results


def run_quality() -> dict:
    quality = {}
    refs = {}
    for scene in SCENES:
        ref_path = OUT_ROOT / "images" / f"{scene}_sobol.exr"
        ref = try_load_exr(ref_path)
        if ref is not None:
            refs[scene] = ref
    if not refs:
        return {"note": "OpenEXR/numpy unavailable or no reference images"}
    for scene, ref in refs.items():
        quality[scene] = {}
        for sampler in SAMPLERS:
            if sampler == "sobol":
                continue
            img = try_load_exr(OUT_ROOT / "images" / f"{scene}_{sampler}.exr")
            if img is None:
                continue
            quality[scene][sampler] = image_metrics(ref, img)
    return quality


def main() -> int:
    global OUT_ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--spp", type=int, default=24)
    ap.add_argument("--measure-lo", type=int, default=8)
    ap.add_argument("--measure-hi", type=int, default=24)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    OUT_ROOT = args.out
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.cli.exists():
        print(f"StrelkaCLI not found: {args.cli}", file=sys.stderr)
        return 1

    timings = run_timings(args.cli, args.width, args.height, args.depth, args.spp,
                          args.measure_lo, args.measure_lo, args.measure_hi)
    quality = run_quality()

    report = {
        "device_note": "see xctrace captures in same directory",
        "config": {
            "width": args.width,
            "height": args.height,
            "depth": args.depth,
            "spp": args.spp,
            "measure_window": [args.measure_lo, args.measure_hi],
        },
        "timings": [asdict(t) for t in timings],
        "quality_vs_sobol": quality,
    }

    out_json = OUT_ROOT / "sobol_ablation.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")

    # Pretty summary
    scenes = sorted({t.scene for t in timings})
    print("\n| scene | sobol ms | pcg ms | sobol_notable ms | pcg/sobol | notable/sobol |")
    print("|---|---:|---:|---:|---:|---:|")
    by = {(t.scene, t.sampler): t.median_ms for t in timings}
    for scene in scenes:
        s = by.get((scene, "sobol"))
        p = by.get((scene, "pcg"))
        n = by.get((scene, "sobol_notable"))
        if s is None:
            continue
        print(f"| {scene} | {s:.2f} | {p:.2f} | {n:.2f} | {p/s:.3f} | {n/s:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
