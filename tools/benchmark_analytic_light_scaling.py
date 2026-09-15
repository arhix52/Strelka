#!/usr/bin/env python3
import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def lights(count: int, shape: str) -> dict:
    columns = math.ceil(math.sqrt(count * 4.0 / 3.0))
    rows = math.ceil(count / columns)
    width = min(0.12, 1.6 / columns * 0.65)
    height = min(0.12, 1.6 / rows * 0.65)
    result = []
    for index in range(count):
        x = -0.8 + 1.6 * ((index % columns) + 0.5) / columns
        z = -0.8 + 1.6 * ((index // columns) + 0.5) / rows
        result.append(
            {
                "type": shape,
                "position": [x, 1.98, z],
                "orientation": [-90.0, 0.0, 0.0],
                "color": [1.0, 1.0, 1.0],
                "intensity": 400.0 / count,
                "width": width,
                "height": height,
                "radius": 0.5 * min(width, height),
            }
        )
    return {"lights": result}


def run_mode(binary: Path, scene: Path, width: int, height: int, frames: int, restir: bool) -> float:
    output = scene.with_name(f"bench-{'restir' if restir else 'nee'}.exr")
    command = [
        str(binary),
        str(scene),
        "--output",
        str(output),
        "--width",
        str(width),
        "--height",
        str(height),
        "--spp",
        "1",
        "--depth",
        "4",
        "--tonemap",
        "none",
        "--audit-frames",
        str(frames),
        "--audit-motion-sequence",
        "6",
    ]
    if restir:
        command += [
            "--restir-di",
            "--restir-candidates",
            "1",
            "--restir-temporal",
            "--restir-spatial",
            "--restir-neighbors",
            "2",
        ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    match = re.search(r"STRELKA_AUDIT_GPU_MEDIAN ([0-9.]+) ms", completed.stdout)
    if completed.returncode or not match:
        mode = "ReSTIR" if restir else "NEE"
        raise RuntimeError(f"{mode} benchmark failed at {width}x{height}\n{completed.stdout[-2000:]}")
    return float(match.group(1))


def run(binary: Path, scene: Path, width: int, height: int, frames: int) -> tuple[float, float]:
    return run_mode(binary, scene, width, height, frames, False), run_mode(
        binary, scene, width, height, frames, True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Metal analytic-light traversal scaling regression")
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--counts", default="1,32,128,512,2048")
    parser.add_argument("--resolutions", default="320x240,1920x1080")
    parser.add_argument("--max-nee-growth", type=float, default=3.0)
    parser.add_argument("--shape", choices=("rect", "disc", "sphere"), default="rect")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = root / "scenes/validation/cornell_box/cornell_box.glb"
    binary = (root / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    counts = [int(value) for value in args.counts.split(",")]
    resolutions = [tuple(map(int, value.split("x"))) for value in args.resolutions.split(",")]

    print("lights,resolution,nee_ms,restir_c1_t1_s2_ms")
    baseline = {}
    latest = {}
    with tempfile.TemporaryDirectory(prefix="strelka-light-scaling-") as directory:
        scene = Path(directory) / "scaling.glb"
        shutil.copy2(source, scene)
        sidecar = scene.with_name("scaling_light.json")
        for count in counts:
            sidecar.write_text(json.dumps(lights(count, args.shape)), encoding="utf-8")
            for width, height in resolutions:
                nee, restir = run(binary, scene, width, height, args.frames)
                print(f"{count},{width}x{height},{nee:.2f},{restir:.2f}", flush=True)
                baseline.setdefault((width, height), nee)
                latest[(width, height)] = nee

    for resolution, first in baseline.items():
        growth = latest[resolution] / first
        if args.max_nee_growth > 0.0 and growth > args.max_nee_growth:
            raise RuntimeError(f"NEE scaling regression at {resolution[0]}x{resolution[1]}: {growth:.2f}x")


if __name__ == "__main__":
    main()
