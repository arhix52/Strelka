#!/usr/bin/python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_restir_bias import load_exr, metrics, run_render, sha256, summarize  # noqa: E402


STAGES = {
    "initial": (False, False),
    "temporal": (True, False),
    "spatial": (False, True),
    "temporal_spatial": (True, True),
}


def variant(mode: str, candidates: int, temporal: bool, spatial: bool) -> list[str]:
    return ["--restir-di", "--restir-candidates", str(candidates),
            f"--restir-temporal={'true' if temporal else 'false'}",
            f"--restir-spatial={'true' if spatial else 'false'}", "--restir-neighbors", "2",
            "--restir-bias-correction", mode]


def verify_manifest(path: Path, scenes: Path, output: Path, width: int, height: int) -> dict:
    manifest = json.loads(path.read_text())
    if manifest["image"]["tonemap"] != "none" or "linear" not in manifest["image"]["encoding"]:
        raise RuntimeError("manifest is not linear HDR")
    for name, entry in manifest["scenes"].items():
        scene = scenes / name / f"{name}.glb"
        sidecar = scenes / name / f"{name}_light.json"
        if sha256(scene) != entry["scene_sha256"] or sha256(sidecar) != entry["config_sha256"]:
            raise RuntimeError(f"{name}: scene/config hash mismatch")
        if [width, height] == manifest["resolution"]:
            reference = output / f"{name}-reference-{entry['reference_spp']}.exr"
            if sha256(reference) != entry["reference_sha256"]:
                raise RuntimeError(f"{name}: reference hash mismatch")
    return manifest


def tail_stats(images: list[np.ndarray], reference: np.ndarray) -> dict:
    rgb = np.concatenate([image.reshape(-1, 3) for image in images])
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    ref_lum = reference.reshape(-1, 3) @ np.array([0.2126, 0.7152, 0.0722])
    tiled_ref = np.tile(ref_lum, len(images))
    count = max(int(np.ceil(lum.size * 0.001)), 1)
    return {
        "luminance_percentiles": dict(zip(("p50", "p95", "p99", "p99_9"),
                                           map(float, np.percentile(lum, (50, 95, 99, 99.9))))),
        "maximum": float(np.max(lum)),
        "pixels_over_10x_reference": int(np.count_nonzero(lum > 10.0 * tiled_ref)),
        "top_0_1_percent_energy": float(np.partition(lum, -count)[-count:].sum() / lum.sum()),
    }


def audit(binary: Path, audit_binary: Path, scene: Path, output: Path,
          width: int, height: int, args: list[str]) -> dict:
    log = run_render(audit_binary, scene, output, width, height, 8, args + ["--audit-render-work"])
    records = re.findall(r"(?m)^\{\"frames\".*\}$", log)
    if not records:
        raise RuntimeError("missing render-work audit")
    record = json.loads(records[-1])
    timing_runs = []
    for run in range(5):
        timing_output = output.with_name(f"{output.stem}-timing-{run}.exr")
        timing_log = run_render(binary, scene, timing_output, width, height, 8, args, profile=True)
        times = [float(value) for value in re.findall(r"CMDBUF gpu ([0-9.]+) ms", timing_log)]
        if len(times) != 8:
            raise RuntimeError("missing GPU timestamps")
        timing_runs.append(float(np.median(times[3:])))
    return {
        "temporal_merges_per_frame": record["temporalReservoirMerges"] / record["frames"],
        "spatial_merges_per_frame": record["spatialReservoirMerges"] / record["frames"],
        "effective_M": record["restirEffectiveM"],
        "gpu_ms": float(np.median(timing_runs)),
        "gpu_ms_runs": timing_runs,
        "eligible_hits_per_frame": record["restirEligibleHits"] / record["frames"],
        "candidate_queries_per_frame": record["restirCandidateQueries"] / record["frames"],
        "reuse_queries_per_frame": record["restirReuseQueries"] / record["frames"],
        "final_visibility_rays_per_frame": record["finalRestirVisibilityRays"] / record["frames"],
        "diagnostic_queries_per_frame": record["restirDiagnosticQueries"] / record["frames"],
    }


def diagnostic_dump(binary: Path, scene: Path, output: Path, reference: np.ndarray,
                    width: int, height: int, args: list[str]) -> list[dict]:
    probe = output.with_name(output.stem + "-probe.exr")
    run_render(binary, scene, probe, width, height, 8, args + ["--audit-render-work"])
    image = load_exr(probe)
    error = np.abs((image - reference) @ np.array([0.2126, 0.7152, 0.0722])).reshape(-1)
    pixels = np.argpartition(error, -RESTIR_DUMP_COUNT)[-RESTIR_DUMP_COUNT:]
    pixels = pixels[np.argsort(error[pixels])[::-1]]
    previous = os.environ.get("STRELKA_RESTIR_DIAGNOSTIC_PIXELS")
    os.environ["STRELKA_RESTIR_DIAGNOSTIC_PIXELS"] = ",".join(map(str, pixels))
    try:
        log = run_render(binary, scene, output, width, height, 8, args + ["--audit-render-work"])
    finally:
        if previous is None:
            os.environ.pop("STRELKA_RESTIR_DIAGNOSTIC_PIXELS", None)
        else:
            os.environ["STRELKA_RESTIR_DIAGNOSTIC_PIXELS"] = previous
    records = re.findall(r"(?m)^\{\"frames\".*\}$", log)
    return json.loads(records[-1])["restirDiagnostics"]


RESTIR_DUMP_COUNT = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused linear-HDR ReSTIR visibility diagnosis")
    parser.add_argument("--binary", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--audit-binary", type=Path, default=Path("build/Debug/StrelkaCLI"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/restir-bias-benchmark.json"))
    parser.add_argument("--scenes", type=Path, default=Path("/private/tmp/strelka-restir-triage-20260904"))
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/restir-visibility-diagnostic"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--window-spp", type=int, default=32)
    parser.add_argument("--reference-spp", type=int, default=512)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = (root / args.binary).resolve() if not args.binary.is_absolute() else args.binary
    audit_binary = ((root / args.audit_binary).resolve() if not args.audit_binary.is_absolute() else
                    args.audit_binary)
    manifest_path = (root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = verify_manifest(manifest_path, args.scenes, args.output, args.width, args.height)
    result = {"manifest_sha256": sha256(manifest_path), "linear_hdr": True,
              "resolution": [args.width, args.height], "window_spp": args.window_spp, "scenes": {}}
    for scene_name in manifest["scenes"]:
        scene = args.scenes / scene_name / f"{scene_name}.glb"
        reference_path = (args.output / f"{scene_name}-reference-{args.reference_spp}.exr" if
                          [args.width, args.height] != manifest["resolution"] else
                          args.output / f"{scene_name}-reference-{manifest['scenes'][scene_name]['reference_spp']}.exr")
        if [args.width, args.height] == manifest["resolution"]:
            source = Path(manifest["scenes"][scene_name].get("reference", ""))
            if source.is_file():
                reference_path = source
            else:
                reference_path = Path("/private/tmp/restir-bias-canonical") / reference_path.name
        if not reference_path.exists():
            run_render(binary, scene, reference_path, args.width, args.height, args.reference_spp, [])
        reference = load_exr(reference_path)
        variants = {"nee": []}
        stages = {"temporal_spatial": STAGES["temporal_spatial"]} if args.final_only else STAGES
        for stage, (temporal, spatial) in stages.items():
            for mode in ("off", "basic"):
                variants[f"{stage}_{mode}_c1"] = variant(mode, 1, temporal, spatial)
        variants["temporal_spatial_off_c2"] = variant("off", 2, True, True)
        variants["temporal_spatial_basic_c2"] = variant("basic", 2, True, True)
        if not args.final_only:
            variants["temporal_spatial_raytraced_c1"] = variant("raytraced-diagnostic", 1, True, True)
        scene_result = {}
        images_by_variant = {}
        for name, flags in variants.items():
            cumulative = []
            for window in range(1, 4):
                spp = window * args.window_spp
                path = args.output / f"{scene_name}-{name}-{spp}.exr"
                if not args.reuse_existing or not path.exists():
                    run_render(binary, scene, path, args.width, args.height, spp, flags)
                cumulative.append(load_exr(path))
            windows, previous, previous_spp = [], np.zeros_like(cumulative[0]), 0
            images = []
            for window, image in enumerate(cumulative, 1):
                spp = window * args.window_spp
                independent = (image * spp - previous * previous_spp) / args.window_spp
                images.append(independent)
                windows.append(metrics(independent, reference))
                previous, previous_spp = image, spp
            summary = summarize(windows)
            summary.update(audit(binary, audit_binary, scene, args.output / f"audit-{scene_name}-{name}.exr",
                                 args.width, args.height, flags))
            scene_result[name] = summary
            images_by_variant[name] = images
        for stage in stages:
            off, basic = scene_result[f"{stage}_off_c1"], scene_result[f"{stage}_basic_c1"]
            scene_result[f"{stage}_off_basic_delta"] = {
                key: basic[key] - off[key] for key in
                ("temporal_merges_per_frame", "spatial_merges_per_frame", "effective_M")
            }
        tail_variants = ["temporal_spatial_off_c1", "temporal_spatial_basic_c1"]
        if not args.final_only:
            tail_variants.append("temporal_spatial_raytraced_c1")
        for name in tail_variants:
            scene_result[name]["tails"] = tail_stats(images_by_variant[name], reference)
        if scene_name == "occluded" and not args.final_only:
            scene_result["top_error_dump"] = diagnostic_dump(
                audit_binary, scene, args.output / "occluded-basic-dump.exr", reference, args.width, args.height,
                variants["temporal_spatial_basic_c1"])
        result["scenes"][scene_name] = scene_result
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
