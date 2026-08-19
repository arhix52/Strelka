#!/usr/bin/env python3
"""Render reproducible SHARC off/on pairs and grade both against a PT reference.

Example:
    tools/sharc_ab.py scenes/validation/cornell_box/cornell_box.toml \
        --spp 64 --reference-spp 512 --out /tmp/strelka-sharc-ab

The generated TOMLs and logs are retained in --out. Images are linear EXR:
tonemapping, gamma, denoising, and upscaling are disabled by the rewrite.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
np = None
load_exr = None


def load_dependencies():
    global np, load_exr
    try:
        import numpy as numpy_module

        sys.path.insert(0, str(SCRIPT_DIR / "feature_tests"))
        from exr_io import load_exr as load_exr_function
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for EXR grading; use a venv with `pip install numpy`"
        ) from exc
    np = numpy_module
    load_exr = load_exr_function


def set_key(text, section, key, value):
    """Replace a scalar TOML key or append it to an existing section."""
    section_match = re.search(rf"(?m)^\[{re.escape(section)}\]\s*$", text)
    if not section_match:
        return text.rstrip() + f"\n\n[{section}]\n{key} = {value}\n"
    section_end = re.search(r"(?m)^\[", text[section_match.end():])
    end = section_match.end() + section_end.start() if section_end else len(text)
    body = text[section_match.end():end]
    key_match = re.search(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$", body)
    if key_match:
        begin = section_match.end() + key_match.start()
        finish = section_match.end() + key_match.end()
        replacement = key_match.group(1) + value
        return text[:begin] + replacement + text[finish:]
    return text[:end].rstrip() + f"\n{key} = {value}\n\n" + text[end:].lstrip("\n")


def quoted(path):
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def resolve_scene(config_path, parsed):
    scene = Path(parsed["scene"]["path"])
    if scene.is_absolute():
        return scene.resolve()
    sibling = config_path.parent / scene.name
    return (sibling if sibling.exists() else config_path.parent / scene).resolve()


def make_config(
    source,
    output,
    spp,
    sharc,
    debug,
    sharc_capacity=None,
    sharc_update_downscale=None,
    sharc_options=None,
):
    parsed = tomllib.loads(source.read_text())
    text = source.read_text()
    text = set_key(text, "scene", "path", quoted(resolve_scene(source, parsed)))
    text = set_key(text, "output", "path", quoted(output))
    text = set_key(text, "render", "spp", str(spp))
    text = set_key(text, "render", "spp_per_launch", "1")
    text = set_key(text, "render", "sharc", "true" if sharc else "false")
    text = set_key(text, "render", "sharc_debug", str(debug if sharc else 0))
    if sharc_capacity is not None:
        text = set_key(text, "render", "sharc_capacity", str(sharc_capacity))
    if sharc_update_downscale is not None:
        text = set_key(
            text,
            "render",
            "sharc_update_downscale",
            str(sharc_update_downscale),
        )
    if sharc:
        for key, value in sharc_options or ():
            text = set_key(text, "render", key, value)
    text = set_key(text, "render", "denoise", "false")
    text = set_key(text, "render", "upscale", "false")
    text = set_key(text, "render", "profile_stages", "false")
    text = set_key(text, "tonemap", "type", '"none"')
    text = set_key(text, "tonemap", "gamma", "0.0")
    return text


def run(cli, config, log_path, timeout):
    proc = subprocess.run(
        [str(cli), "--config", str(config)],
        cwd=cli.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log)
    if proc.returncode:
        tail = "\n".join(log.strip().splitlines()[-8:])
        raise RuntimeError(f"{config.name}: exit {proc.returncode}\n{tail}")
    samples = [float(v) for v in re.findall(r"([0-9]+(?:\.[0-9]+)?) ms/sample", log)]
    steady = samples[len(samples) // 2:] if samples else []
    return statistics.median(steady) if steady else float("nan"), log


def error(test, reference):
    if test.shape != reference.shape:
        raise ValueError(f"shape mismatch: {test.shape} != {reference.shape}")
    delta = test.astype(np.float64) - reference.astype(np.float64)
    reference_mean = float(np.abs(reference).mean())
    return (
        float(np.abs(delta).mean()) / max(reference_mean, 1e-9),
        float(np.sqrt(np.square(delta).mean())),
        float(np.percentile(np.abs(delta), 95)),
    )


def last_sharc_stats(log):
    matches = re.findall(
        r"SHARC stats: insertions=(\d+) failed=(\d+) collisions=(\d+) "
        r"queries=(\d+) hits=(\d+) hit_rate=([0-9.]+)% evictions=(\d+) "
        r"segment_rejects=(\d+) footprint_rejects=(\d+)"
        r"(?: accumulation_clamps=(\d+) nonfinite_rejects=(\d+) "
        r"radiance_bits=(\d+) sample_bits=(\d+))?",
        log,
    )
    return matches[-1] if matches else None


def parse_sharc_option(option):
    if "=" not in option:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, value = option.split("=", 1)
    if not re.fullmatch(r"sharc_[a-z0-9_]+", key):
        raise argparse.ArgumentTypeError("KEY must be a render sharc_* setting")
    if not value or any(character in value for character in "\r\n"):
        raise argparse.ArgumentTypeError("VALUE must be a TOML scalar")
    return key, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="scene TOML files")
    parser.add_argument("--cli", type=Path, default=Path("build/Release/StrelkaCLI"))
    parser.add_argument("--out", type=Path, default=Path("build/Release/sharc_ab"))
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--reference-spp", type=int, default=512)
    parser.add_argument(
        "--sharc-capacity",
        type=int,
        help="override cache entries for SHARC-on and diagnostic runs",
    )
    parser.add_argument(
        "--sharc-update-downscale",
        type=int,
        help="override sparse update block size for SHARC-on and diagnostic runs",
    )
    parser.add_argument(
        "--sharc-option",
        action="append",
        default=[],
        type=parse_sharc_option,
        metavar="KEY=VALUE",
        help="override a render sharc_* setting in SHARC-on and diagnostic runs; repeatable",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--reuse", action="store_true", help="reuse existing EXRs")
    parser.add_argument(
        "--debug",
        type=int,
        default=4,
        help="extra diagnostic run: 1-3 visualize, 4 collects stats without changing radiance; 0 disables",
    )
    args = parser.parse_args()
    if args.sharc_capacity is not None and args.sharc_capacity <= 0:
        parser.error("--sharc-capacity must be positive")
    if args.sharc_update_downscale is not None and args.sharc_update_downscale <= 0:
        parser.error("--sharc-update-downscale must be positive")

    try:
        load_dependencies()
    except RuntimeError as exc:
        parser.error(str(exc))

    cli = args.cli.resolve()
    if not os.access(cli, os.X_OK):
        parser.error(f"not executable: {cli}")
    # The renderer runs with the CLI directory as its working directory so its
    # shader/runtime assets resolve normally. Keep generated configs and output
    # paths absolute; otherwise the default relative --out would be interpreted
    # a second time under build/Release.
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    print("scene                     off ms   on ms speedup  off rel   on rel quality")
    print("-" * 82)
    failed = False
    for source in args.configs:
        source = source.resolve()
        stem = source.stem
        paths = {
            "reference": args.out / f"{stem}_reference.exr",
            "off": args.out / f"{stem}_off.exr",
            "on": args.out / f"{stem}_on.exr",
        }
        spp = {"reference": args.reference_spp, "off": args.spp, "on": args.spp}
        timings = {}
        logs = {}
        try:
            for mode in ("reference", "off", "on"):
                config = args.out / f"{stem}_{mode}.toml"
                config.write_text(
                    make_config(
                        source,
                        paths[mode],
                        spp[mode],
                        mode == "on",
                        0,
                        args.sharc_capacity if mode == "on" else None,
                        args.sharc_update_downscale if mode == "on" else None,
                        args.sharc_option if mode == "on" else None,
                    )
                )
                log_path = args.out / f"{stem}_{mode}.log"
                if args.reuse and paths[mode].exists() and log_path.exists():
                    log = log_path.read_text()
                    samples = [
                        float(v)
                        for v in re.findall(r"([0-9]+(?:\.[0-9]+)?) ms/sample", log)
                    ]
                    timings[mode] = statistics.median(samples[len(samples) // 2:])
                    logs[mode] = log
                else:
                    timings[mode], logs[mode] = run(cli, config, log_path, args.timeout)

            reference = load_exr(str(paths["reference"]))
            off_error = error(load_exr(str(paths["off"])), reference)
            on_error = error(load_exr(str(paths["on"])), reference)
            speedup = timings["off"] / timings["on"]
            quality = off_error[0] / max(on_error[0], 1e-9)
            print(
                f"{stem:24s} {timings['off']:7.2f} {timings['on']:7.2f} "
                f"{speedup:7.3f} {off_error[0]:8.4f} {on_error[0]:8.4f} {quality:7.3f}"
            )
            stats = None
            if args.debug:
                diagnostic_output = args.out / f"{stem}_diagnostic.exr"
                diagnostic_config = args.out / f"{stem}_diagnostic.toml"
                diagnostic_log = args.out / f"{stem}_diagnostic.log"
                diagnostic_config.write_text(
                    make_config(
                        source,
                        diagnostic_output,
                        args.spp,
                        True,
                        args.debug,
                        args.sharc_capacity,
                        args.sharc_update_downscale,
                        args.sharc_option,
                    )
                )
                if args.reuse and diagnostic_output.exists() and diagnostic_log.exists():
                    diagnostic_text = diagnostic_log.read_text()
                else:
                    _, diagnostic_text = run(
                        cli, diagnostic_config, diagnostic_log, args.timeout
                    )
                stats = last_sharc_stats(diagnostic_text)
            if stats:
                (
                    insertions,
                    failed_insertions,
                    collisions,
                    queries,
                    hits,
                    hit_rate,
                    evictions,
                    segment_rejects,
                    footprint_rejects,
                    accumulation_clamps,
                    nonfinite_rejects,
                    radiance_bits,
                    sample_bits,
                ) = stats
                gated_candidates = int(queries) + int(segment_rejects) + int(footprint_rejects)
                effective_hit_rate = 100.0 * int(hits) / max(gated_candidates, 1)
                occupancy = None
                if args.sharc_capacity:
                    occupancy = 100.0 * int(insertions) / args.sharc_capacity
                print(
                    "  cache: insertions=%s failed=%s collisions=%s queries=%s "
                    "hits=%s hit-rate=%s%% evictions=%s segment-rejects=%s "
                    "footprint-rejects=%s" % stats[:9]
                )
                if accumulation_clamps:
                    print(
                        "  accumulation: clamps=%s nonfinite-rejects=%s "
                        "radiance-bits=%s/32 sample-bits=%s/32"
                        % (
                            accumulation_clamps,
                            nonfinite_rejects,
                            radiance_bits,
                            sample_bits,
                        )
                    )
                detail = f"  effective hit-rate={effective_hit_rate:.1f}%"
                if occupancy is not None:
                    detail += f" occupancy={occupancy:.1f}%"
                print(detail)
            print(
                f"  RMSE off/on {off_error[1]:.6f}/{on_error[1]:.6f}; "
                f"p95 abs off/on {off_error[2]:.6f}/{on_error[2]:.6f}"
            )
        except Exception as exc:
            failed = True
            print(f"{stem:24s} FAILED: {exc}")

    print(f"\nArtifacts: {args.out.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
