#!/usr/bin/env python3
"""Rewrite third_party/openpbr_bsdf into headers nvcc will accept in device code.

Adobe's OpenPBR 1.1.1 advertises a CUDA backend, but it has never been through
nvcc: two properties of the vendored source make `openpbr.h` uncompilable as
shipped, and neither can be answered from the including side.

  1. 211 definitions carry no execution-space macro at all -- they are bare
     `float openpbr_average_fresnel(const float eta) {...}` at file scope
     (impl/openpbr_lobe_utils.h). GLSL, MSL and a CUDA *host* build are all fine
     with that; to nvcc each one is a `__host__` function, and the 87 that do
     carry OPENPBR_INLINE_FUNCTION (`__device__ inline`) call them. That is an
     error, not a warning.

  2. The eight lookup tables are declared OPENPBR_CONSTEXPR_GLOBAL, which the
     CUDA interop layer spells `static inline constexpr` -- a host global. Device
     code cannot read one.

Both are mechanical, so this script produces a patched copy of the tree at
configure/build time rather than editing the submodule. The generated directory
goes on the OPTIXIR include path *ahead* of the real one, so `openpbr.h` and
every header it includes relatively resolve to the rewritten copies while the
submodule stays byte-identical to upstream (last checked at 9edf806).

Fast-math can also move remapped random values and dot products a few ulps out
of their documented domains. The device copy clamps those inputs before the
OpenPBR Fresnel and lobe-selection math so a rare edge sample cannot become a
NaN/firefly.

The third incompatibility -- the CUDA interop's `using vec3 = float3`, which has
no three-argument constructor, no `operator[]` and no `.rgb` -- is *not* handled
here. It is answered from the including side, because the interop layer offers
OPENPBR_USE_CUSTOM_VEC_TYPES for exactly that: see
src/material/include/strelka/material/openpbr/openpbr_cuda_vec.h.

Usage: openpbr_device_headers.py <source-root> <output-root>
"""

import pathlib
import re
import shutil
import sys

# The return types every bare definition in the tree starts with. Counted rather
# than guessed: `float` 98, `vec3` 31, `void` 18, `bool` 13, `openpbr_complex`
# 11 (a macro for vec2), `OpenPBR_*` 24, `int` 3, `vec2`/`vec4` 1 each.
TYPE = r"(?:float|vec2|vec3|vec4|void|bool|int|openpbr_complex|OpenPBR_\w+|OPENPBR_UINT\d+)"

# A definition whose return type and name are on one line.
ONE_LINE = re.compile(r"^" + TYPE + r"\s+\w+\s*\(")
# ...and the two-line form, where the return type sits alone above the name.
TYPE_ALONE = re.compile(r"^" + TYPE + r"\s*$")
NAME_BELOW = re.compile(r"^\w+\s*\(")

# The two table-declaration macros, as the CUDA interop defines them.
TABLE_MACRO = re.compile(r"^#define (OPENPBR_(?:MAYBE_)?CONSTEXPR_GLOBAL) static inline constexpr\s*$")

NUMERICAL_REWRITES = (
    (
        '    OPENPBR_ASSERT(rand >= 0.0f, "It is assumed that the random number is already >= 0");',
        '    if (!(rand >= 0.0f))\n        rand = 0.0f;',
    ),
    (
        '    OPENPBR_ASSERT(cos_theta_i >= 0.0f, "Fresnel input cosine must be non-negative");',
        '    const float safe_cos_theta_i = cos_theta_i >= 0.0f ? min(cos_theta_i, 1.0f) : 0.0f;',
    ),
    (
        '    const float sin_theta_i_squared = 1.0f - openpbr_square(cos_theta_i);',
        '    const float sin_theta_i_squared = 1.0f - openpbr_square(safe_cos_theta_i);',
    ),
    (
        '    const float eta_t_over_eta_i_cos_theta_i = eta_t_over_eta_i * cos_theta_i;',
        '    const float eta_t_over_eta_i_cos_theta_i = eta_t_over_eta_i * safe_cos_theta_i;',
    ),
    (
        '    const float Rs = openpbr_square((cos_theta_i - eta_t_over_eta_i_cos_theta_t) / (cos_theta_i + eta_t_over_eta_i_cos_theta_t));',
        '    const float Rs = openpbr_square((safe_cos_theta_i - eta_t_over_eta_i_cos_theta_t) /\n'
        '                                    (safe_cos_theta_i + eta_t_over_eta_i_cos_theta_t));',
    ),
    (
        '    OPENPBR_ASSERT(cos_theta >= 0.0f, "F82-tint input cosine must be non-negative");',
        '    const float safe_cos_theta = cos_theta >= 0.0f ? min(cos_theta, 1.0f) : 0.0f;',
    ),
    (
        '    const float one_minus_cos_theta = 1.0f - cos_theta;',
        '    const float one_minus_cos_theta = 1.0f - safe_cos_theta;',
    ),
    (
        '    const vec3 offset_from_r = (white_minus_r - b * cos_theta * one_minus_cos_theta) * openpbr_fifth_power(one_minus_cos_theta);',
        '    const vec3 offset_from_r = (white_minus_r - b * safe_cos_theta * one_minus_cos_theta) *\n'
        '                               openpbr_fifth_power(one_minus_cos_theta);',
    ),
)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    if not (src / "openpbr.h").is_file():
        print(f"error: {src} does not look like the openpbr_bsdf root", file=sys.stderr)
        return 1

    # Rebuilt whole, not merged: a stale header left behind by a submodule update
    # that deleted its original would silently keep being included.
    if dst.exists():
        shutil.rmtree(dst)

    patched = 0
    numerical = 0
    for path in sorted(src.rglob("*.h")):
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for i, line in enumerate(lines):
            table = TABLE_MACRO.match(line)
            if table:
                lines[i] = f"#define {table.group(1)} __device__ static constexpr\n"
                patched += 1
                continue
            two_line = TYPE_ALONE.match(line) and i + 1 < len(lines) and NAME_BELOW.match(lines[i + 1])
            if ONE_LINE.match(line) or two_line:
                lines[i] = "__device__ inline " + line
                patched += 1
        text = "".join(lines)
        if path.name == "openpbr_lobe_utils.h":
            for before, after in NUMERICAL_REWRITES:
                count = text.count(before)
                if count != 1:
                    raise RuntimeError(f"OpenPBR numerical guard matched {count} times: {before}")
                text = text.replace(before, after)
                numerical += 1
        out.write_text(text, encoding="utf-8")

    # A count that drops after a submodule update means the regexes stopped
    # matching a form upstream now uses, and the build will fail with "calling a
    # __host__ function from a __device__ function" rather than anything that
    # names this script. Printed so the number is in the build log either way.
    print(f"openpbr device headers: patched {patched} declarations and {numerical} numerical guards into {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
