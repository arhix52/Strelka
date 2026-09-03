#ifndef STRELKA_MATERIAL_OPENPBR_SHIM_H
#define STRELKA_MATERIAL_OPENPBR_SHIM_H

// The one place `openpbr.h` (third_party/openpbr_bsdf, Adobe's OpenPBR 1.1.1) is
// allowed to be included. It is vendored unmodified, and it makes four
// assumptions that do not hold inside Strelka's shaders. Every one of them is
// answered here rather than at each use site, because three different compilers
// consume this header and a fix applied unevenly is worse than no fix.
//
//  1. Lookup tables. `openpbr.h` can read its eight precomputed tables either
//     from constant arrays it carries itself, or from textures the renderer
//     binds. Texture mode is the faster one -- and it is unreachable on Metal:
//     the OPENPBR_SAMPLE_*_TEXTURE macros are expanded deep inside openpbr's own
//     inline functions, which take no renderer context, so the texture handles
//     would have to come from a program-scope variable. Metal has no mutable
//     ones (a program-scope variable must live in the constant address space and
//     be initialised at compile time), and a texture handle only exists once a
//     binding does. Threading a context parameter through would mean editing the
//     vendored source, which is the thing this file exists to avoid.
//
//     So: array mode, on all three targets. That also keeps the CPU unit tests
//     an exact oracle for what the two GPUs compute, and keeps Metal and OptiX
//     doing bit-comparable arithmetic instead of one interpolating in ALU and
//     the other in the texture unit. It costs ~264 KB of table data in the
//     metallib -- data, not instructions, so it does not land in the
//     instruction cache the wavefront kernel is bound by (docs/open-perf.md).
//
//  2. saturate(). The CPU and CUDA interop layers define `saturate(float)` and
//     `saturate(vec3)` themselves. Strelka already has both on CUDA (sutil's
//     vec_math_adv.h) and `saturate(float)` on the host (material_math.h), and a
//     second definition of the same signature in one translation unit is a hard
//     redefinition error, not an overload. Suppress openpbr's copies and supply
//     the single overload the host branch is missing.
//
//     Metal is the exception: there openpbr uses the language built-in and
//     defines nothing, while material_math.h defines `saturate` as a *macro*.
//     A function-like macro rewriting openpbr's call sites to metal::saturate is
//     both harmless and correct, so Metal needs nothing here.
//
//  3. GLSL-compatibility macros. The CUDA interop layer unconditionally defines
//     `mix`, `all`, `any`, `equal`, `notEqual`, `greaterThan` and
//     `greaterThanEqual` as bare function-like macros. They do not stay inside
//     openpbr: `mix` in particular would rewrite the *declaration* of
//     material_math.h's own `float3 mix(float3, float3, float)` into syntax
//     garbage in any translation unit that includes both. They are undefined
//     again at the bottom of this file, which is why include order stops
//     mattering.
//
//  4. `inline` variables. openpbr's MSL layer spells program-scope constants
//     `static constexpr constant inline`, and MSL is C++14-based, so each one
//     draws a -Wc++17-extensions warning -- 56 of them, under the -Wall the
//     Metal build already uses. They are suppressed around the include only,
//     rather than by adding a flag in src/shaders/CMakeLists.txt, so the
//     warning stays live for Strelka's own code.

//  5. Linkage. 191 of openpbr's ~280 function definitions carry no linkage
//     macro at all -- they are bare `float openpbr_average_fresnel(float) {...}`
//     at file scope. In GLSL, and in a Metal or CUDA shader module, that is
//     fine: each is one translation unit. In C++ it is not. Two host TUs that
//     both include openpbr.h fail to link with ~200 duplicate symbols, and the
//     unit test suite compiles 79 TUs into one binary.
//
//     So on the host the whole library goes into an anonymous namespace, giving
//     every one of those definitions internal linkage. Each TU gets its own
//     copy, which costs the test binary some size and costs the shaders nothing
//     (they never take this branch -- a shader module *is* a single TU, and
//     there the definitions must keep external linkage so the compiler treats
//     them normally).
//
//     Note this is a host-only answer. nvcc has the same problem for a different
//     reason -- a bare definition is a __host__ function there, and the 87 that
//     do carry OPENPBR_INLINE_FUNCTION (`__device__ inline`) call them, which is
//     an error -- and it cannot be answered from here at all. That one is
//     handled outside the preprocessor, by tools/openpbr_device_headers.py; see
//     (6) below.
//
//  6. CUDA, which needs two things this header cannot express.
//
//     The execution space of those bare definitions is one, and it is not a
//     property any macro reaches: the fix is a build-time rewrite of the
//     vendored headers into ${CMAKE_BINARY_DIR}, put ahead of the submodule on
//     the OPTIXIR include path (src/shaders/CMakeLists.txt). The same rewrite
//     moves the eight lookup tables from `static inline constexpr` -- a host
//     global -- to `__device__ static constexpr`.
//
//     The other is the interop layer's `using vec3 = float3`, which cannot work
//     because CUDA's float3 is a bare aggregate with no three-argument
//     constructor, no operator[] and no .rgb. openpbr offers
//     OPENPBR_USE_CUSTOM_VEC_TYPES for precisely this, so the answer *is* in the
//     preprocessor and it is included below.
//
//     Neither is a workaround for Strelka: openpbr's CUDA backend has evidently
//     never been compiled. Metal and the host both work as shipped.

// (1) -- see above.
#ifndef OPENPBR_USE_TEXTURE_LUTS
#    define OPENPBR_USE_TEXTURE_LUTS 0
#endif

// (5) -- the host is the only backend that compiles more than one translation
// unit against this header.
#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__) && !defined(__CUDACC__)
#    define STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE 1
#endif

#include <strelka/material/material_math.h>

// (6) -- see above. Before openpbr.h, and before the interop layer it pulls in,
// because it defines OPENPBR_USE_CUSTOM_VEC_TYPES.
#if defined(__CUDACC__)
#    include <strelka/material/openpbr/openpbr_cuda_vec.h>
#endif

// (2) -- see above.
#if !defined(__METAL_VERSION__)
#    define OPENPBR_USE_CUSTOM_SATURATE 1
#    if !defined(__CUDA_ARCH__) && !defined(__CUDACC__) && !defined(STRELKA_MATERIAL_CUDA_HOST)
// Host build only: material_math.h has saturate(float), nothing has the vec3
// overload openpbr calls. On CUDA both come from sutil/vec_math_adv.h.
#        ifndef STRELKA_MATERIAL_HAS_SATURATE_FLOAT3
#            define STRELKA_MATERIAL_HAS_SATURATE_FLOAT3
DEVICE_FUNC float3 saturate(float3 v)
{
    return { saturate(v.x), saturate(v.y), saturate(v.z) };
}
#        endif
#    endif
#endif

// (5) -- see above.
#if defined(STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE)
// The two standard headers openpbr's C++ layer pulls in, hoisted to file scope
// *before* the anonymous namespace opens. <cstdint> declares names in namespace
// std, and including it inside an unnamed namespace would declare a private
// `std` instead. Both self-guard, so the include inside openpbr.h then vanishes.
#    include <cassert>
#    include <cstdint>
#endif

// (4) -- see above.
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wc++17-extensions"
#endif
#if defined(STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE)
// An unnamed namespace in a header is normally a defect -- it gives every
// includer its own copy of everything, which is usually an accident. Here it is
// the point: see (5) above. The alternative is ~200 duplicate symbols at link
// time, and the vendored header cannot be edited to add `inline`.
// NOLINTBEGIN(cert-dcl59-cpp, misc-anonymous-namespace-in-header)
namespace
{
#endif

#include "openpbr.h"

#if defined(STRELKA_OPENPBR_NEEDS_INTERNAL_LINKAGE)
} // anonymous namespace
// NOLINTEND(cert-dcl59-cpp, misc-anonymous-namespace-in-header)
#endif
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

// (3) -- see above. Undefining names openpbr did not define is legal and a no-op,
// so this needs no per-backend guard.
#undef mix
#undef all
#undef any
#undef equal
#undef notEqual
#undef greaterThan
#undef greaterThanEqual

#endif // STRELKA_MATERIAL_OPENPBR_SHIM_H
