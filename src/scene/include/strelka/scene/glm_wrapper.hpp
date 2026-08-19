#pragma once

#ifndef GLM_FORCE_SILENT_WARNINGS
#    define GLM_FORCE_SILENT_WARNINGS
#endif
#ifndef GLM_LANG_STL11_FORCED
#    define GLM_LANG_STL11_FORCED
#endif
#ifndef GLM_ENABLE_EXPERIMENTAL
#    define GLM_ENABLE_EXPERIMENTAL
#endif
#ifndef GLM_FORCE_CTOR_INIT
#    define GLM_FORCE_CTOR_INIT
#endif
#ifndef GLM_FORCE_RADIANS
#    define GLM_FORCE_RADIANS
#endif
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#    define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif

#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// Cg-style names (glm::float3, glm::float4x4, ...) used to come from
// gtx/compatibility.hpp, which #includes glm.hpp and therefore every GLM
// header. The aliases are the only part of that extension this tree needs, so
// they live here. gtx/hash.hpp is unused (no glm key in any unordered_map).
// Translate/scale/perspective stay in <glm/gtc/matrix_transform.hpp> at the
// call sites — pulling that into every camera.h / common.h consumer was free
// until GLM 1.0 made the experimental headers much heavier.
namespace glm
{
using float2 = vec<2, float, highp>;
using float3 = vec<3, float, highp>;
using float4 = vec<4, float, highp>;
using float3x3 = mat<3, 3, float, highp>;
using float3x4 = mat<3, 4, float, highp>;
using float4x4 = mat<4, 4, float, highp>;
} // namespace glm

// Packed layouts: Metal/OptiX vertex and curve buffers address float3 as 12
// bytes. GLM 1.0 can emit 16-byte aligned vec3 if GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
// is on; catch that at compile time rather than as a silent GPU stride mismatch.
static_assert(sizeof(glm::float3) == 12, "glm::float3 must stay packed (12 bytes)");
static_assert(sizeof(glm::float4) == 16, "glm::float4 must stay 16 bytes");
static_assert(sizeof(glm::mat4) == 64, "glm::mat4 must stay 64 bytes");
