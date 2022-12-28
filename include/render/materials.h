#pragma once

#ifdef __cplusplus
#    define GLM_FORCE_SILENT_WARNINGS
#    define GLM_LANG_STL11_FORCED
#    define GLM_ENABLE_EXPERIMENTAL
#    define GLM_FORCE_CTOR_INIT
#    define GLM_FORCE_RADIANS
#    define GLM_FORCE_DEPTH_ZERO_TO_ONE
#    include <glm/glm.hpp>
#    include <glm/gtx/compatibility.hpp>
#    define float4 glm::float4
#    define float3 glm::float3
#    define uint glm::uint
#endif

struct MdlMaterial
{
    int arg_block_offset = 0; // in bytes
    int ro_data_segment_offset = 0; // in bytes
    int functionId = 0;
    int pad1;
};

/// Information passed to GPU for mapping id requested in the runtime functions to buffer
/// views of the corresponding type.
struct Mdl_resource_info
{
    // index into the tex2d, tex3d, ... buffers, depending on the type requested
    uint gpu_resource_array_start;
    uint pad0;
    uint pad1;
    uint pad2;
};

#ifdef __cplusplus
#    undef float4
#    undef float3
#    undef uint
#endif
