#include <optix_types.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/compatibility.hpp>

struct Params
{
    uchar4* image;
    unsigned int image_width;
    unsigned int image_height;
    float3 cam_eye;
    float3 cam_u, cam_v, cam_w;
    glm::float4x4 clipToView;
    glm::float4x4 viewToWorld;

    OptixTraversableHandle handle;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};
