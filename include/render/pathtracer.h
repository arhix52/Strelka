#pragma once
#include "common.h"
#include "computepass.h"
#include "gbuffer.h"
#include "pathtracerparam.h"

namespace oka
{

using PathTracerBase = ComputePass<PathTracerParam>;
class PathTracer : public PathTracerBase
{
public:
    VkSampler mMdlSampler;
    std::vector<Image*> mMatTextures;
    std::string mShaderCode;

    VkSampler mCubeMapSampler = VK_NULL_HANDLE;
    Image* mCubeMap = VK_NULL_HANDLE;

    PathTracer(const SharedContext& ctx, std::string& shaderCode);
    ~PathTracer();

    void execute(VkCommandBuffer& cmd, const PathTracerDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex);

    void initialize();
};
} // namespace oka
