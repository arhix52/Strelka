#include "tonemap.h"

namespace oka
{
Tonemap::Tonemap(const SharedContext& ctx) : TonemapBase(ctx)
{
}
Tonemap::~Tonemap()
{
}
void Tonemap::initialize()
{
    TonemapBase::initialize("shaders/tonemap.hlsl");
}
void Tonemap::execute(VkCommandBuffer& cmd, const TonemapDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex)
{
    auto& param = mShaderParamFactory.getNextShaderParameters(frameIndex);
    {
        param.setConstants(desc.constants);
        param.setTexture("input", mSharedCtx.mResManager->getView(desc.input));
        param.setTexture("output", mSharedCtx.mResManager->getView(desc.output));
    }
    int frameVersion = frameIndex % MAX_FRAMES_IN_FLIGHT;
    Result res = updatePipeline(frameVersion);
    assert(res == Result::eOk);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, getPipeline(frameVersion));
    VkDescriptorSet descSet = param.getDescriptorSet(frameVersion);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, getPipeLineLayout(frameVersion), 0, 1, &descSet, 0, nullptr);
    const uint32_t dispX = (width + 15) / 16;
    const uint32_t dispY = (height + 15) / 16;
    vkCmdDispatch(cmd, dispX, dispY, 1);
}
} // namespace oka
