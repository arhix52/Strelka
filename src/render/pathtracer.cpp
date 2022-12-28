#include "pathtracer.h"
namespace oka
{

PathTracer::PathTracer(const SharedContext& ctx, std::string& shaderCode) : PathTracerBase(ctx), mShaderCode(shaderCode)
{
}

PathTracer::~PathTracer()
{
}

void PathTracer::execute(VkCommandBuffer& cmd, const PathTracerDesc& desc, uint32_t width, uint32_t height, uint64_t frameIndex)
{
    assert(height);
    auto& param = mShaderParamFactory.getNextShaderParameters(frameIndex);
    {
        param.setConstants(desc.constants);
        mMdlSampler = desc.matSampler;
        mMatTextures = desc.matTextures;

        param.setBuffer("sampleBuffer", mSharedCtx.mResManager->getVkBuffer(desc.sampleBuffer));
        // mShaderParams.setBuffer("compositingBuffer", mSharedCtx.mResManager->getVkBuffer(desc.compositingBuffer));

        // mdl_ro_data_segment, mdl_argument_block, mdl_resource_infos, mdl_textures_2d, mdl_sampler_tex
        param.setBuffer("mdl_ro_data_segment", mSharedCtx.mResManager->getVkBuffer(desc.mdl_ro_data_segment));
        param.setBuffer("mdl_argument_block", mSharedCtx.mResManager->getVkBuffer(desc.mdl_argument_block));
        param.setBuffer("mdl_resource_infos", mSharedCtx.mResManager->getVkBuffer(desc.mdl_resource_infos));
        param.setBuffer("mdlMaterials", mSharedCtx.mResManager->getVkBuffer(desc.mdl_mdlMaterial));

        param.setTextures("mdl_textures_2d", mMatTextures);
        // mShaderParams.setTextures3d("mdl_textures_3d", mMatTextures);
        param.setSampler("mdl_sampler_tex", mMdlSampler);

        // param.setTexture("gbWPos", mSharedCtx.mResManager->getView(desc.gbuffer->wPos));
        // param.setTexture("gbNormal", mSharedCtx.mResManager->getView(desc.gbuffer->normal));
        // param.setTexture("gbTangent", mSharedCtx.mResManager->getView(desc.gbuffer->tangent));
        // param.setTexture("gbInstId", mSharedCtx.mResManager->getView(desc.gbuffer->instId));
        // param.setTexture("gbUV", mSharedCtx.mResManager->getView(desc.gbuffer->uv));

        param.setBuffer("bvhNodes", mSharedCtx.mResManager->getVkBuffer(desc.bvhNodes));
        param.setBuffer("vb", mSharedCtx.mResManager->getVkBuffer(desc.vb));
        param.setBuffer("ib", mSharedCtx.mResManager->getVkBuffer(desc.ib));
        param.setBuffer("instanceConstants", mSharedCtx.mResManager->getVkBuffer(desc.instanceConst));
        param.setBuffer("lights", mSharedCtx.mResManager->getVkBuffer(desc.lights));

        param.setCubeMap("cubeMap", mSharedCtx.mResManager->getView(mCubeMap));
        param.setSampler("cubeMapSampler", mCubeMapSampler);
    }
    int frameVersion = frameIndex % MAX_FRAMES_IN_FLIGHT;
    Result res = updatePipeline(frameVersion);
    assert(res == Result::eOk);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, getPipeline(frameVersion));
    VkDescriptorSet descSet = param.getDescriptorSet(frameIndex);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, getPipeLineLayout(frameVersion), 0, 1, &descSet, 0, nullptr);
    const uint32_t dispX = (width + 255) / 256;
    const uint32_t dispY = 1;
    vkCmdDispatch(cmd, dispX, dispY, 1);
}

void PathTracer::initialize()
{
    // sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkResult res = vkCreateSampler(mSharedCtx.mDevice, &samplerInfo, nullptr, &mCubeMapSampler);
    if (res != VK_SUCCESS)
    {
        // error
        assert(0);
    }

    std::string texPath[6] = { "misc/skybox/right.jpg",  "misc/skybox/left.jpg",  "misc/skybox/top.jpg",
                               "misc/skybox/bottom.jpg", "misc/skybox/front.jpg", "misc/skybox/back.jpg" };
    oka::TextureManager::Texture cubeMapTex = mSharedCtx.mTextureManager->createCubeMapTextureImage(texPath);
    mCubeMap = cubeMapTex.textureImage;
    PathTracerBase::initializeFromCode(mShaderCode.c_str());
}

} // namespace oka
