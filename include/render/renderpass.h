#pragma once
#include "debugUtils.h"
#include "gbuffer.h"

#include <scene/scene.h>
#include <vulkan/vulkan.h>

#include <array>
#include <resourcemanager.h>
#include <vector>

namespace nevk
{
class RenderPass
{
private:
    struct UniformBufferObject
    {
        alignas(16) glm::mat4 viewToProj;
        alignas(16) glm::mat4 worldToView;
        alignas(16) glm::mat4 lightSpaceMatrix;
        alignas(16) glm::float4 lightPosition;
        alignas(16) glm::float3 CameraPos;
        float pad;
        alignas(16) uint32_t debugView;
    };

    struct InstancePushConstants
    {
        int32_t instanceId = -1;
    };

    static constexpr int MAX_FRAMES_IN_FLIGHT = 3;

    VkDevice mDevice;
    VkPipeline mPipelineOpaque;
    VkPipeline mPipelineTransparent;
    VkPipelineLayout mPipelineLayoutOpaque;
    VkPipelineLayout mPipelineLayoutTransparent;
    VkRenderPass mRenderPass;
    VkDescriptorSetLayout mDescriptorSetLayout;

    bool mEnableValidation = false;

    void beginLabel(VkCommandBuffer cmdBuffer, const char* labelName, const glm::float4& color)
    {
        if (mEnableValidation)
        {
            nevk::debug::beginLabel(cmdBuffer, labelName, color);
        }
    }

    void endLabel(VkCommandBuffer cmdBuffer)
    {
        if (mEnableValidation)
        {
            nevk::debug::endLabel(cmdBuffer);
        }
    }

    void updateDescriptorSets(uint32_t descSetIndex);

    VkShaderModule mVS, mPS;

    ResourceManager* mResManager;
    VkDescriptorPool mDescriptorPool;
    std::vector<Buffer*> uniformBuffers;

    std::vector<VkSampler> mTextureSampler;
    VkSampler mShadowSampler = VK_NULL_HANDLE;

    void createRenderPass();

    void createDescriptorSetLayout();
    void createDescriptorSets(VkDescriptorPool& descriptorPool);

    void createConstantBuffers();

    std::vector<VkDescriptorSet> mDescriptorSets;

    std::vector<VkFramebuffer> mFrameBuffers;
    VkFormat mFrameBufferFormat;

    VkFormat mDepthBufferFormat;
    uint32_t mWidth, mHeight;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Scene::Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions()
    {
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions = {};

        VkVertexInputAttributeDescription attributeDescription = {};

        attributeDescription.binding = 0;
        attributeDescription.location = 0;
        attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescription.offset = offsetof(Scene::Vertex, pos);
        attributeDescriptions.emplace_back(attributeDescription);

        attributeDescription.binding = 0;
        attributeDescription.location = 1;
        attributeDescription.format = VK_FORMAT_R32_UINT;
        attributeDescription.offset = offsetof(Scene::Vertex, tangent);
        attributeDescriptions.emplace_back(attributeDescription);

        attributeDescription.binding = 0;
        attributeDescription.location = 2;
        attributeDescription.format = VK_FORMAT_R32_UINT;
        attributeDescription.offset = offsetof(Scene::Vertex, normal);
        attributeDescriptions.emplace_back(attributeDescription);

        attributeDescription.binding = 0;
        attributeDescription.location = 3;
        attributeDescription.format = VK_FORMAT_R32_UINT;
        attributeDescription.offset = offsetof(Scene::Vertex, uv);
        attributeDescriptions.emplace_back(attributeDescription);

        return attributeDescriptions;
    }

    VkShaderModule createShaderModule(const char* code, uint32_t codeSize);

public:
    int imageViewCounter = 0;

    std::vector<VkImageView> mTextureImageView;
    VkImageView mShadowImageView = VK_NULL_HANDLE;
    //VkBuffer mMaterialBuffer = VK_NULL_HANDLE;
    VkBuffer mInstanceBuffer = VK_NULL_HANDLE;

    bool needDesciptorSetUpdate[MAX_FRAMES_IN_FLIGHT] = {false, false, false};

    VkPipelineLayout createGraphicsPipelineLayout();

    VkPipeline createGraphicsPipeline(VkShaderModule& vertShaderModule, VkShaderModule& fragShaderModule, VkPipelineLayout pipelineLayout, uint32_t width, uint32_t height, bool isTransparent);

    void createFrameBuffers(std::vector<VkImageView>& imageViews, VkImageView& depthImageView, uint32_t width, uint32_t height);

    void setFrameBufferFormat(VkFormat format)
    {
        mFrameBufferFormat = format;
    }

    void setDepthBufferFormat(VkFormat format)
    {
        mDepthBufferFormat = format;
    }

    void setShadowImageView(VkImageView shadowImageView);
    void setTextureImageView(const std::vector<VkImageView>& textureImageView);
    void setTextureSampler(const std::vector<VkSampler>& textureSampler);
    void setShadowSampler(VkSampler shadowSampler);
    void setMaterialBuffer(VkBuffer materialBuffer);
    void setInstanceBuffer(VkBuffer instanceBuffer);

    void init(VkDevice& device, bool enableValidation, const char* vsCode, uint32_t vsCodeSize, const char* psCode, uint32_t psCodeSize, VkDescriptorPool descpool, ResourceManager* resMngr, uint32_t width, uint32_t height)
    {
        mEnableValidation = enableValidation;
        mDevice = device;
        mResManager = resMngr;
        mDescriptorPool = descpool;
        mWidth = width;
        mHeight = height;
        mVS = createShaderModule(vsCode, vsCodeSize);
        mPS = createShaderModule(psCode, psCodeSize);
        createConstantBuffers();

        createRenderPass();
        createDescriptorSetLayout();
        createDescriptorSets(mDescriptorPool);
        mPipelineLayoutOpaque = createGraphicsPipelineLayout();
        mPipelineLayoutTransparent = createGraphicsPipelineLayout();
        mPipelineOpaque = createGraphicsPipeline(mVS, mPS, mPipelineLayoutOpaque, width, height, false);
        mPipelineTransparent = createGraphicsPipeline(mVS, mPS, mPipelineLayoutTransparent, width, height, true);
    }

    void onResize(std::vector<VkImageView>& imageViews, VkImageView& depthImageView, uint32_t width, uint32_t height);

    void onDestroy();

    void updateUniformBuffer(uint32_t currentImage, const glm::float4x4& lightSpaceMatrix, Scene& scene, uint32_t cameraIndex);

    RenderPass(/* args */);
    ~RenderPass();
    void record(VkCommandBuffer& cmd, VkBuffer vertexBuffer, VkBuffer indexBuffer, nevk::Scene& scene, uint32_t width, uint32_t height, uint32_t imageIndex, uint32_t cameraIndex);
};
} // namespace nevk
