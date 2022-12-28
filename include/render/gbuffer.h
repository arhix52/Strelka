#pragma once
#include <resourcemanager/resourcemanager.h>
#include <vulkan/vulkan.h>

namespace oka
{

struct GBuffer
{
    ResourceManager* mResManager = nullptr;
    Image* wPos = VK_NULL_HANDLE;
    Image* depth = VK_NULL_HANDLE;
    Image* normal = VK_NULL_HANDLE;
    Image* tangent = VK_NULL_HANDLE;
    Image* uv = VK_NULL_HANDLE;
    Image* instId = VK_NULL_HANDLE;
    Image* motion = VK_NULL_HANDLE;
    Image* debug = VK_NULL_HANDLE;

    // utils
    VkFormat depthFormat;
    uint32_t width;
    uint32_t height;

    ~GBuffer()
    {
        assert(mResManager);
        if (wPos)
        {
            mResManager->destroyImage(wPos);
        }
        if (depth)
        {
            mResManager->destroyImage(depth);
        }
        if (normal)
        {
            mResManager->destroyImage(normal);
        }
        if (tangent)
        {
            mResManager->destroyImage(tangent);
        }
        if (uv)
        {
            mResManager->destroyImage(uv);
        }
        if (instId)
        {
            mResManager->destroyImage(instId);
        }
        if (motion)
        {
            mResManager->destroyImage(motion);
        }
        if (debug)
        {
            mResManager->destroyImage(debug);
        }
    }
};
} // namespace nevk
