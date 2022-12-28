#include "glfwrender.h"

using namespace oka;

void oka::GLFWRender::init(int width, int height)
{
    mWindowWidth = width;
    mWindowHeight = height;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "Strelka", nullptr, nullptr);
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);
    glfwSetKeyCallback(mWindow, keyCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, handleMouseMoveCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);

    // swapchain support
    mDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    initVulkan();

    createSyncObjects();

    createSwapChain();

    // UI
    QueueFamilyIndices indicesFamily = findQueueFamilies(mPhysicalDevice);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.DescriptorPool = mSharedCtx.mDescriptorPool;
    init_info.Device = mDevice;
    init_info.ImageCount = MAX_FRAMES_IN_FLIGHT;
    init_info.Instance = mInstance;
    init_info.MinImageCount = 2;
    init_info.PhysicalDevice = mPhysicalDevice;
    init_info.Queue = mGraphicsQueue;
    init_info.QueueFamily = indicesFamily.graphicsFamily.value();

    FrameData& frameData = getCurrentFrameData();
    mUi.init(init_info, swapChainImageFormat, mWindow, frameData.cmdPool, frameData.cmdBuffer, mSwapChainExtent.width,
             mSwapChainExtent.height);
    mUi.createFrameBuffers(mDevice, mSwapChainImageViews, mSwapChainExtent.width, mSwapChainExtent.height);
}

void oka::GLFWRender::destroy()
{
    vkDeviceWaitIdle(mDevice);
    for (FrameSyncData& fd : mSyncData)
    {
        vkDestroySemaphore(mDevice, fd.renderFinished, nullptr);
        vkDestroySemaphore(mDevice, fd.imageAvailable, nullptr);
        vkDestroyFence(mDevice, fd.inFlightFence, nullptr);
    }
}

void oka::GLFWRender::setWindowTitle(const char* title)
{
    glfwSetWindowTitle(mWindow, title);
}

bool oka::GLFWRender::windowShouldClose()
{
    return glfwWindowShouldClose(mWindow);
}

void oka::GLFWRender::pollEvents()
{
    glfwPollEvents();
}

FrameSyncData& oka::GLFWRender::getCurrentFrameSyncData()
{
    return mSyncData[mSharedCtx.mFrameNumber % MAX_FRAMES_IN_FLIGHT];
}

FrameSyncData& oka::GLFWRender::getFrameSyncData(uint32_t idx)
{
    return mSyncData[idx];
}

void oka::GLFWRender::createSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mSyncData[i].renderFinished) != VK_SUCCESS ||
            vkCreateSemaphore(mDevice, &semaphoreInfo, nullptr, &mSyncData[i].imageAvailable) != VK_SUCCESS ||
            vkCreateFence(mDevice, &fenceInfo, nullptr, &mSyncData[i].inFlightFence) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

void oka::GLFWRender::onBeginFrame()
{
    FrameSyncData& currFrame = getCurrentFrameSyncData();

    vkWaitForFences(mDevice, 1, &currFrame.inFlightFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result =
        vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, currFrame.imageAvailable, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (getFrameSyncData(imageIndex).imagesInFlight != VK_NULL_HANDLE)
    {
        vkWaitForFences(mDevice, 1, &getFrameSyncData(imageIndex).imagesInFlight, VK_TRUE, UINT64_MAX);
    }
    getFrameSyncData(imageIndex).imagesInFlight = currFrame.inFlightFence;

    static auto prevTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    // double deltaTime = std::chrono::duration<double, std::milli>(currentTime - prevTime).count() / 1000.0;
    prevTime = currentTime;

    const uint32_t frameIndex = imageIndex;
    mSharedCtx.mFrameIndex = frameIndex;

    VkCommandBuffer& cmd = getCurrentFrameData().cmdBuffer;
    result = vkResetCommandBuffer(cmd, 0);
    assert(result == VK_SUCCESS);
    VkCommandBufferBeginInfo cmdBeginInfo = {};
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBeginInfo.pNext = nullptr;
    cmdBeginInfo.pInheritanceInfo = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    result = vkBeginCommandBuffer(cmd, &cmdBeginInfo);
    assert(result == VK_SUCCESS);
}

void oka::GLFWRender::onEndFrame()
{
    FrameSyncData& currFrame = getCurrentFrameSyncData();
    const uint32_t frameIndex = mSharedCtx.mFrameIndex;

    VkCommandBuffer& cmd = getCurrentFrameData().cmdBuffer;

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to record command buffer!");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = { currFrame.imageAvailable };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkSemaphore signalSemaphores[] = { currFrame.renderFinished };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(mDevice, 1, &currFrame.inFlightFence);

    if (vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, currFrame.inFlightFence) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = { mSwapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &frameIndex;

    VkResult result = vkQueuePresentKHR(mPresentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
    {
        framebufferResized = false;
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    ++mSharedCtx.mFrameNumber;
}

void oka::GLFWRender::drawFrame(Image* result, bool& needCopyBuffer, Buffer* screenshotTransferBuffer)
{
    const uint32_t frameIndex = mSharedCtx.mFrameIndex;
    VkCommandBuffer& cmd = getCurrentFrameData().cmdBuffer;

    // Copy to swapchain image
    if (result)
    {
        recordImageBarrier(cmd, result, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        recordBarrier(cmd, mSwapChainImages[frameIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        if (needCopyBuffer)
        {
            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = { 0, 0, 0 };
            region.imageExtent = { static_cast<uint32_t>(mWindowWidth), static_cast<uint32_t>(mWindowHeight), 1 };

            vkCmdCopyImageToBuffer(cmd, mSharedCtx.mResManager->getVkImage(result), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   mSharedCtx.mResManager->getVkBuffer(screenshotTransferBuffer), 1, &region);

            needCopyBuffer = false;
        }

        VkOffset3D srcBlitSize{};
        srcBlitSize.x = mWindowWidth;
        srcBlitSize.y = mWindowHeight;
        srcBlitSize.z = 1;
        VkImageBlit imageBlitRegion{};
        imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.srcSubresource.layerCount = 1;
        imageBlitRegion.srcOffsets[1] = srcBlitSize;
        imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlitRegion.dstSubresource.layerCount = 1;
        VkOffset3D blitSwapSize{};
        blitSwapSize.x = mSwapChainExtent.width;
        blitSwapSize.y = mSwapChainExtent.height;
        blitSwapSize.z = 1;
        imageBlitRegion.dstOffsets[1] = blitSwapSize;
        vkCmdBlitImage(cmd, mSharedCtx.mResManager->getVkImage(result), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       mSwapChainImages[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageBlitRegion,
                       VK_FILTER_NEAREST);

        recordImageBarrier(cmd, result, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        recordBarrier(cmd, mSwapChainImages[frameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT);
    }
}

void oka::GLFWRender::drawUI()
{
    std::string newModelPath;

    mUi.updateUI(mSharedCtx.mSettingsManager);

    const uint32_t frameIndex = mSharedCtx.mFrameIndex;
    VkCommandBuffer& cmd = getCurrentFrameData().cmdBuffer;

    mUi.render(cmd, frameIndex);
}

void oka::GLFWRender::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    assert(window);
    if (width == 0 || height == 0)
    {
        return;
    }

    auto app = reinterpret_cast<GLFWRender*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
    // nevk::Scene* scene = app->getScene();
    // scene->updateCamerasParams(width, height);
}

void oka::GLFWRender::keyCallback(GLFWwindow* window,
                                  [[maybe_unused]] int key,
                                  [[maybe_unused]] int scancode,
                                  [[maybe_unused]] int action,
                                  [[maybe_unused]] int mods)
{
    assert(window);
    auto app = reinterpret_cast<GLFWRender*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    assert(handler);

    handler->keyCallback(key, scancode, action, mods);

    // Camera& camera = scene->getCamera(app->getActiveCameraIndex());

    // const bool keyState = ((GLFW_REPEAT == action) || (GLFW_PRESS == action)) ? true : false;
    // switch (key)

    //{
    // case GLFW_KEY_W: {
    //    camera.keys.forward = keyState;
    //    break;
    //}
    // case GLFW_KEY_S: {
    //    camera.keys.back = keyState;
    //    break;
    //}
    // case GLFW_KEY_A: {
    //    camera.keys.left = keyState;
    //    break;
    //}
    // case GLFW_KEY_D: {
    //    camera.keys.right = keyState;
    //    break;
    //}
    // case GLFW_KEY_Q: {
    //    camera.keys.up = keyState;
    //    break;
    //}
    // case GLFW_KEY_E: {
    //    camera.keys.down = keyState;
    //    break;
    //}
    // default:
    //    break;
    //}
}

void oka::GLFWRender::mouseButtonCallback(GLFWwindow* window,
                                          [[maybe_unused]] int button,
                                          [[maybe_unused]] int action,
                                          [[maybe_unused]] int mods)
{
    assert(window);
    auto app = reinterpret_cast<GLFWRender*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    assert(handler);
    handler->mouseButtonCallback(button, action, mods);

    // Camera& camera = scene->getCamera(app->getActiveCameraIndex());
    // if (button == GLFW_MOUSE_BUTTON_RIGHT)
    //{
    //    if (action == GLFW_PRESS)
    //    {
    //        camera.mouseButtons.right = true;
    //    }
    //    else if (action == GLFW_RELEASE)
    //    {
    //        camera.mouseButtons.right = false;
    //    }
    //}
    // else if (button == GLFW_MOUSE_BUTTON_LEFT)
    //{
    //    if (action == GLFW_PRESS)
    //    {
    //        camera.mouseButtons.left = true;
    //    }
    //    else if (action == GLFW_RELEASE)
    //    {
    //        camera.mouseButtons.left = false;
    //    }
    //}
}

void oka::GLFWRender::handleMouseMoveCallback(GLFWwindow* window, [[maybe_unused]] double xpos, [[maybe_unused]] double ypos)
{
    assert(window);

    auto app = reinterpret_cast<GLFWRender*>(glfwGetWindowUserPointer(window));
    if (app->Ui().wantCaptureMouse())
    {
        return;
    }
    InputHandler* handler = app->getInputHandler();
    assert(handler);
    handler->handleMouseMoveCallback(xpos, ypos);


    // auto app = reinterpret_cast<Render*>(glfwGetWindowUserPointer(window));
    // oka::Scene* scene = app->getScene();
    // Camera& camera = scene->getCamera(app->getActiveCameraIndex());
    // const float dx = camera.mousePos.x - (float)xpos;
    // const float dy = camera.mousePos.y - (float)ypos;

    // ImGuiIO& io = ImGui::GetIO();
    // bool handled = io.WantCaptureMouse;
    // if (handled)
    //{
    //    camera.mousePos = glm::vec2((float)xpos, (float)ypos);
    //    return;
    //}

    // if (camera.mouseButtons.right)
    //{
    //    camera.rotate(-dx, -dy);
    //}
    // if (camera.mouseButtons.left)
    //{
    //    camera.translate(glm::float3(-0.0f, 0.0f, -dy * .005f * camera.movementSpeed));
    //}
    // if (camera.mouseButtons.middle)
    //{
    //    camera.translate(glm::float3(-dx * 0.01f, -dy * 0.01f, 0.0f));
    //}
    // camera.mousePos = glm::float2((float)xpos, (float)ypos);
}

void oka::GLFWRender::scrollCallback(GLFWwindow* window, [[maybe_unused]] double xoffset, [[maybe_unused]] double yoffset)
{
    assert(window);
    // ImGuiIO& io = ImGui::GetIO();
    // bool handled = io.WantCaptureMouse;
    // if (handled)
    //{
    //    return;
    //}

    // auto app = reinterpret_cast<Render*>(glfwGetWindowUserPointer(window));
    // oka::Scene* mScene = app->getScene();
    // Camera& mCamera = mScene->getCamera(app->getActiveCameraIndex());

    // mCamera.translate(glm::vec3(0.0f, 0.0f,
    //                            -yoffset * mCamera.movementSpeed));
}

void oka::GLFWRender::createLogicalDevice()
{
    oka::VkRender::createLogicalDevice();
    // add present queue
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);
    vkGetDeviceQueue(mDevice, indices.presentFamily.value(), 0, &mPresentQueue);
}

void oka::GLFWRender::createSurface()
{
    if (glfwCreateWindowSurface(mInstance, mWindow, nullptr, &mSurface) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface!");
    }
}

void oka::GLFWRender::createSwapChain()
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(mPhysicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
    {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = mSurface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily)
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    VkResult res = vkCreateSwapchainKHR(mDevice, &createInfo, nullptr, &mSwapChain);
    if (res != VK_SUCCESS)
    {
        assert(0);
        return;
    }

    vkGetSwapchainImagesKHR(mDevice, mSwapChain, &imageCount, nullptr);
    mSwapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(mDevice, mSwapChain, &imageCount, mSwapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    mSwapChainExtent = extent;

    mSwapChainImageViews.resize(mSwapChainImages.size());

    for (uint32_t i = 0; i < mSwapChainImages.size(); i++)
    {
        mSwapChainImageViews[i] = mSharedCtx.mTextureManager->createImageView(
            mSwapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void oka::GLFWRender::recreateSwapChain()
{
    int width = 0, height = 0;
    glfwGetFramebufferSize(mWindow, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(mWindow, &width, &height);
        glfwWaitEvents();
    }
    VkResult res = vkDeviceWaitIdle(mDevice);
    assert(res == VK_SUCCESS);

    cleanupSwapChain();
    createSwapChain();

    mUi.onResize(mSwapChainImageViews, mSwapChainExtent.width, mSwapChainExtent.height);
}

void oka::GLFWRender::cleanupSwapChain()
{
    vkDestroySwapchainKHR(mDevice, mSwapChain, nullptr);

    for (auto& imageView : mSwapChainImageViews)
    {
        vkDestroyImageView(mDevice, imageView, nullptr);
    }
}

oka::GLFWRender::SwapChainSupportDetails oka::GLFWRender::querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, mSurface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, mSurface, &formatCount, nullptr);

    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, mSurface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, mSurface, &presentModeCount, nullptr);

    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, mSurface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR oka::GLFWRender::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

VkPresentModeKHR oka::GLFWRender::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D oka::GLFWRender::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
    {
        return capabilities.currentExtent;
    }
    else
    {
        int width, height;
        glfwGetFramebufferSize(mWindow, &width, &height);

        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width =
            std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(
            capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

std::vector<const char*> oka::GLFWRender::getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif // __APPLE__

    return extensions;
}

QueueFamilyIndices oka::GLFWRender::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies)
    {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, mSurface, &presentSupport);

        if (presentSupport)
        {
            indices.presentFamily = i;
        }

        if (indices.isComplete())
        {
            break;
        }

        i++;
    }

    return indices;
}
