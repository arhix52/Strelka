#pragma once

// to supress warnings and faster compilation
#ifdef _WIN32
#    define VC_EXTRALEAN
#    define WIN32_LEAN_AND_MEAN
#    include <Windows.h>
#endif

#define GLFW_INCLUDE_VULKAN
#include "render/vkrender.h"

#include <GLFW/glfw3.h>

#include <ui.h>

namespace oka
{

class InputHandler
{
public:
    virtual void keyCallback(int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) = 0;
    virtual void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods) = 0;
    virtual void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos) = 0;
};

class ResizeHandler
{
public:
    virtual void framebufferResize(int newWidth, int newHeight) = 0;
};

class GLFWRender : public VkRender
{
public:
    void init(int width, int height);
    void destroy();

    void setWindowTitle(const char* title);

    void setInputHandler(InputHandler* handler)
    {
        mInputHandler = handler;
    }
    InputHandler* getInputHandler()
    {
        return mInputHandler;
    }

    Ui& Ui()
    {
        return mUi;
    }

    bool windowShouldClose();
    void pollEvents();

    void onBeginFrame();
    void onEndFrame();

    void drawFrame(Image* result, bool& needCopyBuffer, Buffer* screenshotTransferBuffer);
    void drawUI();

    bool framebufferResized = false;

protected:
    FrameSyncData mSyncData[MAX_FRAMES_IN_FLIGHT] = {};
    FrameSyncData& getCurrentFrameSyncData();
    FrameSyncData& getFrameSyncData(uint32_t idx);
    void createSyncObjects();

    InputHandler* mInputHandler;
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(
        GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void handleMouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    void createLogicalDevice() override;
    void createSurface() override;
    void createSwapChain();
    void recreateSwapChain();
    void cleanupSwapChain();

    struct SwapChainSupportDetails
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    std::vector<const char*> getRequiredExtensions() override;
    // need to find present queue
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) override;

    int mWindowWidth = 800;
    int mWindowHeight = 600;

    GLFWwindow* mWindow;
    VkSurfaceKHR mSurface;
    VkSwapchainKHR mSwapChain;
    std::vector<VkImage> mSwapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D mSwapChainExtent;
    std::vector<VkImageView> mSwapChainImageViews;
    std::vector<VkFramebuffer> mSwapChainFramebuffers;

    oka::Ui::RenderConfig mRenderConfig = {};
    oka::Ui::SceneConfig mSceneConfig = {};
    oka::Ui mUi;
};

} // namespace oka
