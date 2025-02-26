#pragma once

#include <render/common.h>
#include <render/buffer.h>

#include <settings/settings.h>

#include <GLFW/glfw3.h>

namespace oka
{

class InputHandler
{
public:
    virtual void keyCallback(int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) = 0;
    virtual void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods, bool viewPortHovered) = 0;
    virtual void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos) = 0;
};

class ResizeHandler
{
public:
    virtual void framebufferResize(int newWidth, int newHeight) = 0;
};

class Display
{
public:
    Display() = default;
    virtual ~Display() = default;

    virtual void init(int width, int height, SettingsManager* settings) = 0;
    virtual void destroy() = 0;

    virtual void* getDisplayNativeTexure() = 0;
    virtual float getMaxEDR() = 0;

#ifdef __APPLE__
    virtual void setNativeDevice(void* device) = 0;
#endif
    void setWindowTitle(const char* title)
    {
        glfwSetWindowTitle(mWindow, title);
    }

    void setInputHandler(InputHandler* handler)
    {
        mInputHandler = handler;
    }
    InputHandler* getInputHandler()
    {
        return mInputHandler;
    }

    void setResizeHandler(ResizeHandler* handler)
    {
        mResizeHandler = handler;
    }
    ResizeHandler* getResizeHandler()
    {
        return mResizeHandler;
    }

    bool windowShouldClose()
    {
        return glfwWindowShouldClose(mWindow) != 0;
    }

    void pollEvents()
    {
        glfwPollEvents();
    }

    virtual void onBeginFrame() = 0;
    virtual void onEndFrame() = 0;

    virtual void drawFrame(ImageBuffer& result) = 0;
    virtual void drawUI() = 0;

    void setViewPortHovered(bool state)
    {
        mViewPortHovered = state;
    }

protected:
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(
        GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void handleMouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    int mWindowWidth = 800;
    int mWindowHeight = 600;
    bool mViewPortHovered = false;

    InputHandler* mInputHandler = nullptr;
    ResizeHandler* mResizeHandler = nullptr;

    SettingsManager* mSettings = nullptr;

    GLFWwindow* mWindow;
};

class DisplayFactory
{
public:
    static Display* createDisplay();
};

} // namespace oka
