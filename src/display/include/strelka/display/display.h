#pragma once

#include <strelka/render/common.h>
#include <strelka/render/buffer.h>
#include <strelka/render/render.h>
#include <strelka/display/output_policy.h>
#include <strelka/display/gamepad.h>
#include <strelka/display/glfw_gamepad.h>

#include <settings.h>

#include <GLFW/glfw3.h>

namespace oka
{

class InputHandler
{
public:
    virtual ~InputHandler() = default;

    virtual void keyCallback(int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) = 0;
    virtual void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods, bool viewPortHovered) = 0;
    virtual void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos) = 0;
    /// Wheel or two-finger scroll over the viewport. Not pure: a handler that has
    /// nothing to zoom is a valid handler.
    virtual void scrollCallback([[maybe_unused]] double xoffset, [[maybe_unused]] double yoffset)
    {
    }
};

class ResizeHandler
{
public:
    virtual ~ResizeHandler() = default;

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
    virtual display_output::DisplayCapabilities getOutputCapabilities() const
    {
        return {};
    }

#ifdef __APPLE__
    virtual void setNativeDevice(void* device) = 0;
    virtual void setCommandQueue(void* queue) = 0;
#endif

    /// The renderer whose output this display samples. Needed only so the
    /// display can wait on the frame event when the two are on different queues;
    /// a backend that shares a queue can ignore it.
    ///
    /// Not inside the __APPLE__ guard above: `mRender` is declared
    /// unconditionally and EditorApp calls this unconditionally, so guarding it
    /// only meant the editor did not compile off Apple. The Metal-specific part
    /// is the device/queue interop, not the pointer.
    void setRender(Render* render)
    {
        mRender = render;
    }

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

    void requestClose()
    {
        glfwSetWindowShouldClose(mWindow, GLFW_TRUE);
    }

    void pollEvents()
    {
        glfwPollEvents();
        // Here rather than in either backend: both windowing paths are GLFW and
        // the joystick API is per-process, so a copy in each would be two
        // answers to one question. Also here rather than in the editor's main
        // loop, because the editor has half a dozen other loops -- scene load,
        // benchmarks, convergence runs -- that pump events without going through
        // it, and a pad plugged in during one of those has to be noticed too.
        glfw_gamepad::poll(mGamepad);
    }

    /// The gamepad as of the last pollEvents(), or a disconnected state.
    ///
    /// Detection is automatic and continuous: nothing has to be enabled, and a
    /// pad plugged in or pulled mid-session is picked up on the next frame.
    const GamepadState& getGamepadState() const
    {
        return mGamepad;
    }

    virtual void onBeginFrame() = 0;
    virtual void onEndFrame() = 0;

    virtual void drawFrame(ImageBuffer& result) = 0;
    virtual void drawUI() = 0;
    virtual void resetFrame() {}

    /// False when the backend skipped Metal/GL NewFrame (minimised, no drawable,
    /// semaphore timeout). The editor must not run an ImGui frame in that case.
    virtual bool isFrameValid() const
    {
        return true;
    }

    void setViewPortHovered(bool state)
    {
        mViewPortHovered = state;
    }

    bool isViewPortHovered() const
    {
        return mViewPortHovered;
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
    Render* mRender = nullptr;

    GLFWwindow* mWindow = nullptr;
    GamepadState mGamepad;
};

class DisplayFactory
{
public:
    static Display* createDisplay();
};

} // namespace oka
