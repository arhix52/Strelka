#pragma once

#include <render/common.h>
#include <render/buffer.h>

#include <GLFW/glfw3.h>

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

class Display
{
public:
    Display() {}
    virtual ~Display() {}
    
    virtual void init(int width, int height, oka::SharedContext* ctx) = 0;
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

    void setResizeHandler(ResizeHandler* handler)
    {
        mResizeHandler = handler;
    }
    ResizeHandler* getResizeHandler()
    {
        return mResizeHandler;
    }

    bool windowShouldClose();
    void pollEvents();

    void onBeginFrame();
    void onEndFrame();

    void drawFrame(ImageBuffer& result);
    void drawUI();

protected:
    InputHandler* mInputHandler = nullptr;
    ResizeHandler* mResizeHandler = nullptr;

    oka::SharedContext* mCtx = nullptr;

    GLFWwindow* mWindow;
};

} // namespace oka
