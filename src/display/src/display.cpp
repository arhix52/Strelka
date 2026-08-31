#include <strelka/display/display.h>

#ifdef __APPLE__
#include "metal/glfw_display.h"
#else
#include "vulkan/glfw_display.h"
#endif

using namespace oka;

Display* oka::createDisplay()
{
    return new GlfwDisplay();
}

void Display::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    assert(window);
    if (width == 0 || height == 0)
    {
        return;
    }

    auto app = static_cast<Display*>(glfwGetWindowUserPointer(window));
    // app->framebufferResized = true;
    ResizeHandler* handler = app->getResizeHandler();
    if (handler)
    {
        handler->framebufferResize(width, height);
    }
}

void Display::keyCallback(GLFWwindow* window,
                          [[maybe_unused]] int key,
                          [[maybe_unused]] int scancode,
                          [[maybe_unused]] int action,
                          [[maybe_unused]] int mods)
{
    assert(window);
    auto app = static_cast<Display*>(glfwGetWindowUserPointer(window));
    if (!app->mViewPortHovered)
    {
        return;
    }
    InputHandler* handler = app->getInputHandler();
    assert(handler);
    handler->keyCallback(key, scancode, action, mods);
}

void Display::mouseButtonCallback(GLFWwindow* window,
                                  [[maybe_unused]] int button,
                                  [[maybe_unused]] int action,
                                  [[maybe_unused]] int mods)
{
    assert(window);
    auto app = static_cast<Display*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->mouseButtonCallback(button, action, mods, app->mViewPortHovered);
    }
}

void Display::handleMouseMoveCallback(GLFWwindow* window, [[maybe_unused]] double xpos, [[maybe_unused]] double ypos)
{
    assert(window);
    auto app = static_cast<Display*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->handleMouseMoveCallback(xpos, ypos);
    }
}

void Display::scrollCallback(GLFWwindow* window, [[maybe_unused]] double xoffset, [[maybe_unused]] double yoffset)
{
    assert(window);
    auto app = static_cast<Display*>(glfwGetWindowUserPointer(window));
    // Gated on hover like the keyboard is, and for the same reason: the wheel over
    // a panel belongs to that panel's scrollbar, and ImGui has already had it.
    if (!app->mViewPortHovered)
    {
        return;
    }
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->scrollCallback(xoffset, yoffset);
    }
}
