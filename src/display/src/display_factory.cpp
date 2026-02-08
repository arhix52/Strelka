#ifdef __APPLE__
#include "metal/glfw_display.h"
#else
#include "opengl/glfw_display.h"
#endif

using namespace oka;

Display* DisplayFactory::createDisplay()
{
    return new GlfwDisplay();
}
