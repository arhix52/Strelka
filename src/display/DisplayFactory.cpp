#ifdef __APPLE__
#include "metal/glfwdisplay.h"
#else
#include "opengl/glfwdisplay.h"
#endif

using namespace oka;

Display* DisplayFactory::createDisplay()
{
    return new GlfwDisplay();
}
