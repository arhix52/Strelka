#include "glfwdisplay.h"

#include <sstream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define GL_SILENCE_DEPRECATION

using namespace oka;

inline const char* getGLErrorString(GLenum error)
{
    switch (error)
    {
    case GL_NO_ERROR:
        return "No error";
    case GL_INVALID_ENUM:
        return "Invalid enum";
    case GL_INVALID_VALUE:
        return "Invalid value";
    case GL_INVALID_OPERATION:
        return "Invalid operation";
        // case GL_STACK_OVERFLOW:      return "Stack overflow";
        // case GL_STACK_UNDERFLOW:     return "Stack underflow";
    case GL_OUT_OF_MEMORY:
        return "Out of memory";
        // case GL_TABLE_TOO_LARGE:     return "Table too large";
    default:
        return "Unknown GL error";
    }
}

inline void glCheck(const char* call, const char* file, unsigned int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString(err) << " at " << file << "(" << line << "): " << call << '\n';
        STRELKA_FATAL("{}", ss.str());
        assert(0);
    }
}

#define GL_CHECK(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        call;                                                                                                          \
        glCheck(#call, __FILE__, __LINE__);                                                                            \
    } while (false)


GlfwDisplay::GlfwDisplay(/* args */)
{
}

GlfwDisplay::~GlfwDisplay()
{
    cudaGraphicsUnregisterResource(cudaResource);

    glDeleteTextures(1, &m_render_tex);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void GlfwDisplay::init(int width, int height, SettingsManager* settings)
{
    mWindowWidth = width;
    mWindowHeight = height;
    mSettings = settings;

    glfwInit();
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RED_BITS, 16);
    glfwWindowHint(GLFW_GREEN_BITS, 16);
    glfwWindowHint(GLFW_BLUE_BITS, 16);
    glfwWindowHint(GLFW_ALPHA_BITS, 16);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);


    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "Strelka", nullptr, nullptr);
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);
    glfwSetKeyCallback(mWindow, keyCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, handleMouseMoveCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);

    glfwMakeContextCurrent(mWindow);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        STRELKA_FATAL("Failed to initialize GLAD");
        assert(0);
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    const char* glsl_version = "#version 450 core";
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void* GlfwDisplay::getDisplayNativeTexure()
{
    return reinterpret_cast<void*>(m_render_tex);
}

float GlfwDisplay::getMaxEDR()
{
    return 1.0f;
}

void GlfwDisplay::drawFrame(ImageBuffer& result)
{
    glClear(GL_COLOR_BUFFER_BIT);
    int framebuf_res_x = 0, framebuf_res_y = 0;
    glfwGetFramebufferSize(mWindow, &framebuf_res_x, &framebuf_res_y);

    // TODO: add resize checking
    if (m_render_tex == 0)
    {
        glGenTextures(1, &m_render_tex);
        glBindTexture(GL_TEXTURE_2D, m_render_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, result.width, result.height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        cudaError_t st = cudaGraphicsGLRegisterImage(&cudaResource, m_render_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    }

    cudaArray_t array;
    cudaGraphicsMapResources(1, &cudaResource);
    cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);

    // cudaMemcpyToArray(array, 0, 0, result.deviceData, result.dataSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy2DToArray(array, 0, 0, result.deviceData, result.width * 4 * sizeof(float), result.width* 4 * sizeof(float), result.height, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cudaResource);
}

void GlfwDisplay::destroy()
{
}

void GlfwDisplay::onBeginFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
}

void GlfwDisplay::onEndFrame()
{
    glfwSwapBuffers(mWindow);
}

void GlfwDisplay::drawUI()
{
    int display_w, display_h;
    glfwGetFramebufferSize(mWindow, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
