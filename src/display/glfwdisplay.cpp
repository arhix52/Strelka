#include "glfwdisplay.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define GL_SILENCE_DEPRECATION

#include <sstream>

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
        std::cerr << ss.str() << std::endl;
        // throw Exception(ss.str().c_str());
        assert(0);
    }
}

#define GL_CHECK(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        call;                                                                                                          \
        glCheck(#call, __FILE__, __LINE__);                                                                            \
    } while (false)

const std::string glfwdisplay::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string glfwdisplay::s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

GLuint createGLShader(const std::string& source, GLuint shader_type)
{
    GLuint shader = glCreateShader(shader_type);
    {
        const GLchar* source_data = reinterpret_cast<const GLchar*>(source.data());
        glShaderSource(shader, 1, &source_data, nullptr);
        glCompileShader(shader);

        GLint is_compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
        if (is_compiled == GL_FALSE)
        {
            GLint max_length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

            std::string info_log(max_length, '\0');
            GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
            glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

            glDeleteShader(shader);
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

            return 0;
        }
    }

    // GL_CHECK_ERRORS();

    return shader;
}

GLuint createGLProgram(const std::string& vert_source, const std::string& frag_source)
{
    GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
    if (vert_shader == 0)
        return 0;

    GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);
    if (frag_shader == 0)
    {
        glDeleteShader(vert_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE)
    {
        GLint max_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

        std::string info_log(max_length, '\0');
        GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
        glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
        std::cerr << "Linking of program failed: " << info_log << std::endl;

        glDeleteProgram(program);
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        return 0;
    }

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    // GL_CHECK_ERRORS();

    return program;
}

GLint getGLUniformLocation(GLuint program, const std::string& name)
{
    GLint loc = glGetUniformLocation(program, name.c_str());
    return loc;
}


glfwdisplay::glfwdisplay(/* args */)
{
}

glfwdisplay::~glfwdisplay()
{
}

void glfwdisplay::init(int width, int height, oka::SharedContext* ctx)
{
    mWindowWidth = width;
    mWindowHeight = height;
    mCtx = ctx;

    glfwInit();
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    mWindow = glfwCreateWindow(mWindowWidth, mWindowHeight, "Strelka", nullptr, nullptr);
    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, framebufferResizeCallback);
    glfwSetKeyCallback(mWindow, keyCallback);
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    glfwSetCursorPosCallback(mWindow, handleMouseMoveCallback);
    glfwSetScrollCallback(mWindow, scrollCallback);

    glfwMakeContextCurrent(mWindow);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        assert(0);
    }

    GLuint m_vertex_array;
    GL_CHECK(glGenVertexArrays(1, &m_vertex_array));
    GL_CHECK(glBindVertexArray(m_vertex_array));

    m_program = createGLProgram(s_vert_source, s_frag_source);
    m_render_tex_uniform_loc = getGLUniformLocation(m_program, "render_tex");

    GL_CHECK(glGenTextures(1, &m_render_tex));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_render_tex));

    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    static const GLfloat g_quad_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f,  0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
    };

    GL_CHECK(glGenBuffers(1, &m_quad_vertex_buffer));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW));

    // GL_CHECK_ERRORS();
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void glfwdisplay::setWindowTitle(const char* title)
{
    glfwSetWindowTitle(mWindow, title);
}

bool glfwdisplay::windowShouldClose()
{
    return glfwWindowShouldClose(mWindow);
}

void glfwdisplay::pollEvents()
{
    glfwPollEvents();
}

void glfwdisplay::framebufferResizeCallback(GLFWwindow* window, int width, int height)
{
    assert(window);
    if (width == 0 || height == 0)
    {
        return;
    }

    auto app = reinterpret_cast<glfwdisplay*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
    ResizeHandler* handler = app->getResizeHandler();
    if (handler)
    {
        handler->framebufferResize(width, height);
    }
}

void glfwdisplay::keyCallback(GLFWwindow* window,
                              [[maybe_unused]] int key,
                              [[maybe_unused]] int scancode,
                              [[maybe_unused]] int action,
                              [[maybe_unused]] int mods)
{
    assert(window);
    auto app = reinterpret_cast<glfwdisplay*>(glfwGetWindowUserPointer(window));
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

void glfwdisplay::mouseButtonCallback(GLFWwindow* window,
                                      [[maybe_unused]] int button,
                                      [[maybe_unused]] int action,
                                      [[maybe_unused]] int mods)
{
    assert(window);
    auto app = reinterpret_cast<glfwdisplay*>(glfwGetWindowUserPointer(window));
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->mouseButtonCallback(button, action, mods);
    }
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

void glfwdisplay::handleMouseMoveCallback(GLFWwindow* window, [[maybe_unused]] double xpos, [[maybe_unused]] double ypos)
{
    assert(window);

    auto app = reinterpret_cast<glfwdisplay*>(glfwGetWindowUserPointer(window));
    // if (app->Ui().wantCaptureMouse())
    // {
    //     return;
    // }
    InputHandler* handler = app->getInputHandler();
    if (handler)
    {
        handler->handleMouseMoveCallback(xpos, ypos);
    }

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

void glfwdisplay::scrollCallback(GLFWwindow* window, [[maybe_unused]] double xoffset, [[maybe_unused]] double yoffset)
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

void glfwdisplay::display(const int32_t screen_res_x,
                          const int32_t screen_res_y,
                          const int32_t framebuf_res_x,
                          const int32_t framebuf_res_y,
                          const uint32_t pbo) const
{
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
    GL_CHECK(glViewport(0, 0, framebuf_res_x, framebuf_res_y));

    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(m_program));

    // Bind our texture in Texture Unit 0
    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_render_tex));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));

    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4)); // TODO!!!!!!

    size_t elmt_size = 4 * sizeof(float); // pixelFormatSize(m_image_format);
    if (elmt_size % 8 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if (elmt_size % 4 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if (elmt_size % 2 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    // if (m_image_format == BufferImageFormat::UNSIGNED_BYTE4)
    // {
    //     // input is assumed to be in srgb since it is only 1 byte per channel in
    //     // size
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    //     convertToSrgb = false;
    // }
    // else if (m_image_format == BufferImageFormat::FLOAT3)
    //     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screen_res_x, screen_res_y, 0, GL_RGB, GL_FLOAT, nullptr);

    // else if (m_image_format == BufferImageFormat::FLOAT4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT, nullptr);
        convertToSrgb = false;

    // else
    //     throw Exception("Unknown buffer format");

    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glUniform1i(m_render_tex_uniform_loc, 0));

    // 1st attribute buffer : vertices
    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer));
    GL_CHECK(glVertexAttribPointer(0, // attribute 0. No particular reason for 0,
                                      // but must match the layout in the shader.
                                   3, // size
                                   GL_FLOAT, // type
                                   GL_FALSE, // normalized?
                                   0, // stride
                                   (void*)0 // array buffer offset
                                   ));

    if (convertToSrgb)
        GL_CHECK(glEnable(GL_FRAMEBUFFER_SRGB));
    else
        GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    // Draw the triangles !
    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0,
                          6)); // 2*3 indices starting at 0 -> 2 triangles

    GL_CHECK(glDisableVertexAttribArray(0));

    GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    // GL_CHECK_ERRORS();
}

void glfwdisplay::drawFrame(ImageBuffer& result)
{
    glClear(GL_COLOR_BUFFER_BIT);
    int framebuf_res_x = 0, framebuf_res_y = 0;
    glfwGetFramebufferSize(mWindow, &framebuf_res_x, &framebuf_res_y);

    if (m_dislpayPbo == 0)
    {
        GL_CHECK(glGenBuffers(1, &m_dislpayPbo));
    }
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_dislpayPbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, result.dataSize, result.data, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

    display(result.width, result.height, framebuf_res_x, framebuf_res_y, m_dislpayPbo);
}

void glfwdisplay::destroy()
{
}

void glfwdisplay::onBeginFrame()
{
}

void glfwdisplay::onEndFrame()
{
    glfwSwapBuffers(mWindow);
}

void glfwdisplay::drawUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();

    const char* debugItems[] = { "None", "Normals" };
    static int currentDebugItemId = 0;

        /*
        bool openFD = false;
        static uint32_t showPropertiesId = -1;
        static uint32_t lightId = -1;
        static bool isLight = false;
        static bool openInspector = false;

        const char* tonemapItems[] = { "None", "Reinhard", "ACES", "Filmic" };
        static int currentTonemapItemId = 1;

        const char* stratifiedSamplingItems[] = { "None", "Random", "Stratified", "Optimized" };
        static int currentSamplingItemId = 1;
        */

    ImGui::Begin("Menu:"); // begin window

    if (ImGui::BeginCombo("Debug view", debugItems[currentDebugItemId]))
    {
        for (int n = 0; n < IM_ARRAYSIZE(debugItems); n++)
        {
            bool is_selected = (currentDebugItemId == n);
            if (ImGui::Selectable(debugItems[n], is_selected))
            {
                currentDebugItemId = n;
            }
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    mCtx->mSettingsManager->setAs<uint32_t>("render/pt/debug", currentDebugItemId);

    if (ImGui::TreeNode("Path Tracer"))
    {
        uint32_t maxDepth = mCtx->mSettingsManager->getAs<uint32_t>("render/pt/depth");
        ImGui::SliderInt("Max Depth", (int*)&maxDepth, 1, 16);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/depth", maxDepth);

        uint32_t sppTotal = mCtx->mSettingsManager->getAs<uint32_t>("render/pt/sppTotal");
        ImGui::SliderInt("SPP Total", (int*)&sppTotal, 1, 10000);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/sppTotal", sppTotal);

        uint32_t sppSubframe = mCtx->mSettingsManager->getAs<uint32_t>("render/pt/spp");
        ImGui::SliderInt("SPP Subframe", (int*)&sppSubframe, 1, 32);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/spp", sppSubframe);

        bool enableAccumulation = mCtx->mSettingsManager->getAs<bool>("render/pt/enableAcc");
        ImGui::Checkbox("Enable Path Tracer Acc", &enableAccumulation);
        mCtx->mSettingsManager->setAs<bool>("render/pt/enableAcc", enableAccumulation);
            /*
            if (ImGui::BeginCombo("Stratified Sampling", stratifiedSamplingItems[currentSamplingItemId]))
            {
                for (int n = 0; n < IM_ARRAYSIZE(stratifiedSamplingItems); n++)
                {
                    bool is_selected = (currentSamplingItemId == n);
                    if (ImGui::Selectable(stratifiedSamplingItems[n], is_selected))
                    {
                        currentSamplingItemId = n;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            mCtx->mSettingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", currentSamplingItemId);
             */
            ImGui::TreePop();
    }
        /*
        if (ImGui::BeginCombo("Tonemap", tonemapItems[currentTonemapItemId]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(tonemapItems); n++)
            {
                bool is_selected = (currentTonemapItemId == n);
                if (ImGui::Selectable(tonemapItems[n], is_selected))
                {
                    currentTonemapItemId = n;
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        mCtx->mSettingsManager->setAs<bool>("render/pt/enableTonemap", currentTonemapItemId != 0 ? true : false);
        mCtx->mSettingsManager->setAs<uint32_t>("render/pt/tonemapperType", currentTonemapItemId - 1);

        bool enableUpscale = mCtx->mSettingsManager->getAs<bool>("render/pt/enableUpscale");
        ImGui::Checkbox("Enable Upscale", &enableUpscale);
        mCtx->mSettingsManager->setAs<bool>("render/pt/enableUpscale", enableUpscale);

        float upscaleFactor = 0.0f;
        if (enableUpscale)
        {
            upscaleFactor = 0.5f;
        }
        else
        {
            upscaleFactor = 1.0f;
        }
        mCtx->mSettingsManager->setAs<float>("render/pt/upscaleFactor", upscaleFactor);

        if (ImGui::Button("Capture Screen"))
        {
            mCtx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", true);
        }

        // bool isRecreate = ImGui::Button("Recreate BVH");
        // renderConfig.recreateBVH = isRecreate ? true : false;
        */

    ImGui::End(); // end window

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(mWindow, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
