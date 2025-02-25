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

const std::string GlfwDisplay::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string GlfwDisplay::s_frag_source = R"(
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
            STRELKA_FATAL("Compilation of shader failed: {}", info_log);

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
        STRELKA_FATAL("Linking of program failed: {}", info_log);

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

GlfwDisplay::GlfwDisplay(/* args */)
{
}

GlfwDisplay::~GlfwDisplay()
{
}

void GlfwDisplay::init(int width, int height, SettingsManager* settings)
{
    mWindowWidth = width;
    mWindowHeight = height;
    mSettings = settings;

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

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        STRELKA_FATAL("Failed to initialize GLAD");
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
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

void* GlfwDisplay::getDisplayNativeTexure()
{
    return (void*) m_render_tex;
}

void GlfwDisplay::display(const int32_t screen_res_x,
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

void GlfwDisplay::drawFrame(ImageBuffer& result)
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
    // Display::drawUI();

    int display_w, display_h;
    glfwGetFramebufferSize(mWindow, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
