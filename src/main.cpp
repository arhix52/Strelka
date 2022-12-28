
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>


#include <glad/gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>

#include "optixTriangle.h"

#include <array>
#include <vector>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>


#include <cmath>

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

class Exception : public std::runtime_error
{
public:
    Exception(const char* msg) : std::runtime_error(msg)
    {
    }

    Exception(OptixResult res, const char* msg) : std::runtime_error(createMessage(res, msg).c_str())
    {
    }

private:
    std::string createMessage(OptixResult res, const char* msg)
    {
        std::ostringstream out;
        out << optixGetErrorName(res) << ": " << msg;
        return out.str();
    }
};

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
        throw Exception(ss.str().c_str());
    }
}

inline void optixCheck(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        throw Exception(res, ss.str().c_str());
    }
}

inline void optixCheckLog(OptixResult res,
                          const char* log,
                          size_t sizeof_log,
                          size_t sizeof_log_returned,
                          const char* call,
                          const char* file,
                          unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
           << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
        throw Exception(res, ss.str().c_str());
    }
}

inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '" << cudaGetErrorString(error) << "' (" << file << ":"
           << line << ")\n";
        throw Exception(ss.str().c_str());
    }
}

inline void cudaSyncCheck(const char* file, unsigned int line)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '" << cudaGetErrorString(error) << "' (" << file << ":" << line
           << ")\n";
        throw Exception(ss.str().c_str());
    }
}

#define GL_CHECK(call)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        call;                                                                                                          \
        glCheck(#call, __FILE__, __LINE__);                                                                            \
    } while (false)

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() cudaSyncCheck(__FILE__, __LINE__)

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------
#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)

#define OPTIX_CHECK_LOG(call) optixCheckLog(call, log, sizeof(log), sizeof_log, #call, __FILE__, __LINE__)

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}
static void errorCallback(int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}
static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
}

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

enum class CUDAOutputBufferType
{
    CUDA_DEVICE = 0, // not preferred, typically slower than ZERO_COPY
    GL_INTEROP = 1, // single device only, preferred for single device
    ZERO_COPY = 2, // general case, preferred for multi-gpu if not fully nvlink connected
    CUDA_P2P = 3 // fully connected only, preferred for fully nvlink connected
};

template <typename PIXEL_FORMAT>
class CUDAOutputBuffer
{
public:
    CUDAOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height);
    ~CUDAOutputBuffer();

    void setDevice(int32_t device_idx)
    {
        m_device_idx = device_idx;
    }
    void setStream(CUstream stream)
    {
        m_stream = stream;
    }

    void resize(int32_t width, int32_t height);

    // Allocate or update device pointer as necessary for CUDA access
    PIXEL_FORMAT* map();
    void unmap();

    int32_t width() const
    {
        return m_width;
    }
    int32_t height() const
    {
        return m_height;
    }

    // Get output buffer
    GLuint getPBO();
    void deletePBO();
    PIXEL_FORMAT* getHostPointer();

private:
    void makeCurrent()
    {
        CUDA_CHECK(cudaSetDevice(m_device_idx));
    }

    CUDAOutputBufferType m_type;

    int32_t m_width = 0u;
    int32_t m_height = 0u;

    cudaGraphicsResource* m_cuda_gfx_resource = nullptr;
    GLuint m_pbo = 0u;
    PIXEL_FORMAT* m_device_pixels = nullptr;
    PIXEL_FORMAT* m_host_zcopy_pixels = nullptr;
    std::vector<PIXEL_FORMAT> m_host_pixels;

    CUstream m_stream = 0u;
    int32_t m_device_idx = 0;
};

void ensureMinimumSize(int& w, int& h)
{
    if (w <= 0)
        w = 1;
    if (h <= 0)
        h = 1;
}

void ensureMinimumSize(unsigned& w, unsigned& h)
{
    if (w == 0)
        w = 1;
    if (h == 0)
        h = 1;
}

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer(CUDAOutputBufferType type, int32_t width, int32_t height)
    : m_type(type)
{
    // Output dimensions must be at least 1 in both x and y to avoid an error
    // with cudaMalloc.
#if 0
	if (width < 1 || height < 1)
	{
		throw sutil::Exception("CUDAOutputBuffer dimensions must be at least 1 in both x and y.");
	}
#else
    ensureMinimumSize(width, height);
#endif

    // If using GL Interop, expect that the active device is also the display
    // device.
    if (type == CUDAOutputBufferType::GL_INTEROP)
    {
        int current_device, is_display_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
        if (!is_display_device)
        {
            throw Exception(
                "GL interop is only available on display device, please "
                "use display device for optimal "
                "performance.  Alternatively you can disable GL interop "
                "with --no-gl-interop and run with "
                "degraded performance.");
        }
    }
    resize(width, height);
}

template <typename PIXEL_FORMAT>
CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer()
{
    try
    {
        makeCurrent();
        if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_pixels)));
        }
        else if (m_type == CUDAOutputBufferType::ZERO_COPY)
        {
            CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zcopy_pixels)));
        }
        else if (m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P)
        {
            CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_gfx_resource));
        }

        if (m_pbo != 0u)
        {
            GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
            GL_CHECK(glDeleteBuffers(1, &m_pbo));
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "CUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
    }
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::resize(int32_t width, int32_t height)
{
    // Output dimensions must be at least 1 in both x and y to avoid an error
    // with cudaMalloc.
    ensureMinimumSize(width, height);

    if (m_width == width && m_height == height)
        return;

    m_width = width;
    m_height = height;

    makeCurrent();

    if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_device_pixels)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_device_pixels), m_width * m_height * sizeof(PIXEL_FORMAT)));
    }

    if (m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P)
    {
        // GL buffer gets resized below
        GL_CHECK(glGenBuffers(1, &m_pbo));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height, nullptr, GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_gfx_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));
    }

    if (m_type == CUDAOutputBufferType::ZERO_COPY)
    {
        CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zcopy_pixels)));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&m_host_zcopy_pixels),
                                 m_width * m_height * sizeof(PIXEL_FORMAT), cudaHostAllocPortable | cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(
            reinterpret_cast<void**>(&m_device_pixels), reinterpret_cast<void*>(m_host_zcopy_pixels), 0 /*flags*/
            ));
    }

    if (m_type != CUDAOutputBufferType::GL_INTEROP && m_type != CUDAOutputBufferType::CUDA_P2P && m_pbo != 0u)
    {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height, nullptr, GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));
    }

    if (!m_host_pixels.empty())
        m_host_pixels.resize(m_width * m_height);
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::map()
{
    if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P)
    {
        // nothing needed
    }
    else if (m_type == CUDAOutputBufferType::GL_INTEROP)
    {
        makeCurrent();

        size_t buffer_size = 0u;
        CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&m_device_pixels), &buffer_size, m_cuda_gfx_resource));
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        // nothing needed
    }

    return m_device_pixels;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::unmap()
{
    makeCurrent();

    if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P)
    {
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    }
    else if (m_type == CUDAOutputBufferType::GL_INTEROP)
    {
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    }
}

template <typename PIXEL_FORMAT>
GLuint CUDAOutputBuffer<PIXEL_FORMAT>::getPBO()
{
    if (m_pbo == 0u)
        GL_CHECK(glGenBuffers(1, &m_pbo));

    const size_t buffer_size = m_width * m_height * sizeof(PIXEL_FORMAT);

    if (m_type == CUDAOutputBufferType::CUDA_DEVICE)
    {
        // We need a host buffer to act as a way-station
        if (m_host_pixels.empty())
            m_host_pixels.resize(m_width * m_height);

        makeCurrent();
        CUDA_CHECK(
            cudaMemcpy(static_cast<void*>(m_host_pixels.data()), m_device_pixels, buffer_size, cudaMemcpyDeviceToHost));

        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, buffer_size, static_cast<void*>(m_host_pixels.data()), GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }
    else if (m_type == CUDAOutputBufferType::GL_INTEROP)
    {
        // Nothing needed
    }
    else if (m_type == CUDAOutputBufferType::CUDA_P2P)
    {
        makeCurrent();
        void* pbo_buff = nullptr;
        size_t dummy_size = 0;

        CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&pbo_buff, &dummy_size, m_cuda_gfx_resource));
        CUDA_CHECK(cudaMemcpy(pbo_buff, m_device_pixels, buffer_size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, buffer_size, static_cast<void*>(m_host_zcopy_pixels), GL_STREAM_DRAW));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    }

    return m_pbo;
}

template <typename PIXEL_FORMAT>
void CUDAOutputBuffer<PIXEL_FORMAT>::deletePBO()
{
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_CHECK(glDeleteBuffers(1, &m_pbo));
    m_pbo = 0;
}

template <typename PIXEL_FORMAT>
PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer()
{
    if (m_type == CUDAOutputBufferType::CUDA_DEVICE || m_type == CUDAOutputBufferType::CUDA_P2P ||
        m_type == CUDAOutputBufferType::GL_INTEROP)
    {
        m_host_pixels.resize(m_width * m_height);

        makeCurrent();
        CUDA_CHECK(cudaMemcpy(static_cast<void*>(m_host_pixels.data()), map(),
                              m_width * m_height * sizeof(PIXEL_FORMAT), cudaMemcpyDeviceToHost));
        unmap();

        return m_host_pixels.data();
    }
    else // m_type == CUDAOutputBufferType::ZERO_COPY
    {
        return m_host_zcopy_pixels;
    }
}

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data = nullptr;
    unsigned int width = 0;
    unsigned int height = 0;
    BufferImageFormat pixel_format;
};

class Camera
{
public:
    Camera()
        : m_eye(glm::float3(1.0f)),
          m_lookat(glm::float3(0.0f)),
          m_up(glm::float3(0.0f, 1.0f, 0.0f)),
          m_fovY(35.0f),
          m_aspectRatio(1.0f)
    {
    }

    Camera(const glm::float3& eye, const glm::float3& lookat, const glm::float3& up, float fovY, float aspectRatio)
        : m_eye(eye), m_lookat(lookat), m_up(up), m_fovY(fovY), m_aspectRatio(aspectRatio)
    {
    }

    glm::float3 direction() const
    {
        return glm::normalize(m_lookat - m_eye);
    }
    void setDirection(const glm::float3& dir)
    {
        m_lookat = m_eye + glm::length(m_lookat - m_eye) * dir;
    }

    const glm::float3& eye() const
    {
        return m_eye;
    }
    void setEye(const glm::float3& val)
    {
        m_eye = val;
    }
    const glm::float3& lookat() const
    {
        return m_lookat;
    }
    void setLookat(const glm::float3& val)
    {
        m_lookat = val;
    }
    const glm::float3& up() const
    {
        return m_up;
    }
    void setUp(const glm::float3& val)
    {
        m_up = val;
    }
    const float& fovY() const
    {
        return m_fovY;
    }
    void setFovY(const float& val)
    {
        m_fovY = val;
    }
    const float& aspectRatio() const
    {
        return m_aspectRatio;
    }
    void setAspectRatio(const float& val)
    {
        m_aspectRatio = val;
    }

    // UVW forms an orthogonal, but not orthonormal basis!
    void UVWFrame(glm::float3& U, glm::float3& V, glm::float3& W) const
    {
        W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
        float wlen = glm::length(W);
        U = glm::normalize(glm::cross(W, m_up));
        V = glm::normalize(glm::cross(U, W));

        float vlen = wlen * tanf(0.5f * m_fovY * M_PI / 180.0f);
        V *= vlen;
        float ulen = vlen * m_aspectRatio;
        U *= ulen;
    }

private:
    glm::float3 m_eye;
    glm::float3 m_lookat;
    glm::float3 m_up;
    float m_fovY;
    float m_aspectRatio;
};

void configureCamera(Camera& cam, const uint32_t width, const uint32_t height)
{
    cam.setEye({ 0.0f, 0.0f, 2.0f });
    cam.setLookat({ 0.0f, 0.0f, 0.0f });
    cam.setUp({ 0.0f, 1.0f, 3.0f });
    cam.setFovY(45.0f);
    cam.setAspectRatio((float)width / (float)height);
}

class GLDisplay
{
public:
    GLDisplay(BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE4);

    void display(const int32_t screen_res_x,
                 const int32_t screen_res_y,
                 const int32_t framebuf_res_x,
                 const int32_t framebuf_res_y,
                 const uint32_t pbo) const;

private:
    GLuint m_render_tex = 0u;
    GLuint m_program = 0u;
    GLint m_render_tex_uniform_loc = -1;
    GLuint m_quad_vertex_buffer = 0;

    BufferImageFormat m_image_format;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};
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
    // SUTIL_ASSERT_MSG(loc != -1, "Failed to get uniform loc for '" + name +
    // "'");
    return loc;
}

const std::string GLDisplay::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string GLDisplay::s_frag_source = R"(
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

GLDisplay::GLDisplay(BufferImageFormat image_format) : m_image_format(image_format)
{
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
}

size_t pixelFormatSize(BufferImageFormat format)
{
    switch (format)
    {
    case BufferImageFormat::UNSIGNED_BYTE4:
        return sizeof(char) * 4;
    case BufferImageFormat::FLOAT3:
        return sizeof(float) * 3;
    case BufferImageFormat::FLOAT4:
        return sizeof(float) * 4;
    default:
        throw Exception("pixelFormatSize: Unrecognized buffer format");
    }
}

void GLDisplay::display(const int32_t screen_res_x,
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

    size_t elmt_size = pixelFormatSize(m_image_format);
    if (elmt_size % 8 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if (elmt_size % 4 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if (elmt_size % 2 == 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool convertToSrgb = true;

    if (m_image_format == BufferImageFormat::UNSIGNED_BYTE4)
    {
        // input is assumed to be in srgb since it is only 1 byte per channel in
        // size
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_res_x, screen_res_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        convertToSrgb = false;
    }
    else if (m_image_format == BufferImageFormat::FLOAT3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screen_res_x, screen_res_y, 0, GL_RGB, GL_FLOAT, nullptr);

    else if (m_image_format == BufferImageFormat::FLOAT4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res_x, screen_res_y, 0, GL_RGBA, GL_FLOAT, nullptr);

    else
        throw Exception("Unknown buffer format");

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

int main(int argc, char* argv[])
{
    //
    // Initialize GLFW state
    //
    GLFWwindow* window = nullptr;
    glfwSetErrorCallback(errorCallback);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                   GL_TRUE); // To make Apple happy -- should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    int width = 1024;
    int height = 768;

    window = glfwCreateWindow(width, height, "Strelka", nullptr, nullptr);

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    char log[2048]; // For error reporting from OptiX creation functions

    //
    // Initialize CUDA and create OptiX context
    //
    OptixDeviceContext context = nullptr;
    {
        // Initialize CUDA
        cudaFree(0);

        CUcontext cuCtx = 0; // zero means take the current context
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }
    //
    // accel handling
    //
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Triangle build input: simple list of three vertices
        const std::array<glm::float3, 3> vertices = {
            { { -0.5f, -0.5f, 0.0f }, { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.5f, 0.0f } }
        };

        const size_t vertices_size = sizeof(float3) * vertices.size();
        CUdeviceptr d_vertices = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice));

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
        triangle_input.triangleArray.vertexBuffers = &d_vertices;
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(context,
                                    0, // CUDA stream
                                    &accel_options, &triangle_input,
                                    1, // num build inputs
                                    d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                    gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                    nullptr, // emitted property list
                                    0 // num emitted properties
                                    ));

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    }
    //
    // Create module
    //
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 3;
        pipeline_compile_options.numAttributeValues = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur
             // significant performance cost and should only be done during
             // development.
        pipeline_compile_options.exceptionFlags =
            OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        size_t inputSize = 0;
        std::string optixSource;
        readSourceFile(optixSource, "./build/Debug/StrelkaOptiX_generated_optixTriangle.cu.optixir");
        const char* input = optixSource.c_str();
        inputSize = optixSource.size();
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context, &module_compile_options, &pipeline_compile_options, input, inputSize, log, &sizeof_log, &module));
    }

    //
    // Create program groups
    //
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    {
        OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc = {}; //
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &miss_prog_group));

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
                                                1, // num program groups
                                                &program_group_options, log, &sizeof_log, &hitgroup_prog_group));
    }
    //
    // Link pipeline
    //
    OptixPipeline pipeline = nullptr;
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options, program_groups,
                                            sizeof(program_groups) / sizeof(program_groups[0]), log, &sizeof_log,
                                            &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0, // maxCCDepth
                                               0, // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1 // maxTraversableDepth
                                              ));
    }
    //
    // Set up shader binding table
    //
    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        ms_sbt.data = { 0.3f, 0.1f, 0.2f };
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr hitgroup_record;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = 1;
    }

    CUDAOutputBuffer<uchar4> output_buffer(CUDAOutputBufferType::CUDA_DEVICE, width, height);

    //
    // launch
    //
    {
        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        Camera cam;
        configureCamera(cam, width, height);

        Params params;
        params.image = output_buffer.map();
        params.image_width = width;
        params.image_height = height;
        params.handle = gas_handle;
        params.cam_eye = make_float3(cam.eye().x, cam.eye().y, cam.eye().z);

        glm::float3 cam_u, cam_v, cam_w;

        cam.UVWFrame(cam_u, cam_v, cam_w);

        params.cam_u = make_float3(cam_u.x, cam_u.y, cam_u.z);
        params.cam_v = make_float3(cam_v.x, cam_v.y, cam_v.z);
        params.cam_w = make_float3(cam_w.x, cam_w.y, cam_w.z);

        CUdeviceptr d_param;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params, sizeof(params), cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
        CUDA_SYNC_CHECK();

        output_buffer.unmap();
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
    }

    //
    // Display results
    //
    ImageBuffer buffer;
    buffer.data = output_buffer.getHostPointer();
    buffer.width = width;
    buffer.height = height;
    buffer.pixel_format = BufferImageFormat::UNSIGNED_BYTE4;

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
        throw Exception("Failed to initialize GL");

    GLDisplay display(buffer.pixel_format);
    GLuint pbo = 0u;
    GL_CHECK(glGenBuffers(1, &pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, pixelFormatSize(buffer.pixel_format) * buffer.width * buffer.height,
                          buffer.data, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    int framebuf_res_x = 0, framebuf_res_y = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);

        display.display(buffer.width, buffer.height, framebuf_res_x, framebuf_res_y, pbo);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();

    //
    // Cleanup
    //
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
        OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
        OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
        OPTIX_CHECK(optixModuleDestroy(module));

        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }

    return 0;
}
