#pragma once

#include <stdint.h>
#include <vector>

#include <glad/gl.h>

#include <cuda_gl_interop.h>
#include <cuda.h>

#include <vector_types.h>

#include <sstream>
#include <iostream>
#include <assert.h>

namespace oka
{

inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '" << cudaGetErrorString(error) << "' (" << file << ":"
           << line << ")\n";
        std::cerr << ss.str() << std::endl;
        assert(0);
        // throw Exception(ss.str().c_str());
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
        std::cerr << ss.str() << std::endl;
        // throw Exception(ss.str().c_str());
        assert(0);
    }
}

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() cudaSyncCheck(__FILE__, __LINE__)

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
    size_t getHostDataSize();

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

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data = nullptr;
    size_t dataSize = 0;
    unsigned int width = 0;
    unsigned int height = 0;
    BufferImageFormat pixel_format;
};

static void ensureMinimumSize(int& w, int& h)
{
    if (w <= 0)
        w = 1;
    if (h <= 0)
        h = 1;
}

static void ensureMinimumSize(unsigned& w, unsigned& h)
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
            // throw Exception(
            //     "GL interop is only available on display device, please "
            //     "use display device for optimal "
            //     "performance.  Alternatively you can disable GL interop "
            //     "with --no-gl-interop and run with "
            //     "degraded performance.");
            assert(0);
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

    // if (m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P)
    // {
    //     // GL buffer gets resized below
    //     GL_CHECK(glGenBuffers(1, &m_pbo));
    //     GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    //     GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height, nullptr, GL_STREAM_DRAW));
    //     GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

    //     CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_gfx_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));
    // }

    // if (m_type == CUDAOutputBufferType::ZERO_COPY)
    // {
    //     CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(m_host_zcopy_pixels)));
    //     CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&m_host_zcopy_pixels),
    //                              m_width * m_height * sizeof(PIXEL_FORMAT), cudaHostAllocPortable | cudaHostAllocMapped));
    //     CUDA_CHECK(cudaHostGetDevicePointer(
    //         reinterpret_cast<void**>(&m_device_pixels), reinterpret_cast<void*>(m_host_zcopy_pixels), 0 /*flags*/
    //         ));
    // }

    // if (m_type != CUDAOutputBufferType::GL_INTEROP && m_type != CUDAOutputBufferType::CUDA_P2P && m_pbo != 0u)
    // {
    //     GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    //     GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PIXEL_FORMAT) * m_width * m_height, nullptr, GL_STREAM_DRAW));
    //     GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));
    // }

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
        // fprintf( stderr, "glBindBuffer = %p\n", glBindBuffer );

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

template <typename PIXEL_FORMAT>
size_t CUDAOutputBuffer<PIXEL_FORMAT>::getHostDataSize()
{
    return sizeof(PIXEL_FORMAT) * m_width * m_height;
}

} // namespace oka
