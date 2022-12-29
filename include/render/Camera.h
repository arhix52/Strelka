#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/compatibility.hpp>

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
