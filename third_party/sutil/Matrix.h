// Lightweight row-major matrix types for GPU (CUDA) and host code.
// Replaces the NVIDIA sutil Matrix.h template with only the operations
// actually used in Strelka: construct, mat*vec, scalar*mat, mat+mat,
// getData(), and make_matrix3x3().

#pragma once

#include <sutil/Preprocessor.h>
#include <sutil/vec_math.h>

#if !defined(__CUDACC_RTC__)
#include <initializer_list>
#endif

namespace sutil
{

struct Matrix4x4
{
    float m[16];

    SUTIL_HOSTDEVICE Matrix4x4() {}

    SUTIL_HOSTDEVICE explicit Matrix4x4(const float data[16])
    {
        for (int i = 0; i < 16; ++i)
            m[i] = data[i];
    }

#if !defined(__CUDACC_RTC__)
    SUTIL_HOSTDEVICE Matrix4x4(const std::initializer_list<float>& list)
    {
        int i = 0;
        for (auto it = list.begin(); it != list.end(); ++it)
            m[i++] = *it;
    }
#endif

    SUTIL_HOSTDEVICE float  operator[](unsigned int i) const { return m[i]; }
    SUTIL_HOSTDEVICE float& operator[](unsigned int i) { return m[i]; }

    SUTIL_HOSTDEVICE float*       getData() { return m; }
    SUTIL_HOSTDEVICE const float* getData() const { return m; }
};

struct Matrix3x3
{
    float m[9];

    SUTIL_HOSTDEVICE Matrix3x3() {}

    SUTIL_HOSTDEVICE explicit Matrix3x3(const float data[9])
    {
        for (int i = 0; i < 9; ++i)
            m[i] = data[i];
    }

#if !defined(__CUDACC_RTC__)
    SUTIL_HOSTDEVICE Matrix3x3(const std::initializer_list<float>& list)
    {
        int i = 0;
        for (auto it = list.begin(); it != list.end(); ++it)
            m[i++] = *it;
    }
#endif

    SUTIL_HOSTDEVICE float  operator[](unsigned int i) const { return m[i]; }
    SUTIL_HOSTDEVICE float& operator[](unsigned int i) { return m[i]; }

    SUTIL_HOSTDEVICE float*       getData() { return m; }
    SUTIL_HOSTDEVICE const float* getData() const { return m; }
};

// Matrix4x4 * float4 (row-major)
SUTIL_INLINE SUTIL_HOSTDEVICE float4 operator*(const Matrix4x4& mat, const float4& v)
{
    float4 r;
    r.x = mat[0] * v.x + mat[1] * v.y + mat[ 2] * v.z + mat[ 3] * v.w;
    r.y = mat[4] * v.x + mat[5] * v.y + mat[ 6] * v.z + mat[ 7] * v.w;
    r.z = mat[8] * v.x + mat[9] * v.y + mat[10] * v.z + mat[11] * v.w;
    r.w = mat[12] * v.x + mat[13] * v.y + mat[14] * v.z + mat[15] * v.w;
    return r;
}

// Matrix3x3 * float3 (row-major)
SUTIL_INLINE SUTIL_HOSTDEVICE float3 operator*(const Matrix3x3& mat, const float3& v)
{
    float3 r;
    r.x = mat[0] * v.x + mat[1] * v.y + mat[2] * v.z;
    r.y = mat[3] * v.x + mat[4] * v.y + mat[5] * v.z;
    r.z = mat[6] * v.x + mat[7] * v.y + mat[8] * v.z;
    return r;
}

// scalar * Matrix4x4
SUTIL_INLINE SUTIL_HOSTDEVICE Matrix4x4 operator*(float f, const Matrix4x4& mat)
{
    Matrix4x4 r;
    for (int i = 0; i < 16; ++i)
        r[i] = f * mat[i];
    return r;
}

// Matrix4x4 + Matrix4x4
SUTIL_INLINE SUTIL_HOSTDEVICE Matrix4x4 operator+(const Matrix4x4& a, const Matrix4x4& b)
{
    Matrix4x4 r;
    for (int i = 0; i < 16; ++i)
        r[i] = a[i] + b[i];
    return r;
}

// Extract upper-left 3x3 from a 4x4 matrix
SUTIL_INLINE SUTIL_HOSTDEVICE Matrix3x3 make_matrix3x3(const Matrix4x4& mat)
{
    Matrix3x3 r;
    r[0] = mat[0]; r[1] = mat[1]; r[2] = mat[ 2];
    r[3] = mat[4]; r[4] = mat[5]; r[5] = mat[ 6];
    r[6] = mat[8]; r[7] = mat[9]; r[8] = mat[10];
    return r;
}

} // namespace sutil
