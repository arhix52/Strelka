#pragma once

#include <cuda.h>

#include <cstddef>
#include <type_traits>

namespace oka::optix
{

/// The three conversions between a device address and a pointer-shaped thing,
/// in one place.
///
/// A CUdeviceptr is an integer that names memory in the *device's* address
/// space. The host stores it, does arithmetic on it and hands it to the driver,
/// but never dereferences it. performance-no-int-to-ptr is written about
/// integers that become host pointers -- where the compiler loses the
/// provenance it needs for alias analysis -- which is not what happens here, and
/// there is no way to express the CUDA and OptiX entry points without the cast.
///
/// So the suppression is here, once, with the reason attached, rather than at
/// the seventy call sites in OptixRender.cpp and OptixDenoiser.cpp that would
/// otherwise each carry a bare NOLINT nobody could audit.

static_assert(sizeof(CUdeviceptr) == sizeof(void*),
              "deviceAllocTarget() lets cudaMalloc write a void* into CUdeviceptr storage");

/// A device address as a typed pointer. `T` may be cv-qualified: devicePtr<const
/// void>(p) is what the source side of a cudaMemcpy wants.
template <typename T>
inline T* devicePtr(CUdeviceptr ptr)
{
    // NOLINTNEXTLINE(performance-no-int-to-ptr,cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<T*>(ptr);
}

/// The out-parameter for cudaMalloc, whether the caller keeps the result as a
/// CUdeviceptr or as a typed device pointer inside Params. Pointer-to-pointer,
/// not integer-to-pointer -- the address handed to the driver is that of host
/// storage, which is why this one is only about spelling.
inline void** deviceAllocTarget(CUdeviceptr& ptr)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<void**>(&ptr);
}

template <typename T>
inline void** deviceAllocTarget(T*& ptr)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<void**>(&ptr);
}

/// The reverse: a device address the driver handed back as void*, as the integer
/// the rest of the backend passes around.
inline CUdeviceptr deviceAddress(void* ptr)
{
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<CUdeviceptr>(ptr);
}

} // namespace oka::optix
