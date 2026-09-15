#pragma once

#include <cuda.h>

#include <cstddef>
#include <type_traits>

namespace oka::optix
{

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
