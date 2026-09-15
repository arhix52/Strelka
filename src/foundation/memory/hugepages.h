#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace oka
{

inline void adviseHugePages([[maybe_unused]] void* data, [[maybe_unused]] size_t bytes)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr size_t kHugePage = size_t{ 2 } * 1024 * 1024;
    if (data == nullptr || bytes < kHugePage)
    {
        return;
    }
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* const bytesBegin = static_cast<unsigned char*>(data);
    const size_t misalignment = reinterpret_cast<uintptr_t>(bytesBegin) % page;
    const size_t skip = misalignment == 0 ? 0 : page - misalignment;
    if (skip >= bytes)
    {
        return;
    }
    const size_t length = (bytes - skip) & ~(page - 1);
    if (length < kHugePage)
    {
        return;
    }
    (void)madvise(bytesBegin + skip, length, MADV_HUGEPAGE);
#endif
}

} // namespace oka
