#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace oka
{

/// Ask for transparent huge pages over an allocation that is about to be
/// written end to end.
///
/// A scene load faults in every byte of what it allocates exactly once, and at
/// 4 KiB a page that is a fault per page: loading the pine forest took 2.0 M
/// minor faults and spent 2.25 s of its 5.1 s in the kernel taking them. The
/// buffers this is called on are hundreds of megabytes to gigabytes, which is
/// where a 2 MiB page removes 511 of every 512 of those faults.
///
/// Distributions ship transparent_hugepage/enabled as `madvise`, so nothing
/// gets a huge page without asking -- this is the ask. Best-effort by
/// construction: the kernel is free to decline, the mode may be `never`, and
/// the call does not exist off Linux. A failure costs the small pages that
/// would have been used anyway, so the result is deliberately ignored.
inline void adviseHugePages([[maybe_unused]] void* data, [[maybe_unused]] size_t bytes)
{
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr size_t kHugePage = size_t{ 2 } * 1024 * 1024;
    if (data == nullptr || bytes < kHugePage)
    {
        return;
    }
    // Aligned up at the start, down at the end, because madvise() rejects an
    // unaligned address outright -- and what this is called with comes from
    // operator new, which hands back a pointer a header's width into the
    // kernel's mapping and so is never page-aligned. Passing it raw returns
    // EINVAL and advises nothing, which is what the first version of this did:
    // the counters did not move and /proc/<pid>/smaps_rollup still read
    // AnonHugePages: 0 kB.
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
