#pragma once

#include <cstdint>

namespace oka
{
namespace display_output
{

enum class OutputMode : uint32_t
{
    Auto = 0,
    HDR10,
    SDR,
};

enum class SurfaceEncoding : uint32_t
{
    SDR = 0,
    HDR10,
};

struct OutputCapabilities
{
    bool hdr10 = false;
    bool hdrSelected = false;
    bool hdrMetadata = false;
    bool swapchainColorspace = false;
};

inline SurfaceEncoding selectSurfaceEncoding(OutputMode requested, const OutputCapabilities& capabilities)
{
    if (requested == OutputMode::SDR)
    {
        return SurfaceEncoding::SDR;
    }
    if (capabilities.hdr10)
    {
        return SurfaceEncoding::HDR10;
    }
    return SurfaceEncoding::SDR;
}

enum class PresentMode : uint32_t
{
    Fifo = 0,
    FifoRelaxed,
    Mailbox,
    Immediate,
};

struct PresentCapabilities
{
    bool fifo = true;
    bool fifoRelaxed = false;
    bool mailbox = false;
    bool immediate = false;
    bool vrr = false;
    bool vrrOnFifo = false;
    bool vrrOnFifoRelaxed = false;
};

inline PresentMode selectPresentMode(bool requestVrr, const PresentCapabilities& capabilities)
{
    (void)requestVrr;
    (void)capabilities;
    // FIFO is the Vulkan correctness baseline. Present modes, including
    // IMMEDIATE, do not prove that the OS or compositor has enabled VRR.
    return PresentMode::Fifo;
}

enum class VrrStatus : uint32_t
{
    Unknown = 0,
    Supported,
    Active,
    Fixed,
};

inline const char *vrrStatusName(VrrStatus status)
{
    if (status == VrrStatus::Supported)
    {
        return "Supported";
    }
    if (status == VrrStatus::Active)
    {
        return "Active";
    }
    if (status == VrrStatus::Fixed)
    {
        return "Fixed";
    }
    return "Unknown";
}

inline VrrStatus interpretVrrStatus(bool requested,
                                    PresentMode selected,
                                    const PresentCapabilities& capabilities,
                                    VrrStatus platformStatus)
{
    (void)requested;
    (void)selected;
    (void)capabilities;
    // The compositor or OS owns VRR state. An application preference and a
    // Vulkan present mode cannot promote an unknown platform result.
    return platformStatus;
}

struct DisplayCapabilities
{
    OutputCapabilities output;
    PresentCapabilities present;
    SurfaceEncoding surfaceEncoding = SurfaceEncoding::SDR;
    PresentMode presentMode = PresentMode::Fifo;
    VrrStatus vrrStatus = VrrStatus::Unknown;
    float minRefreshRateHz = 0.0f;
    float maxRefreshRateHz = 0.0f;
    float currentRefreshRateHz = 0.0f;
    bool presentWait = false;
    bool presentId = false;
    bool displayTiming = false;
};

} // namespace display_output
} // namespace oka
