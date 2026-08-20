#pragma once

#include <cstdint>
#include <string>


namespace oka::display_output
{

/// Which backend filled a DisplayCapabilities.
///
/// The two backends answer different questions. Vulkan reports swapchain
/// formats, present modes and the compositor's VRR state; Metal reports the EDR
/// headroom the window server currently grants and the CAMetalLayer
/// configuration. Neither set of fields means anything while the other backend
/// is running -- an all-zero PresentCapabilities on macOS says "this is not a
/// Vulkan swapchain", not "this display cannot present" -- so the UI reads this
/// to decide which half of the struct it is allowed to show.
enum class DisplayBackend : uint32_t
{
    Unknown = 0,
    Vulkan,
    Metal,
};

enum class OutputMode : uint32_t
{
    Auto = 0,
    /// The platform's high dynamic range path: an HDR10/ST 2084 swapchain on
    /// Vulkan, extended-range EDR content on Metal. Deliberately not named HDR10
    /// -- macOS has no HDR10 surface for an app to pick, and calling the mode
    /// after the Windows encoding made the Metal UI describe something that does
    /// not exist there.
    HDR,
    SDR,
    /// Reference-white HDR. Metal only: an XDR panel in a reference preset
    /// reports its own headroom over the 100-nit reference white, which is what
    /// grading wants and is not the headroom the compositor hands an ordinary
    /// window. Last in the enum on purpose, so the Vulkan path's clamp turns it
    /// into SDR -- a safe mode -- rather than into an unknown one.
    ReferenceHDR,
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

/// What the display the window currently sits on can do with values above SDR
/// white.
///
/// All three headrooms are multipliers of SDR white, not nits. macOS never tells
/// an application the panel's absolute luminance, because the user's brightness
/// slider moves it: a paper-white-in-nits control, which is the right knob for
/// an HDR10 swapchain, has nothing to configure here.
struct EdrCapabilities
{
    /// Granted right now. Falls when another window claims the backlight or the
    /// panel is thermally limited, so it is the value content must be mapped to.
    float currentHeadroom = 1.0f;
    /// The most this display can ever grant. Used to decide whether an HDR mode
    /// is worth offering at all, never to scale content.
    float potentialHeadroom = 1.0f;
    /// Non-zero only while the display is in a reference preset (Pro Display
    /// XDR, the XDR built-ins). Zero means there is no reference mode to select.
    float referenceHeadroom = 0.0f;
    bool wideGamut = false;
    /// Whether the layer is currently asking for extended-range content.
    bool edrRequested = false;
};

inline bool outputModeSupported(OutputMode mode, const EdrCapabilities& capabilities)
{
    if (mode == OutputMode::ReferenceHDR)
    {
        return capabilities.referenceHeadroom > 1.0f;
    }
    if (mode == OutputMode::HDR)
    {
        return capabilities.potentialHeadroom > 1.0f;
    }
    return true;
}

/// Headroom the tone curve should map its brightest output to.
///
/// Auto and HDR follow the *current* headroom rather than the potential one:
/// rendering past what the compositor grants clips in the compositor, after the
/// tone curve has already spent its range, instead of rolling off inside it.
/// `headroomLimit` is the user's ceiling and is ignored below 1, which is how
/// "no limit" is spelled -- a limit under 1 would be a request for less than SDR
/// white, which no mode means.
inline float selectEdrHeadroom(OutputMode requested, const EdrCapabilities& capabilities, float headroomLimit)
{
    float headroom = 1.0f;

    if (requested == OutputMode::SDR)
    {
        return 1.0f;
    }
    if (requested == OutputMode::ReferenceHDR && capabilities.referenceHeadroom > 1.0f)
    {
        headroom = capabilities.referenceHeadroom;
    }
    else
    {
        headroom = capabilities.currentHeadroom;
    }
    if (headroom < 1.0f)
    {
        headroom = 1.0f;
    }
    if (headroomLimit >= 1.0f && headroomLimit < headroom)
    {
        headroom = headroomLimit;
    }
    return headroom;
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

/// Refresh range a display advertises, turned into a VRR verdict.
///
/// Deliberately never returns Active. macOS tells an app the interval range the
/// panel supports, not the rate the window server is driving it at this instant;
/// claiming the range as proof that VRR is engaged is the same overreach
/// interpretVrrStatus() refuses to make for a Vulkan present mode.
inline VrrStatus interpretRefreshRange(float minRefreshRateHz, float maxRefreshRateHz)
{
    if (maxRefreshRateHz <= 0.0f)
    {
        return VrrStatus::Unknown;
    }
    // Half a hertz of slack: a 120 Hz panel that reports its two intervals as
    // 1/120 and 1/119.88 is fixed-rate rounding, not a variable range.
    if (minRefreshRateHz > 0.0f && maxRefreshRateHz - minRefreshRateHz > 0.5f)
    {
        return VrrStatus::Supported;
    }
    return VrrStatus::Fixed;
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

    /// Which of the blocks above and below carries meaning; see DisplayBackend.
    DisplayBackend backend = DisplayBackend::Unknown;

    // Metal. The window server, not the app, owns the encoding, so what is worth
    // reporting is the headroom granted and how the layer was configured to use
    // it -- there is no surface format negotiation to show.
    EdrCapabilities edr;
    /// Empty when the platform does not name its displays.
    std::string displayName;
    std::string colorSpaceName;
    /// Headroom the tone curve was actually given this frame, after the mode and
    /// the user's ceiling. Shown because it is frequently neither the display's
    /// current headroom nor the requested one.
    float appliedHeadroom = 1.0f;
    /// 0 means "present at the display's own rate".
    float frameRateLimitHz = 0.0f;
    uint32_t maxDrawableCount = 0;
    bool displaySync = true;
};

} // namespace oka::display_output

