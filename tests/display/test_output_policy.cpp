#include <doctest/doctest.h>

#include <strelka/display/output_policy.h>

using oka::display_output::EdrCapabilities;
using oka::display_output::OutputCapabilities;
using oka::display_output::OutputMode;
using oka::display_output::PresentCapabilities;
using oka::display_output::PresentMode;
using oka::display_output::SurfaceEncoding;
using oka::display_output::VrrStatus;
// Explicit: interpretRefreshRange takes only floats, so nothing drags the
// namespace in by ADL the way the capability-struct overloads do.
using oka::display_output::interpretRefreshRange;
using oka::display_output::outputModeSupported;
using oka::display_output::selectEdrHeadroom;

TEST_CASE("output mode selection falls back safely")
{
    OutputCapabilities capabilities{};

    CHECK(selectSurfaceEncoding(OutputMode::Auto, capabilities) == SurfaceEncoding::SDR);
    CHECK(selectSurfaceEncoding(OutputMode::HDR, capabilities) == SurfaceEncoding::SDR);

    capabilities.hdr10 = true;
    CHECK(selectSurfaceEncoding(OutputMode::Auto, capabilities) == SurfaceEncoding::HDR10);
    CHECK(selectSurfaceEncoding(OutputMode::HDR, capabilities) == SurfaceEncoding::HDR10);
    CHECK(selectSurfaceEncoding(OutputMode::SDR, capabilities) == SurfaceEncoding::SDR);
}

TEST_CASE("present mode selection retains FIFO correctness baseline")
{
    PresentCapabilities capabilities{};

    capabilities.fifoRelaxed = true;
    capabilities.mailbox = true;
    capabilities.immediate = true;
    capabilities.vrr = true;
    capabilities.vrrOnFifoRelaxed = true;

    CHECK(selectPresentMode(true, capabilities) == PresentMode::Fifo);
    CHECK(selectPresentMode(false, capabilities) == PresentMode::Fifo);
}

TEST_CASE("present mode selection degrades to guaranteed FIFO")
{
    PresentCapabilities capabilities{};

    capabilities.mailbox = false;
    capabilities.vrr = false;

    CHECK(selectPresentMode(true, capabilities) == PresentMode::Fifo);
    CHECK(selectPresentMode(false, capabilities) == PresentMode::Fifo);
}

TEST_CASE("VRR status remains OS and compositor authoritative")
{
    PresentCapabilities const capabilities{};

    CHECK(interpretVrrStatus(false, PresentMode::Fifo, capabilities,
                             VrrStatus::Active) == VrrStatus::Active);
    CHECK(interpretVrrStatus(true, PresentMode::Fifo, capabilities,
                             VrrStatus::Fixed) == VrrStatus::Fixed);
    CHECK(interpretVrrStatus(true, PresentMode::Immediate, capabilities,
                             VrrStatus::Unknown) == VrrStatus::Unknown);
}

TEST_CASE("EDR modes are offered only where the display can honour them")
{
    EdrCapabilities capabilities{};

    CHECK(outputModeSupported(OutputMode::Auto, capabilities));
    CHECK(outputModeSupported(OutputMode::SDR, capabilities));
    CHECK_FALSE(outputModeSupported(OutputMode::HDR, capabilities));
    CHECK_FALSE(outputModeSupported(OutputMode::ReferenceHDR, capabilities));

    capabilities.potentialHeadroom = 4.0f;
    CHECK(outputModeSupported(OutputMode::HDR, capabilities));
    CHECK_FALSE(outputModeSupported(OutputMode::ReferenceHDR, capabilities));

    capabilities.referenceHeadroom = 16.0f;
    CHECK(outputModeSupported(OutputMode::ReferenceHDR, capabilities));
}

TEST_CASE("EDR headroom follows the mode, the display and the user ceiling")
{
    EdrCapabilities capabilities{};

    capabilities.currentHeadroom = 3.0f;
    capabilities.potentialHeadroom = 16.0f;
    capabilities.referenceHeadroom = 16.0f;

    // SDR is absolute: neither a display with headroom nor a ceiling above it
    // may raise the tone curve past white.
    CHECK(selectEdrHeadroom(OutputMode::SDR, capabilities, 0.0f) == 1.0f);
    CHECK(selectEdrHeadroom(OutputMode::SDR, capabilities, 8.0f) == 1.0f);

    // Auto and HDR track what is granted now, not what the panel could reach.
    CHECK(selectEdrHeadroom(OutputMode::Auto, capabilities, 0.0f) == 3.0f);
    CHECK(selectEdrHeadroom(OutputMode::HDR, capabilities, 0.0f) == 3.0f);
    CHECK(selectEdrHeadroom(OutputMode::ReferenceHDR, capabilities, 0.0f) == 16.0f);

    // The ceiling only ever lowers.
    CHECK(selectEdrHeadroom(OutputMode::Auto, capabilities, 2.0f) == 2.0f);
    CHECK(selectEdrHeadroom(OutputMode::Auto, capabilities, 9.0f) == 3.0f);

    // Under 1 is how "no ceiling" is spelled, and must not be read as a request
    // for less than SDR white.
    CHECK(selectEdrHeadroom(OutputMode::Auto, capabilities, 0.25f) == 3.0f);
}

TEST_CASE("EDR headroom never drops below SDR white")
{
    EdrCapabilities capabilities{};

    // A display that is off, asleep or refusing extended content reports 0.
    capabilities.currentHeadroom = 0.0f;
    CHECK(selectEdrHeadroom(OutputMode::Auto, capabilities, 0.0f) == 1.0f);
    CHECK(selectEdrHeadroom(OutputMode::HDR, capabilities, 0.0f) == 1.0f);

    // A reference headroom of 1 is not a reference preset; fall back to what is
    // actually granted rather than pinning the tone curve to 1.
    capabilities.currentHeadroom = 2.0f;
    capabilities.referenceHeadroom = 1.0f;
    CHECK(selectEdrHeadroom(OutputMode::ReferenceHDR, capabilities, 0.0f) == 2.0f);
}

TEST_CASE("refresh range reports variability without claiming it is engaged")
{
    CHECK(interpretRefreshRange(0.0f, 0.0f) == VrrStatus::Unknown);
    CHECK(interpretRefreshRange(60.0f, 60.0f) == VrrStatus::Fixed);
    CHECK(interpretRefreshRange(0.0f, 60.0f) == VrrStatus::Fixed);
    // Rounding, not a range.
    CHECK(interpretRefreshRange(119.88f, 120.0f) == VrrStatus::Fixed);
    // ProMotion.
    CHECK(interpretRefreshRange(47.95f, 120.0f) == VrrStatus::Supported);
    // Never Active: the window server does not tell an app the rate it is
    // driving the panel at.
    CHECK(interpretRefreshRange(47.95f, 120.0f) != VrrStatus::Active);
}
