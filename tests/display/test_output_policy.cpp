#include <doctest/doctest.h>

#include <strelka/display/output_policy.h>

using oka::display_output::OutputCapabilities;
using oka::display_output::OutputMode;
using oka::display_output::PresentCapabilities;
using oka::display_output::PresentMode;
using oka::display_output::SurfaceEncoding;
using oka::display_output::VrrStatus;

TEST_CASE("output mode selection falls back safely")
{
    OutputCapabilities capabilities{};

    CHECK(selectSurfaceEncoding(OutputMode::Auto, capabilities) == SurfaceEncoding::SDR);
    CHECK(selectSurfaceEncoding(OutputMode::HDR10, capabilities) == SurfaceEncoding::SDR);

    capabilities.hdr10 = true;
    CHECK(selectSurfaceEncoding(OutputMode::Auto, capabilities) == SurfaceEncoding::HDR10);
    CHECK(selectSurfaceEncoding(OutputMode::HDR10, capabilities) == SurfaceEncoding::HDR10);
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
