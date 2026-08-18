#include <doctest/doctest.h>

#include <strelka/display/output_probe.h>

using oka::display_output::PlatformDisplayState;
using oka::display_output::VrrStatus;
using oka::display_output::parseGdctlOutput;

TEST_CASE("gdctl parser selects the named active VRR monitor")
{
    constexpr const char *output = R"(
Monitors:
Monitor DP-1 (Fixed Panel)
   Refresh rate: 60.000
   is-current ⇒  yes
   display-name ⇒  Fixed Panel
Monitor DP-3 (Samsung Electric Company 35")
   Refresh rate: 174.962
   is-current ⇒  yes
   refresh-rate-mode ⇒  variable
   display-name ⇒  Samsung Electric Company 35"
   min-refresh-rate ⇒  48
Logical monitors:
)";
    PlatformDisplayState state;

    state = parseGdctlOutput(output, "Samsung Electric Company 35\"");

    CHECK(state.vrrStatus == VrrStatus::Active);
    CHECK(state.minRefreshRateHz == doctest::Approx(48.0f));
    CHECK(state.maxRefreshRateHz == doctest::Approx(174.962f));
    CHECK(state.currentRefreshRateHz == doctest::Approx(174.962f));
}

TEST_CASE("gdctl parser reports fixed mode on a VRR-capable monitor")
{
    constexpr const char *output = R"(
Monitors:
Monitor HDMI-1 (VRR Display)
   Refresh rate: 120.000
   refresh-rate-mode ⇒  variable
   Refresh rate: 60.000
   is-current ⇒  yes
   min-refresh-rate ⇒  40
Logical monitors:
)";
    PlatformDisplayState state;

    state = parseGdctlOutput(output, "HDMI-1");

    CHECK(state.vrrStatus == VrrStatus::Fixed);
    CHECK(state.minRefreshRateHz == doctest::Approx(40.0f));
    CHECK(state.maxRefreshRateHz == doctest::Approx(60.0f));
    CHECK(state.currentRefreshRateHz == doctest::Approx(60.0f));
}

TEST_CASE("gdctl parser safely rejects an unmatched monitor")
{
    constexpr const char *output = R"(
Monitors:
Monitor DP-3 (Known Display)
   Refresh rate: 144.000
   is-current ⇒  yes
   refresh-rate-mode ⇒  variable
   min-refresh-rate ⇒  48
Monitor HDMI-1 (Other Display)
   Refresh rate: 60.000
   is-current ⇒  yes
Logical monitors:
)";
    PlatformDisplayState state;

    state = parseGdctlOutput(output, "Different Display");

    CHECK(state.vrrStatus == VrrStatus::Unknown);
    CHECK(state.currentRefreshRateHz == 0.0f);
}

TEST_CASE("gdctl parser reports support when no current mode is marked")
{
    constexpr const char *output = R"(
Monitors:
Monitor DP-3 (VRR Display)
   Refresh rate: 144.000
   refresh-rate-mode ⇒  variable
   min-refresh-rate ⇒  48
Logical monitors:
)";
    PlatformDisplayState state;

    state = parseGdctlOutput(output, "VRR Display");

    CHECK(state.vrrStatus == VrrStatus::Supported);
    CHECK(state.maxRefreshRateHz == doctest::Approx(144.0f));
    CHECK(state.currentRefreshRateHz == 0.0f);
}
