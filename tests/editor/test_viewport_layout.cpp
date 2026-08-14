#include <doctest/doctest.h>

#include "editor_viewport_layout.h"

using namespace oka::editor_viewport;

TEST_CASE("preview presets are unique and include the default")
{
    CHECK(findPreset(kDefaultPreviewWidth, kDefaultPreviewHeight) >= 0);
    for (size_t i = 0; i < kPreviewPresets.size(); ++i)
    {
        CHECK(kPreviewPresets[i].width > 0);
        CHECK(kPreviewPresets[i].height > 0);
        for (size_t j = i + 1; j < kPreviewPresets.size(); ++j)
        {
            const bool dimensionsMatch = kPreviewPresets[i].width == kPreviewPresets[j].width &&
                                         kPreviewPresets[i].height == kPreviewPresets[j].height;
            CHECK_FALSE(dimensionsMatch);
        }
    }
}

TEST_CASE("fit presentation preserves aspect and centers letterboxing")
{
    const Rect panel = { { 0.0f, 0.0f }, { 1000.0f, 800.0f } };
    const Layout layout = computeLayout(panel, 1920, 1080, { 1.0f, 1.0f }, PresentationMode::Fit);

    CHECK(layout.imageRect.width() == doctest::Approx(1000.0f));
    CHECK(layout.imageRect.height() == doctest::Approx(562.5f));
    CHECK(layout.imageRect.min.y == doctest::Approx(118.75f));
    CHECK(layout.imageRect.width() / layout.imageRect.height() == doctest::Approx(16.0f / 9.0f));
    CHECK(layout.visibleRect.min.x == doctest::Approx(layout.imageRect.min.x));
    CHECK(layout.visibleRect.max.y == doctest::Approx(layout.imageRect.max.y));
}

TEST_CASE("fit presentation rejects black bars and maps image coordinates")
{
    const Layout layout =
        computeLayout({ { 10.0f, 20.0f }, { 1010.0f, 820.0f } }, 1920, 1080, { 1.0f, 1.0f }, PresentationMode::Fit);

    CHECK_FALSE(screenToImageUv({ 500.0f, 30.0f }, layout).has_value());
    const std::optional<Point> center = screenToImageUv({ (layout.imageRect.min.x + layout.imageRect.max.x) * 0.5f,
                                                          (layout.imageRect.min.y + layout.imageRect.max.y) * 0.5f },
                                                        layout);
    REQUIRE(center.has_value());
    const Point centerValue = center.value_or(Point{});
    CHECK(centerValue.x == doctest::Approx(0.5f));
    CHECK(centerValue.y == doctest::Approx(0.5f));
}

TEST_CASE("one to one maps texture pixels to framebuffer pixels")
{
    const Layout layout =
        computeLayout({ { 0.0f, 0.0f }, { 800.0f, 600.0f } }, 960, 540, { 2.0f, 2.0f }, PresentationMode::OneToOne);

    CHECK(layout.imageRect.width() == doctest::Approx(480.0f));
    CHECK(layout.imageRect.height() == doctest::Approx(270.0f));
    CHECK(layout.imageRect.min.x == doctest::Approx(160.0f));
    CHECK(layout.imageRect.min.y == doctest::Approx(165.0f));
}

TEST_CASE("fill presentation crops without distorting image coordinates")
{
    const Rect panel = { { 0.0f, 0.0f }, { 600.0f, 800.0f } };
    const Layout layout = computeLayout(panel, 1920, 1080, { 1.0f, 1.0f }, PresentationMode::Fill);

    CHECK(layout.imageRect.height() == doctest::Approx(800.0f));
    CHECK(layout.imageRect.width() == doctest::Approx(800.0f * 16.0f / 9.0f));
    CHECK(layout.imageRect.min.x < panel.min.x);
    CHECK(layout.visibleRect.min.x == doctest::Approx(panel.min.x));
    CHECK(layout.visibleRect.max.x == doctest::Approx(panel.max.x));

    const std::optional<Point> center = screenToImageUv({ 300.0f, 400.0f }, layout);
    REQUIRE(center.has_value());
    const Point centerValue = center.value_or(Point{});
    CHECK(centerValue.x == doctest::Approx(0.5f));
    CHECK(centerValue.y == doctest::Approx(0.5f));
}

TEST_CASE("invalid viewport dimensions never produce a usable layout")
{
    const Layout zeroPanel =
        computeLayout({ { 0.0f, 0.0f }, { 0.0f, 600.0f } }, 960, 540, { 1.0f, 1.0f }, PresentationMode::Fit);
    const Layout zeroImage =
        computeLayout({ { 0.0f, 0.0f }, { 800.0f, 600.0f } }, 0, 540, { 1.0f, 1.0f }, PresentationMode::Fit);

    CHECK_FALSE(zeroPanel.imageRect.valid());
    CHECK_FALSE(zeroImage.imageRect.valid());
    CHECK_FALSE(screenToImageUv({ 0.0f, 0.0f }, zeroPanel).has_value());
}

TEST_CASE("preview dimensions are clamped to supported limits")
{
    CHECK(clampPreviewDimension(-1) == kMinPreviewDimension);
    CHECK(clampPreviewDimension(960) == 960);
    CHECK(clampPreviewDimension(100000) == kMaxPreviewDimension);
}
