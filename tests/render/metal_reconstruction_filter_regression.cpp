#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#include <tinyexr.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace
{
constexpr int kSkipExitCode = 77;

bool renderMean(const std::filesystem::path& cli,
                const std::filesystem::path& config,
                const std::filesystem::path& output,
                const char* filter,
                double& mean)
{
    const std::string command = "\"" + cli.string() + "\" -c \"" + config.string() + "\" --output \"" +
                                output.string() + "\" --width 192 --height 128 --spp 64 --spp-per-launch 4 " +
                                "--reconstruction-filter " + filter + " > /dev/null 2>&1";
    // NOLINTNEXTLINE(bugprone-command-processor,concurrency-mt-unsafe)
    if (std::system(command.c_str()) != 0)
    {
        return false;
    }

    float* pixels = nullptr;
    int width = 0;
    int height = 0;
    const char* error = nullptr;
    if (LoadEXR(&pixels, &width, &height, output.string().c_str(), &error) != TINYEXR_SUCCESS)
    {
        std::printf("FAIL: could not read %s: %s\n", output.string().c_str(), error ? error : "unknown");
        FreeEXRErrorMessage(error);
        return false;
    }
    if (pixels == nullptr || width != 192 || height != 128)
    {
        std::printf("FAIL: unexpected render size for %s\n", filter);
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        free(pixels);
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    double sum = 0.0;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        sum += pixels[i * 4] + pixels[i * 4 + 1] + pixels[i * 4 + 2];
    }
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    free(pixels);
    mean = sum / static_cast<double>(pixelCount * 3u);
    return std::isfinite(mean) && mean > 0.01;
}
} // namespace

int main()
{
    const std::filesystem::path cli = std::filesystem::path(STRELKA_TEST_BINARY_DIR) / "StrelkaCLI";
    const std::filesystem::path config =
        std::filesystem::path(STRELKA_TEST_ASSETS_DIR) / "reconstruction_filter" / "reconstruction_filter.toml";
    if (!std::filesystem::exists(cli) || !std::filesystem::exists(config))
    {
        std::printf("SKIP: StrelkaCLI or reconstruction filter scene is not present\n");
        return kSkipExitCode;
    }

    const std::filesystem::path root = std::filesystem::temp_directory_path();
    const std::filesystem::path box = root / "strelka_filter_regression_box.exr";
    const std::filesystem::path mitchell = root / "strelka_filter_regression_mitchell.exr";
    const std::filesystem::path lanczos = root / "strelka_filter_regression_lanczos.exr";
    double boxMean = 0.0;
    double mitchellMean = 0.0;
    double lanczosMean = 0.0;
    const bool rendered = renderMean(cli, config, box, "box", boxMean) &&
                          renderMean(cli, config, mitchell, "mitchell", mitchellMean) &&
                          renderMean(cli, config, lanczos, "lanczos2", lanczosMean);
    std::error_code ec;
    std::filesystem::remove(box, ec);
    std::filesystem::remove(mitchell, ec);
    std::filesystem::remove(lanczos, ec);
    if (!rendered)
    {
        std::printf("FAIL: could not render or read reconstruction filter chart\n");
        return 1;
    }

    std::printf("Filter chart mean RGB: box %.6f, Mitchell %.6f, Lanczos2 %.6f\n", boxMean, mitchellMean, lanczosMean);
    if (std::abs(mitchellMean / boxMean - 1.0) > 0.1 || std::abs(lanczosMean / boxMean - 1.0) > 0.1)
    {
        std::printf("FAIL: a signed reconstruction filter changed the chart's mean brightness\n");
        return 1;
    }
    return 0;
}
