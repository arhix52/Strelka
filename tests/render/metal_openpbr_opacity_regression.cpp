#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 1
#include <tinyexr.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/wait.h>

namespace
{
namespace fs = std::filesystem;

int render(const fs::path& cli, const fs::path& scene, const fs::path& output)
{
    const std::string command = "\"" + cli.string() + "\" \"" + scene.string() + "\" -o \"" + output.string() +
                                "\" -w 96 --height 72 --spp 16 --spp-per-launch 4 --depth 3 --sampler sobol "
                                "> /dev/null 2>&1";
    // NOLINTNEXTLINE(bugprone-command-processor,concurrency-mt-unsafe)
    const int status = std::system(command.c_str());
    return status != -1 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::vector<float> readExr(const fs::path& path)
{
    float* rgba = nullptr;
    int width = 0;
    int height = 0;
    const char* error = nullptr;
    if (LoadEXR(&rgba, &width, &height, path.string().c_str(), &error) != TINYEXR_SUCCESS)
    {
        if (error)
        {
            FreeEXRErrorMessage(error);
        }
        return {};
    }
    std::vector<float> result(rgba, rgba + static_cast<size_t>(width) * height * 4u);
    std::free(rgba);
    return result;
}

double meanDifference(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.empty() || a.size() != b.size())
    {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i += 4u)
    {
        for (size_t c = 0; c < 3u; ++c)
        {
            sum += std::abs(static_cast<double>(a[i + c]) - b[i + c]);
        }
    }
    return sum / static_cast<double>(a.size()) * (4.0 / 3.0);
}

void writeSidecar(const fs::path& path, const char* opacity)
{
    std::ofstream stream(path);
    stream << "{\"version\":1,\"materials\":[{\"gltfMaterial\":\"RedDiffuse\",\"openpbr\":{"
              "\"base_color\":[0.55,0.06,0.05],\"geometry_opacity\":"
           << opacity << "}}]}\n";
}
} // namespace

int main()
{
    const fs::path cli = fs::path(STRELKA_TEST_BINARY_DIR) / "StrelkaCLI";
    const fs::path assets = STRELKA_TEST_ASSETS_DIR;
    if (!fs::exists(cli) || !fs::exists(assets / "openpbr/openpbr_cornell.glb"))
    {
        return 77;
    }
    const fs::path root =
        fs::temp_directory_path() /
        ("strelka-openpbr-opacity-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const fs::path scene = root / "opacity.glb";
    const fs::path sidecar = root / "opacity_openpbr.json";
    fs::copy_file(assets / "openpbr/openpbr_cornell.glb", scene);
    fs::copy_file(assets / "openpbr/openpbr_cornell_light.json", root / "opacity_light.json");
    fs::copy_file(assets / "projector/slide.png", root / "opacity.png");

    writeSidecar(sidecar, "1.0");
    const int opaqueStatus = render(cli, scene, root / "opaque.exr");
    if (opaqueStatus == 3)
    {
        std::error_code error;
        fs::remove_all(root, error);
        return 77;
    }
    writeSidecar(sidecar, "0.0");
    const int zeroStatus = render(cli, scene, root / "zero.exr");
    {
        std::ofstream stream(sidecar);
        stream << "{\"version\":1,\"materials\":[{\"gltfMaterial\":\"RedDiffuse\","
                  "\"openpbr\":{\"base_color\":[0.55,0.06,0.05]},"
                  "\"textures\":{\"geometry_opacity\":\"opacity.png\"}}]}\n";
    }
    const int mapStatus = render(cli, scene, root / "mapped.exr");

    const auto opaque = readExr(root / "opaque.exr");
    const auto zero = readExr(root / "zero.exr");
    const auto mapped = readExr(root / "mapped.exr");
    const double opaqueToZero = meanDifference(opaque, zero);
    const double opaqueToMapped = meanDifference(opaque, mapped);
    const double mappedToZero = meanDifference(mapped, zero);
    std::printf("OpenPBR opacity mean RGB differences: opaque/zero %.6f, opaque/map %.6f, map/zero %.6f\n",
                opaqueToZero, opaqueToMapped, mappedToZero);
    std::error_code error;
    fs::remove_all(root, error);
    if (opaqueStatus != 0 || zeroStatus != 0 || mapStatus != 0 || opaque.empty() || zero.empty() || mapped.empty())
    {
        std::printf("FAIL: OpenPBR opacity render did not complete\n");
        return 1;
    }
    if (opaqueToZero < 0.001 || opaqueToMapped < 0.001 || mappedToZero < 0.001)
    {
        std::printf("FAIL: native OpenPBR constant or red-channel map did not affect geometry coverage\n");
        return 1;
    }
    return 0;
}
