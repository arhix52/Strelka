#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <sys/wait.h>

namespace
{
namespace fs = std::filesystem;

int run(const fs::path& cli, const fs::path& scene, const fs::path& output, const std::string& options)
{
    const std::string command = "\"" + cli.string() + "\" \"" + scene.string() + "\" -o \"" + output.string() +
                                "\" -w 64 --height 48 --spp 16 --spp-per-launch 4 --depth 3 " + options +
                                " > /dev/null 2>&1";
    // NOLINTNEXTLINE(bugprone-command-processor,concurrency-mt-unsafe)
    const int status = std::system(command.c_str());
    return status != -1 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string bytes(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}
} // namespace

int main()
{
    const fs::path cli = fs::path(STRELKA_TEST_BINARY_DIR) / "StrelkaCLI";
    const fs::path scene = fs::path(STRELKA_TEST_ASSETS_DIR) / "cornell_box/cornell_box.glb";
    if (!fs::exists(cli) || !fs::exists(scene))
    {
        return 77;
    }
    const fs::path root =
        fs::temp_directory_path() /
        ("strelka-resume-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const fs::path staged = root / "staged.exr";
    const fs::path state = root / "staged.checkpoint.stc";
    const fs::path resumed = root / "resumed.exr";
    const fs::path reference = root / "reference.exr";
    const int first = run(cli, scene, staged, "--checkpoint-spp 8");
    if (first == 3)
    {
        std::error_code error;
        fs::remove_all(root, error);
        return 77;
    }
    const int second = run(cli, scene, resumed, "--resume \"" + state.string() + "\" --postprocess-package");
    const int third = run(cli, scene, reference, "");
    const int rejected =
        run(cli, scene, root / "rejected.exr", "--resume \"" + state.string() + "\" --reconstruction-filter box");
    const std::string resumedBytes = bytes(resumed);
    const std::string referenceBytes = bytes(reference);
    const bool packageWritten =
        fs::exists(root / "resumed.preview.png") && fs::exists(root / "resumed.render.json") &&
        bytes(root / "resumed.render.json").find("\"scene-linear Strelka RGB\"") != std::string::npos;
    std::error_code error;
    fs::remove_all(root, error);
    if (first != 0 || second != 0 || third != 0 || rejected == 0 || resumedBytes.empty() ||
        resumedBytes != referenceBytes || !packageWritten)
    {
        std::printf("FAIL: checkpoint resume is not identical to uninterrupted accumulation, or package is missing\n");
        return 1;
    }
    std::printf("Checkpoint resume: identical 16-spp EXR; mismatched filter rejected; package written\n");
    return 0;
}
