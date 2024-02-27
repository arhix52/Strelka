#include <display/Display.h>

#include <render/common.h>
#include <render/buffer.h>

#include <render/Render.h>

#include <log.h>
#include <logmanager.h>
#include <cxxopts.hpp>
#include <filesystem>

int main(int argc, const char* argv[])
{
    const oka::Logmanager loggerManager;
    cxxopts::Options options("Strelka -s <USD Scene path>", "commands");

    // clang-format off
    options.add_options()
        ("s, scene", "scene path", cxxopts::value<std::string>()->default_value(""))
        ("i, iteration", "Iteration to capture", cxxopts::value<int32_t>()->default_value("-1"))
        ("h, help", "Print usage")("t, spp_total", "spp total", cxxopts::value<int32_t>()->default_value("64"))
        ("f, spp_subframe", "spp subframe", cxxopts::value<int32_t>()->default_value("1"))
        ("c, need_screenshot", "Screenshot after spp total", cxxopts::value<bool>()->default_value("false"))
        ("v, validation", "Enable Validation", cxxopts::value<bool>()->default_value("false"));
    // clang-format on  
    options.parse_positional({ "s" });
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // check params
    const std::string usdFile(result["s"].as<std::string>());
    if (usdFile.empty())
    {
        STRELKA_FATAL("Specify usd file name");
        return 1;
    }
    if (!std::filesystem::exists(usdFile))
    {
        STRELKA_FATAL("Specified usd file: {} doesn't exist", usdFile.c_str());
        return -1;
    }
    const std::filesystem::path usdFilePath = { usdFile.c_str() };
    const std::string resourceSearchPath = usdFilePath.parent_path().string();


    auto* ctx = new oka::SharedContext();
        // Set up rendering context.
    uint32_t imageWidth = 1024;
    uint32_t imageHeight = 768;
    ctx->mSettingsManager = new oka::SettingsManager();

    ctx->mSettingsManager->setAs<uint32_t>("render/width", imageWidth);
    ctx->mSettingsManager->setAs<uint32_t>("render/height", imageHeight);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/depth", 4);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/sppTotal", result["t"].as<int32_t>());
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/spp", result["f"].as<int32_t>());
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/iteration", 0);
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", 0); // 0 - none, 1 - random, 2 -
                                                                                   // stratified sampling, 3 - optimized
                                                                                   // stratified sampling
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0); // 0 - reinhard, 1 - aces, 2 - filmic
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
    ctx->mSettingsManager->setAs<float>("render/cameraSpeed", 1.0f);
    ctx->mSettingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
    ctx->mSettingsManager->setAs<bool>("render/pt/enableUpscale", true);
    ctx->mSettingsManager->setAs<bool>("render/pt/enableAcc", true);
    ctx->mSettingsManager->setAs<bool>("render/pt/enableTonemap", true);
    ctx->mSettingsManager->setAs<bool>("render/pt/isResized", false);
    ctx->mSettingsManager->setAs<bool>("render/pt/needScreenshot", false);
    ctx->mSettingsManager->setAs<bool>("render/pt/screenshotSPP", result["c"].as<bool>());
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", 0);
    ctx->mSettingsManager->setAs<bool>("render/enableValidation", result["v"].as<bool>());
    ctx->mSettingsManager->setAs<std::string>("resource/searchPath", resourceSearchPath);
    // Postprocessing settings:
    ctx->mSettingsManager->setAs<float>("render/post/tonemapper/filmIso", 100.0f);
    ctx->mSettingsManager->setAs<float>("render/post/tonemapper/cm2_factor", 1.0f);
    ctx->mSettingsManager->setAs<float>("render/post/tonemapper/fStop", 4.0f);
    ctx->mSettingsManager->setAs<float>("render/post/tonemapper/shutterSpeed", 100.0f);

    ctx->mSettingsManager->setAs<float>("render/post/gamma", 2.4f); // 0.0f - off
    // Dev settings:
    ctx->mSettingsManager->setAs<float>("render/pt/dev/shadowRayTmin", 0.0f); // offset to avoid self-collision in light
                                                                              // sampling
    ctx->mSettingsManager->setAs<float>("render/pt/dev/materialRayTmin", 0.0f); // offset to avoid self-collision in

    oka::Display* display = oka::DisplayFactory::createDisplay();


    display->init(imageWidth, imageHeight, ctx);

    oka::RenderType type = oka::RenderType::eOptiX;
    oka::Render* render = oka::RenderFactory::createRender(type);
    oka::Scene scene;
    
    oka::Camera camera;
    camera.name = "Main";
    camera.fov = 45;
    camera.position = glm::vec3(0, 0, -10);
    camera.mOrientation = glm::quat(glm::vec3(0,0,0));
    camera.updateViewMatrix();
    scene.addCamera(camera);

    render->setScene(&scene);
    render->setSharedContext(ctx);
    render->init();
    ctx->mRender = render;

    oka::BufferDesc desc{};
    desc.format = oka::BufferFormat::FLOAT4;
    desc.width = imageWidth;
    desc.height = imageHeight;

    oka::Buffer* outputBuffer = ctx->mRender->createBuffer(desc);



    while (!display->windowShouldClose())
    {
        auto start = std::chrono::high_resolution_clock::now();

        display->pollEvents();

        display->onBeginFrame();

        render->render(outputBuffer);
        oka::ImageBuffer outputImage;
        outputImage.data = outputBuffer->getHostPointer();
        outputImage.dataSize = outputBuffer->getHostDataSize();
        outputImage.height = outputBuffer->height();
        outputImage.width = outputBuffer->width();
        outputImage.pixel_format = oka::BufferFormat::FLOAT4;
        display->drawFrame(outputImage); // blit rendered image to swapchain
        display->drawUI(); // render ui to swapchain image in window resolution
        display->onEndFrame(); // submit command buffer and present
        const uint32_t currentSpp = ctx->mSubframeIndex;
        auto finish = std::chrono::high_resolution_clock::now();
        const double frameTime = std::chrono::duration<double, std::milli>(finish - start).count();

        display->setWindowTitle((std::string("Strelka") + " [" + std::to_string(frameTime) + " ms]" + " [" +
                                 std::to_string(currentSpp) + " spp]")
                                    .c_str());    }
    display->destroy();
    return 0;
}
