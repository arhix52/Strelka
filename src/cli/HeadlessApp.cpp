#include "HeadlessApp.h"

#include <log.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>

#include <tinyexr.h>
#include <stb_image_write.h>

#include <toml++/toml.hpp>

namespace fs = std::filesystem;

namespace oka
{

// ---------------------------------------------------------------------------
// TOML config parsing
// ---------------------------------------------------------------------------

static uint32_t parseIntegrator(const std::string& s)
{
    if (s == "pt")   return 0;
    if (s == "bdpt") return 1;
    if (s == "vcm")  return 2;
    STRELKA_WARNING("Unknown integrator '{}', defaulting to pt", s);
    return 0;
}

static uint32_t parseSampler(const std::string& s)
{
    if (s == "halton") return 0;
    if (s == "pcg")    return 1;
    if (s == "sobol")  return 2;
    STRELKA_WARNING("Unknown sampler '{}', defaulting to halton", s);
    return 0;
}

static uint32_t parseTonemap(const std::string& s)
{
    if (s == "none")     return 0;
    if (s == "reinhard") return 1;
    if (s == "aces")     return 2;
    if (s == "filmic")   return 3;
    STRELKA_WARNING("Unknown tonemap '{}', defaulting to aces", s);
    return 2;
}

RenderConfig parseTomlConfig(const std::string& tomlPath)
{
    RenderConfig cfg;

    toml::table tbl = toml::parse_file(tomlPath);

    // [scene]
    if (auto v = tbl["scene"]["path"].value<std::string>())
        cfg.scenePath = *v;

    // [output]
    if (auto v = tbl["output"]["path"].value<std::string>())
        cfg.outputPath = *v;
    if (auto v = tbl["output"]["width"].value<int64_t>())
        cfg.width = static_cast<uint32_t>(*v);
    if (auto v = tbl["output"]["height"].value<int64_t>())
        cfg.height = static_cast<uint32_t>(*v);

    // [render]
    if (auto v = tbl["render"]["integrator"].value<std::string>())
        cfg.integrator = parseIntegrator(*v);
    if (auto v = tbl["render"]["spp"].value<int64_t>())
        cfg.spp = static_cast<uint32_t>(*v);
    if (auto v = tbl["render"]["spp_per_launch"].value<int64_t>())
        cfg.sppPerLaunch = static_cast<uint32_t>(*v);
    if (auto v = tbl["render"]["max_depth"].value<int64_t>())
        cfg.maxDepth = static_cast<uint32_t>(*v);
    if (auto v = tbl["render"]["sampler"].value<std::string>())
        cfg.samplerType = parseSampler(*v);

    // [camera]
    if (auto v = tbl["camera"]["index"].value<int64_t>())
        cfg.cameraIndex = static_cast<int>(*v);
    if (auto arr = tbl["camera"]["position"].as_array())
    {
        if (arr->size() == 3)
        {
            cfg.cameraPosition = glm::vec3(
                arr->get(0)->value_or(0.0), arr->get(1)->value_or(0.0), arr->get(2)->value_or(0.0));
        }
    }
    if (auto arr = tbl["camera"]["target"].as_array())
    {
        if (arr->size() == 3)
        {
            cfg.cameraTarget = glm::vec3(
                arr->get(0)->value_or(0.0), arr->get(1)->value_or(0.0), arr->get(2)->value_or(0.0));
        }
    }
    if (auto v = tbl["camera"]["fov"].value<double>())
        cfg.cameraFov = static_cast<float>(*v);

    // [tonemap]
    if (auto v = tbl["tonemap"]["type"].value<std::string>())
        cfg.tonemapType = parseTonemap(*v);
    if (auto v = tbl["tonemap"]["gamma"].value<double>())
        cfg.gamma = static_cast<float>(*v);
    if (auto v = tbl["tonemap"]["exposure_iso"].value<double>())
        cfg.filmIso = static_cast<float>(*v);
    if (auto v = tbl["tonemap"]["exposure_fstop"].value<double>())
        cfg.fStop = static_cast<float>(*v);
    if (auto v = tbl["tonemap"]["exposure_shutter"].value<double>())
        cfg.shutterSpeed = static_cast<float>(*v);

    return cfg;
}

// ---------------------------------------------------------------------------
// HeadlessApp
// ---------------------------------------------------------------------------

HeadlessApp::HeadlessApp(const RenderConfig& config)
    : m_config(config)
{
    m_settings = std::make_unique<SettingsManager>();
    m_scene = std::make_unique<Scene>();
    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_sharedCtx = std::make_unique<SharedContext>();
    m_loader = std::make_unique<GltfLoader>();

    m_render->setScene(m_scene.get());
    m_render->setSettingsManager(m_settings.get());
    m_render->setSharedContext(m_sharedCtx.get());
}

void HeadlessApp::populateSettings()
{
    const fs::path sceneFilePath(m_config.scenePath);
    const std::string resourceSearchPath = sceneFilePath.parent_path().string();

    m_settings->setAs<uint32_t>("render/width", m_config.width);
    m_settings->setAs<uint32_t>("render/height", m_config.height);
    m_settings->setAs<uint32_t>("render/integrator", m_config.integrator);
    m_settings->setAs<uint32_t>("render/pt/depth", m_config.maxDepth);
    m_settings->setAs<uint32_t>("render/pt/sppTotal", m_config.spp);
    m_settings->setAs<uint32_t>("render/pt/spp", m_config.sppPerLaunch);
    m_settings->setAs<uint32_t>("render/pt/iteration", 0);
    m_settings->setAs<uint32_t>("render/pt/stratifiedSamplingType", 0);
    m_settings->setAs<uint32_t>("render/pt/tonemapperType", m_config.tonemapType);
    m_settings->setAs<uint32_t>("render/pt/debug", 0);
    m_settings->setAs<uint32_t>("render/pt/samplerType", m_config.samplerType);
    m_settings->setAs<uint32_t>("render/pt/rectLightSamplingMethod", 0);
    m_settings->setAs<uint32_t>("render/pt/misHeuristic", 0);
    m_settings->setAs<float>("render/cameraSpeed", 1.0f);
    m_settings->setAs<float>("render/pt/upscaleFactor", 0.5f);
    m_settings->setAs<bool>("render/pt/enableUpscale", false);
    m_settings->setAs<bool>("render/pt/enableAcc", true);
    m_settings->setAs<bool>("render/pt/enableTonemap", true);
    m_settings->setAs<bool>("render/pt/isResized", false);
    m_settings->setAs<bool>("render/enableValidation", false);
    m_settings->setAs<uint32_t>("render/selectedCamera", static_cast<uint32_t>(m_config.cameraIndex));
    m_settings->setAs<bool>("render/enableMotionBlur", false);
    m_settings->setAs<bool>("render/isMotionBlurVisible", false);
    m_settings->setAs<bool>("render/enableCameraMotionBlur", false);
    m_settings->setAs<float>("render/motionBlur/shutterTime", 1.0f / 24.0f);
    m_settings->setAs<uint32_t>("render/motionBlur/shutterMode", 1);
    m_settings->setAs<float>("render/animation/speed", 1.0f);
    m_settings->setAs<std::string>("resource/searchPath", resourceSearchPath);

    // Postprocessing
    m_settings->setAs<float>("render/post/tonemapper/filmIso", m_config.filmIso);
    m_settings->setAs<float>("render/post/tonemapper/cm2_factor", 1.0f);
    m_settings->setAs<float>("render/post/tonemapper/fStop", m_config.fStop);
    m_settings->setAs<float>("render/post/tonemapper/shutterSpeed", m_config.shutterSpeed);
    m_settings->setAs<float>("render/post/tonemapper/maxEDR", 1.0f); // no HDR headroom in headless mode
    m_settings->setAs<float>("render/post/gamma", m_config.gamma);

    // Dev settings
    m_settings->setAs<float>("render/pt/dev/shadowRayTmin", 0.0f);
    m_settings->setAs<float>("render/pt/dev/materialRayTmin", 0.0f);

    // Per-animation settings (renderer queries these each frame)
    char key[64];
    for (int i = 0; i < static_cast<int>(m_scene->getAnimations().size()); ++i)
    {
        snprintf(key, sizeof(key), "render/animation/anim%d/state", i);
        m_settings->setAs<bool>(key, false);

        snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
        m_settings->setAs<float>(key, m_scene->getAnimations()[i].start);
    }
}

void HeadlessApp::saveOutput(Buffer* buf)
{
    const uint32_t w = buf->width();
    const uint32_t h = buf->height();
    const float* data = static_cast<const float*>(buf->getHostPointer());

    auto dotPos = m_config.outputPath.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? m_config.outputPath.substr(dotPos) : "";

    if (ext == ".exr")
    {
        const char* err = nullptr;
        int ret = SaveEXR(data, w, h, 4, 0, m_config.outputPath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to save EXR: {}", err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
        }
    }
    else if (ext == ".png")
    {
        std::vector<uint8_t> pixels(w * h * 4);
        for (uint32_t i = 0; i < w * h; ++i)
        {
            for (int c = 0; c < 4; ++c)
            {
                float v = std::max(0.0f, std::min(1.0f, data[i * 4 + c]));
                pixels[i * 4 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
            }
        }
        int ret = stbi_write_png(m_config.outputPath.c_str(), w, h, 4, pixels.data(), w * 4);
        if (!ret)
        {
            STRELKA_ERROR("Failed to save PNG: {}", m_config.outputPath);
        }
    }
    else
    {
        STRELKA_ERROR("Unsupported output format '{}'. Use .exr or .png", ext);
    }
}

void HeadlessApp::printProgress(uint32_t currentSpp, uint32_t totalSpp, double lastRenderMs)
{
    const int barWidth = 30;
    float fraction = static_cast<float>(currentSpp) / static_cast<float>(totalSpp);
    int filled = static_cast<int>(fraction * barWidth);

    char bar[64];
    for (int i = 0; i < barWidth; ++i)
        bar[i] = (i < filled) ? '=' : ' ';
    bar[barWidth] = '\0';

    double etaSec = 0.0;
    if (currentSpp > 0)
        etaSec = lastRenderMs * (totalSpp - currentSpp) / 1000.0;

    fprintf(stdout, "\rRendering [%s] %u/%u spp | %.1f ms/sample | ETA: %.1fs   ",
            bar, currentSpp, totalSpp, lastRenderMs, etaSec);
    fflush(stdout);
}

int HeadlessApp::run()
{
    using namespace std::chrono;

    // 1. Load scene
    STRELKA_INFO("Loading scene: {}", m_config.scenePath);
    if (!m_loader->loadGltf(m_config.scenePath, *m_scene))
    {
        STRELKA_FATAL("Failed to load scene: {}", m_config.scenePath);
        return 1;
    }

    // 2. Add a free-fly "Main" camera as the last entry (same as Editor)
    {
        Camera camera;
        camera.name = "Main";
        camera.fov = 45.0f;

        // Compute position that fits the scene
        const auto& vertices = m_scene->getVertices();
        if (!vertices.empty())
        {
            glm::vec3 aabbMin(std::numeric_limits<float>::max());
            glm::vec3 aabbMax(std::numeric_limits<float>::lowest());
            for (const auto& v : vertices)
            {
                aabbMin = glm::min(aabbMin, v.pos);
                aabbMax = glm::max(aabbMax, v.pos);
            }
            glm::vec3 center = (aabbMin + aabbMax) * 0.5f;
            float radius = glm::length(aabbMax - center);
            if (radius < 1e-6f)
                radius = 1.0f;
            float halfFovRad = glm::radians(camera.fov * 0.5f);
            float distance = radius / std::tan(halfFovRad);
            camera.position = center + glm::vec3(0.0f, 0.0f, distance);
        }
        camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
        camera.updateViewMatrix();
        m_scene->addCamera(camera);
    }

    // 3. Apply camera overrides if provided
    if (m_config.cameraIndex >= 0 &&
        m_config.cameraIndex < static_cast<int>(m_scene->getCameraCount()))
    {
        Camera& cam = m_scene->getCamera(static_cast<uint32_t>(m_config.cameraIndex));

        if (m_config.cameraFov)
            cam.fov = *m_config.cameraFov;

        if (m_config.cameraPosition)
            cam.position = *m_config.cameraPosition;

        if (m_config.cameraTarget)
        {
            glm::vec3 dir = glm::normalize(*m_config.cameraTarget - cam.position);
            // Compute orientation quaternion from direction
            glm::vec3 forward(0.0f, 0.0f, -1.0f);
            glm::vec3 axis = glm::cross(forward, dir);
            float dot = glm::dot(forward, dir);
            if (glm::length(axis) > 1e-6f)
            {
                float angle = std::acos(std::clamp(dot, -1.0f, 1.0f));
                cam.mOrientation = glm::angleAxis(angle, glm::normalize(axis));
            }
        }

        cam.updateViewMatrix();
    }

    // 4. Populate settings
    populateSettings();

    // 5. Init renderer
    STRELKA_INFO("Initializing renderer ({}x{}, {} spp, integrator={})...",
                 m_config.width, m_config.height, m_config.spp, m_config.integrator);
    m_render->init();

    // 6. Create output buffer
    BufferDesc desc{};
    desc.format = BufferFormat::FLOAT4;
    desc.width = m_config.width;
    desc.height = m_config.height;
    Buffer* outputBuf = m_render->createBuffer(desc);

    // 7. Render loop
    auto startTime = high_resolution_clock::now();

    while (m_sharedCtx->mSubframeIndex < m_config.spp)
    {
        m_render->renderSync(outputBuf);

        printProgress(static_cast<uint32_t>(m_sharedCtx->mSubframeIndex),
                      m_config.spp,
                      m_render->getLastRenderTimeMs());
    }

    auto totalTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime);

    // 8. Save output
    saveOutput(outputBuf);

    // 9. Print summary
    fprintf(stdout, "\nDone: %u spp in %.1f s -> %s\n",
            m_config.spp, totalTime.count() / 1000.0, m_config.outputPath.c_str());

    delete outputBuf;
    return 0;
}

} // namespace oka
