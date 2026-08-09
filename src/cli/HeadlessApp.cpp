#include "HeadlessApp.h"

#include <tonemappers.h>

#include <log.h>

#include <strelka/sceneloader/gltfloader.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <stb_image_write.h>
#include <tinyexr.h>
#include <toml++/toml.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace oka
{

static uint32_t parseIntegratorName(const std::string& name)
{
    if (name == "pt")
    {
        return 0;
    }
    if (name == "bdpt")
    {
        return 1;
    }
    if (name == "vcm")
    {
        return 2;
    }
    throw std::invalid_argument("Unknown integrator: " + name);
}

uint32_t parseSamplerName(const std::string& name)
{
    if (name == "halton")
    {
        return 0;
    }
    if (name == "pcg")
    {
        return 1;
    }
    if (name == "sobol")
    {
        return 2;
    }
    // Sobol scrambled with a blue-noise tile (editor: "Sobol + blue noise")
    if (name == "sobol_bn" || name == "sobol+bn")
    {
        return 3;
    }
    // Blue noise for the first N spp, then Sobol (editor: "Hybrid")
    if (name == "hybrid" || name == "bluenoise")
    {
        return 4;
    }
    throw std::invalid_argument(
        "Unknown sampler: " + name + " (halton|pcg|sobol|sobol_bn|hybrid)");
}

uint32_t parseTonemapName(const std::string& name)
{
    if (name == "none")
    {
        return 0;
    }
    if (name == "reinhard")
    {
        return 1;
    }
    if (name == "aces")
    {
        return 2;
    }
    if (name == "filmic")
    {
        return 3;
    }
    throw std::invalid_argument("Unknown tonemap: " + name);
}

static uint32_t parseEnumOrDefault(const std::string& name, uint32_t (*parse)(const std::string&), uint32_t fallback,
                                   const char* label)
{
    try
    {
        return parse(name);
    }
    catch (const std::invalid_argument&)
    {
        STRELKA_WARNING("Unknown {} '{}', defaulting to {}", label, name, fallback);
        return fallback;
    }
}

RenderConfig parseTomlConfig(const std::string& tomlPath)
{
    RenderConfig cfg;

    const toml::table tbl = toml::parse_file(tomlPath);
    const fs::path tomlDir = fs::path(tomlPath).parent_path();

    if (auto v = tbl["scene"]["path"].value<std::string>())
    {
        fs::path scenePath(*v);
        if (scenePath.is_relative())
        {
            // Harness tomls often use ../../scenes/... paths meant for a build CWD.
            // Prefer the glb that sits next to the toml (bdpt_tests pattern).
            const fs::path sibling = tomlDir / scenePath.filename();
            if (fs::exists(sibling))
            {
                scenePath = sibling;
            }
            else if (fs::exists(tomlDir / scenePath))
            {
                scenePath = tomlDir / scenePath;
            }
        }
        cfg.scenePath = fs::weakly_canonical(scenePath).string();
    }

    // Output stays CWD-relative so batch runs don't write into scenes/.
    if (auto v = tbl["output"]["path"].value<std::string>())
    {
        cfg.outputPath = *v;
    }
    if (auto v = tbl["output"]["width"].value<int64_t>())
    {
        cfg.width = static_cast<uint32_t>(*v);
    }
    if (auto v = tbl["output"]["height"].value<int64_t>())
    {
        cfg.height = static_cast<uint32_t>(*v);
    }

    if (auto v = tbl["render"]["integrator"].value<std::string>())
    {
        cfg.integrator = parseEnumOrDefault(*v, parseIntegratorName, 0, "integrator");
    }
    if (auto v = tbl["render"]["spp"].value<int64_t>())
    {
        cfg.spp = static_cast<uint32_t>(*v);
    }
    if (auto v = tbl["render"]["spp_per_launch"].value<int64_t>())
    {
        cfg.sppPerLaunch = static_cast<uint32_t>(*v);
    }
    if (auto v = tbl["render"]["max_depth"].value<int64_t>())
    {
        cfg.maxDepth = static_cast<uint32_t>(*v);
    }
    if (auto v = tbl["render"]["clamp_indirect"].value<double>())
    {
        cfg.clampIndirect = static_cast<float>(*v);
    }
    // 1 = shading normals, 3..6 = denoiser guides; see DebugMode in ShaderTypes.h.
    if (auto v = tbl["render"]["debug"].value<int64_t>())
        cfg.debugMode = (uint32_t)*v;
    // Plain divisor: 2 halves every texture, which is the quickest way to find
    // out whether a scene fits at all.
    if (auto v = tbl["render"]["texture_downscale"].value<int64_t>())
        cfg.textureDownscale = (uint32_t)*v;
    if (auto v = tbl["render"]["texture_max_dim"].value<int64_t>())
        cfg.textureMaxDim = (uint32_t)*v;
    if (auto v = tbl["render"]["volume_model"].value<std::string>())
        cfg.volumeModel = (*v == "cycles") ? 1u : 0u;
    if (auto v = tbl["render"]["denoise"].value<bool>())
        cfg.denoise = *v;
    if (auto v = tbl["render"]["profile_stages"].value<bool>())
        cfg.profileStages = *v;
    if (auto v = tbl["render"]["upscale"].value<bool>())
        cfg.upscale = *v;
    if (auto v = tbl["render"]["upscale_factor"].value<double>())
        cfg.upscaleFactor = (float)*v;
    if (auto v = tbl["render"]["upscale_mode"].value<std::string>())
        cfg.upscaleMode = (*v == "temporal") ? 1u : 0u;
    if (auto v = tbl["render"]["metal4"].value<bool>())
        cfg.metal4 = *v ? 1u : 0u;
    if (auto v = tbl["render"]["sort_rays"].value<bool>())
        cfg.sortRays = *v;
    if (auto v = tbl["render"]["texture_lod"].value<bool>())
        cfg.textureLod = *v;
    if (auto v = tbl["render"]["ris_candidates"].value<int64_t>())
        cfg.risCandidates = (uint32_t)*v;
    if (auto v = tbl["render"]["estimator_mode"].value<int64_t>())
        cfg.estimatorMode = (uint32_t)*v;
    if (auto v = tbl["render"]["sharc"].value<bool>())
        cfg.sharc = *v;
    if (auto v = tbl["render"]["sharc_depth"].value<int64_t>())
        cfg.sharcDepth = (uint32_t)*v;
    if (auto v = tbl["render"]["sharc_min_samples"].value<int64_t>())
        cfg.sharcMinSamples = (uint32_t)*v;
    if (auto v = tbl["render"]["sharc_base_size"].value<double>())
        cfg.sharcBaseSize = (float)*v;
    if (auto v = tbl["render"]["sampler"].value<std::string>())
    {
        cfg.samplerType = parseEnumOrDefault(*v, parseSamplerName, 0, "sampler");
    }
    if (auto v = tbl["render"]["blue_noise_switch_spp"].value<int64_t>())
    {
        cfg.blueNoiseSwitchSpp = static_cast<uint32_t>(*v);
    }

    if (auto v = tbl["camera"]["index"].value<int64_t>())
    {
        cfg.cameraIndex = static_cast<int>(*v);
    }
    if (auto* arr = tbl["camera"]["position"].as_array())
    {
        if (arr->size() == 3)
        {
            cfg.cameraPosition = glm::vec3(arr->get(0)->value_or(0.0), arr->get(1)->value_or(0.0),
                                           arr->get(2)->value_or(0.0));
        }
    }
    if (auto* arr = tbl["camera"]["target"].as_array())
    {
        if (arr->size() == 3)
        {
            cfg.cameraTarget = glm::vec3(arr->get(0)->value_or(0.0), arr->get(1)->value_or(0.0),
                                         arr->get(2)->value_or(0.0));
        }
    }
    if (auto v = tbl["camera"]["fov"].value<double>())
    {
        cfg.cameraFov = static_cast<float>(*v);
    }

    if (auto v = tbl["tonemap"]["type"].value<std::string>())
    {
        cfg.tonemapType = parseEnumOrDefault(*v, parseTonemapName, 2, "tonemap");
    }
    if (auto v = tbl["tonemap"]["gamma"].value<double>())
    {
        cfg.gamma = static_cast<float>(*v);
    }
    if (auto v = tbl["tonemap"]["exposure_iso"].value<double>())
    {
        cfg.filmIso = static_cast<float>(*v);
        cfg.exposureOverridden = true;
    }
    if (auto v = tbl["tonemap"]["exposure_fstop"].value<double>())
    {
        cfg.fStop = static_cast<float>(*v);
        cfg.exposureOverridden = true;
    }
    if (auto v = tbl["tonemap"]["exposure_shutter"].value<double>())
    {
        cfg.shutterSpeed = static_cast<float>(*v);
        cfg.exposureOverridden = true;
    }

    return cfg;
}

HeadlessApp::HeadlessApp(const RenderConfig& config)
    : m_config(config)
{
    m_settings = std::make_unique<SettingsManager>();
    m_scene = std::make_unique<Scene>();
    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_sharedCtx = std::make_unique<SharedContext>();

    m_render->setScene(m_scene.get());
    m_render->setSettingsManager(m_settings.get());
    m_render->setSharedContext(m_sharedCtx.get());
}

void HeadlessApp::populateSettings()
{
    const std::string resourceSearchPath = fs::path(m_config.scenePath).parent_path().string();

    if (m_config.integrator != 0)
    {
        STRELKA_WARNING("Integrator {} (bdpt/vcm) is not available on this Metal wavefront build; using path tracing",
                        m_config.integrator);
        m_config.integrator = 0;
    }

    // Only keys MetalRender reads (plus per-animation state/time).
    m_settings->setAs<uint32_t>("render/width", m_config.width);
    m_settings->setAs<uint32_t>("render/height", m_config.height);
    m_settings->setAs<uint32_t>("render/pt/depth", m_config.maxDepth);
    m_settings->setAs<uint32_t>("render/pt/sppTotal", m_config.spp);
    m_settings->setAs<uint32_t>("render/pt/spp", m_config.sppPerLaunch);
    m_settings->setAs<uint32_t>("render/pt/tonemapperType", m_config.tonemapType);
    m_settings->setAs<uint32_t>("render/pt/debug", m_config.debugMode);
    m_settings->setAs<uint32_t>("render/pt/samplerType", m_config.samplerType);
    m_settings->setAs<uint32_t>("render/pt/blueNoiseSwitchSpp", m_config.blueNoiseSwitchSpp);
    m_settings->setAs<uint32_t>("render/pt/rectLightSamplingMethod", 0);
    m_settings->setAs<float>("render/pt/upscaleFactor", m_config.upscaleFactor);
    m_settings->setAs<bool>("render/pt/enableUpscale", m_config.upscale);
    m_settings->setAs<uint32_t>("render/pt/upscaleMode", m_config.upscaleMode);
    m_settings->setAs<bool>("render/pt/enableAcc", true);
    m_settings->setAs<uint32_t>("render/selectedCamera", static_cast<uint32_t>(std::max(0, m_config.cameraIndex)));
    m_settings->setAs<bool>("render/enableMotionBlur", false);
    m_settings->setAs<bool>("render/isMotionBlurVisible", false);
    m_settings->setAs<bool>("render/enableCameraMotionBlur", false);
    m_settings->setAs<float>("render/motionBlur/shutterTime", 1.0f / 24.0f);
    m_settings->setAs<uint32_t>("render/motionBlur/shutterMode", 1);
    m_settings->setAs<std::string>("resource/searchPath", resourceSearchPath);

    // Wavefront + single submission: deterministic sync, no banding overhead.
    m_settings->setAs<uint32_t>("render/pt/profileStages", 0);
    m_settings->setAs<uint32_t>("render/pt/risCandidates", 1u);
    m_settings->setAs<uint32_t>("render/pt/writeAov", 0);
    m_settings->setAs<bool>("render/pt/denoise", m_config.denoise);
    m_settings->setAs<uint32_t>("render/pt/profileStages", m_config.profileStages ? 1u : 0u);
    m_settings->setAs<uint32_t>("render/pt/risCandidates", m_config.risCandidates);
    m_settings->setAs<uint32_t>("render/pt/jitterSign", 0);
    m_settings->setAs<uint32_t>("render/pt/denoiseDepthMode", 0);
    m_settings->setAs<float>("render/pt/denoiseFireflyClamp", 8.0f);
    m_settings->setAs<float>("render/pt/clampIndirect", m_config.clampIndirect);
    m_settings->setAs<bool>("render/pt/denoisePlaybackMotionBlur", false);
    m_settings->setAs<uint32_t>("render/pt/metal4", m_config.metal4);
    m_settings->setAs<uint32_t>("render/pt/sortRays", m_config.sortRays ? 1u : 0u);
    m_settings->setAs<uint32_t>("render/pt/textureLod", m_config.textureLod ? 1u : 0u);
    m_settings->setAs<uint32_t>("render/pt/staticTraversal", 1);
    m_settings->setAs<uint32_t>("render/validate/estimatorMode", m_config.estimatorMode);
    // Absorption convention for transmissive media: 0 = glTF, 1 = Cycles.
    m_settings->setAs<uint32_t>("render/material/volumeModel", m_config.volumeModel);
    // 0 = load textures at full resolution.
    m_settings->setAs<uint32_t>("render/texture/maxDimension", m_config.textureMaxDim);
    // Headless: nothing picks, nothing saves the scene back out.
    m_settings->setAs<bool>("scene/releaseHostGeometry", true);
    m_settings->setAs<uint32_t>("render/texture/downscale", m_config.textureDownscale);
    // An HDRI carries radiance; normalising it away makes physical parity
    // impossible. See loadEnvMap().
    m_settings->setAs<bool>("render/env/autoCalibrate", false);
    // Spatially hashed radiance cache. Off by default: it trades a little
    // bias for a large cut in path length, which is a choice a scene makes.
    m_settings->setAs<bool>("render/pt/sharc", m_config.sharc);
    m_settings->setAs<uint32_t>("render/pt/sharcCapacity", 1u << 22);
    m_settings->setAs<uint32_t>("render/pt/sharcMinSamples", m_config.sharcMinSamples);
    m_settings->setAs<uint32_t>("render/pt/sharcDepth", m_config.sharcDepth);
    m_settings->setAs<float>("render/pt/sharcVoxelPixels", m_config.sharcBaseSize);
    // Block compression and a disk cache for the finished textures.
    // The cache holds them downscaled, mipped and compressed, so a second
    // launch skips the decode, the resample, the mip chain and the encode.
    m_settings->setAs<bool>("render/texture/compress", true);
    m_settings->setAs<std::string>("render/texture/cachePath",
                           (std::filesystem::temp_directory_path() / "strelka_texcache").string());
    m_settings->setAs<bool>("render/validate/analyticLights", true);

    // A scene may state its own exposure in the light sidecar, and when it does
    // it wins over the defaults here -- an exposure is a property of the shot,
    // not of the renderer. It does not win over an exposure the caller asked for:
    // tools/feature_tests pins all three to hold the factor at exactly 1.0, and a
    // scene quietly overriding that would make every row measure its exposure.
    float filmIso = m_config.filmIso;
    float fStop = m_config.fStop;
    float shutterSpeed = m_config.shutterSpeed;
    float cm2Factor = 1.0f;
    if (!m_config.exposureOverridden)
    {
        if (const auto& exp = m_scene->getExposure(); exp.has_value())
        {
            filmIso = exp->filmIso;
            fStop = exp->fStop;
            shutterSpeed = exp->shutterSpeed;
            cm2Factor = exp->cm2Factor;
            STRELKA_INFO("Exposure from scene: iso={} fstop={} shutter={}", filmIso, fStop,
                         shutterSpeed);
        }
    }
    m_settings->setAs<float>("render/post/tonemapper/filmIso", filmIso);
    m_settings->setAs<float>("render/post/tonemapper/cm2_factor", cm2Factor);
    m_settings->setAs<float>("render/post/tonemapper/fStop", fStop);
    m_settings->setAs<float>("render/post/tonemapper/shutterSpeed", shutterSpeed);
    m_settings->setAs<float>("render/post/tonemapper/maxEDR", 1.0f);
    m_settings->setAs<float>("render/post/gamma", m_config.gamma);

    for (size_t i = 0; i < m_scene->getAnimations().size(); ++i)
    {
        char key[64];
        snprintf(key, sizeof(key), "render/animation/anim%zu/state", i);
        m_settings->setAs<bool>(key, false);
        snprintf(key, sizeof(key), "render/animation/anim%zu/time", i);
        m_settings->setAs<float>(key, m_scene->getAnimations()[i].start);
    }
}

void HeadlessApp::saveOutput(Buffer* buf)
{
    const uint32_t w = buf->width();
    const uint32_t h = buf->height();
    const float* data = static_cast<const float*>(buf->getHostPointer());

    const fs::path outPath(m_config.outputPath);
    if (outPath.has_parent_path())
    {
        fs::create_directories(outPath.parent_path());
    }

    const std::string ext = outPath.extension().string();
    if (ext == ".exr")
    {
        const char* err = nullptr;
        const int ret = SaveEXR(data, static_cast<int>(w), static_cast<int>(h), 4, 0, m_config.outputPath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to save EXR: {}", err ? err : "unknown");
            if (err)
            {
                FreeEXRErrorMessage(err);
            }
        }
    }
    else if (ext == ".png")
    {
        // A PNG is a display image and has to go through the tone curve; the EXR
        // above is data and must not. The renderer's own tonemap pass writes to a
        // texture rather than back into this buffer -- deliberately, so the
        // buffer keeps linear radiance -- which left this writer emitting the
        // same file whatever `--tonemap` said. Byte-identical, measured on the
        // Cornell box with `none` against `aces`.
        //
        // The curve comes from the same header the shader uses, so the two
        // cannot drift.
        const float exposure = m_settings->getAs<float>("render/post/tonemapper/filmIso") > 0.0f ?
                                   m_settings->getAs<float>("render/post/tonemapper/cm2_factor") *
                                       m_settings->getAs<float>("render/post/tonemapper/filmIso") /
                                       (m_settings->getAs<float>("render/post/tonemapper/shutterSpeed") *
                                        m_settings->getAs<float>("render/post/tonemapper/fStop") *
                                        m_settings->getAs<float>("render/post/tonemapper/fStop")) /
                                       100.0f :
                                   m_settings->getAs<float>("render/post/tonemapper/cm2_factor");
        const uint32_t curve = m_settings->getAs<uint32_t>("render/pt/tonemapperType");
        const float gamma = m_settings->getAs<float>("render/post/gamma");

        std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 4);
        for (uint32_t i = 0; i < w * h; ++i)
        {
            oka::tonemap::float3 c = simd_make_float3(data[i * 4 + 0], data[i * 4 + 1], data[i * 4 + 2]) * exposure;
            switch (static_cast<oka::tonemap::ToneMapperType>(curve))
            {
            case oka::tonemap::ToneMapperType::eReinhard:
                c = oka::tonemap::reinhard(c);
                break;
            case oka::tonemap::ToneMapperType::eACES:
                c = oka::tonemap::ACESFitted(c);
                break;
            case oka::tonemap::ToneMapperType::eFilmic:
                c = oka::tonemap::ACESFilm(c);
                break;
            case oka::tonemap::ToneMapperType::eNone:
                break;
            }
            if (gamma > 0.0f)
            {
                c = oka::tonemap::srgbGamma(c, gamma);
            }
            const float rgba[4] = { c.x, c.y, c.z, data[i * 4 + 3] };
            for (int k = 0; k < 4; ++k)
            {
                const float v = std::clamp(rgba[k], 0.0f, 1.0f);
                pixels[i * 4 + k] = static_cast<uint8_t>(v * 255.0f + 0.5f);
            }
        }
        if (!stbi_write_png(m_config.outputPath.c_str(), static_cast<int>(w), static_cast<int>(h), 4, pixels.data(),
                            static_cast<int>(w * 4)))
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
    constexpr int barWidth = 30;
    const float fraction = static_cast<float>(currentSpp) / static_cast<float>(std::max(1u, totalSpp));
    const int filled = static_cast<int>(fraction * barWidth);

    char bar[barWidth + 1];
    for (int i = 0; i < barWidth; ++i)
    {
        bar[i] = (i < filled) ? '=' : ' ';
    }
    bar[barWidth] = '\0';

    const double etaSec = (currentSpp > 0) ? lastRenderMs * (totalSpp - currentSpp) / 1000.0 : 0.0;
    fprintf(stdout, "\rRendering [%s] %u/%u spp | %.1f ms/sample | ETA: %.1fs   ", bar, currentSpp, totalSpp,
            lastRenderMs, etaSec);
    fflush(stdout);
}

int HeadlessApp::run()
{
    using namespace std::chrono;

    STRELKA_INFO("Loading scene: {}", m_config.scenePath);
    GltfLoader loader;
    if (!loader.loadGltf(m_config.scenePath, *m_scene))
    {
        STRELKA_FATAL("Failed to load scene: {}", m_config.scenePath);
        return 1;
    }

    // Fit a free-fly camera only when the glTF has none.
    if (m_scene->getCameraCount() == 0)
    {
        Camera camera;
        camera.name = "Main";
        camera.fov = 45.0f;

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
            const glm::vec3 center = (aabbMin + aabbMax) * 0.5f;
            float radius = glm::length(aabbMax - center);
            if (radius < 1e-6f)
            {
                radius = 1.0f;
            }
            const float halfFovRad = glm::radians(camera.fov * 0.5f);
            const float distance = radius / std::tan(halfFovRad);
            camera.position = center + glm::vec3(0.0f, 0.0f, distance);
        }
        camera.mOrientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        camera.updateViewMatrix();
        m_scene->addCamera(camera);
    }

    if (m_config.cameraIndex >= 0 && m_config.cameraIndex < static_cast<int>(m_scene->getCameraCount()))
    {
        Camera& cam = m_scene->getCamera(static_cast<uint32_t>(m_config.cameraIndex));

        if (m_config.cameraFov)
        {
            cam.fov = *m_config.cameraFov;
        }
        if (m_config.cameraPosition)
        {
            cam.position = *m_config.cameraPosition;
        }
        if (m_config.cameraTarget)
        {
            // firstperson view = R * T(-p); upper-left of lookAt is that R.
            const glm::mat4 view = glm::lookAt(cam.position, *m_config.cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));
            cam.mOrientation = glm::normalize(glm::quat_cast(glm::mat3(view)));
        }
        cam.updateViewMatrix();
    }

    populateSettings();

    STRELKA_INFO("Initializing renderer ({}x{}, {} spp)...", m_config.width, m_config.height, m_config.spp);
    m_render->init();

    BufferDesc desc{};
    desc.format = BufferFormat::FLOAT4;
    desc.width = m_config.width;
    desc.height = m_config.height;
    std::unique_ptr<Buffer> outputBuf(m_render->createBuffer(desc));

    const auto startTime = high_resolution_clock::now();
    bool announced = false;
    // Headroom sweep. Every variant is measured inside one process and one
    // thermal state, cycling round-robin, because the alternative -- a run per
    // variant -- puts a scene load and a fresh clock ramp between the numbers
    // being compared. Measured sequentially, this probe reported that adding
    // dependent loads made the frame *faster*, which is drift and not headroom.
    while (m_sharedCtx->mSubframeIndex < m_config.spp)
    {
        m_render->renderSync(outputBuf.get());

        if (!announced && !m_config.capturePath.empty())
        {
            // One frame, in steady state: the point of the capture is the
            // shading kernels, and starting it at launch would record the
            // acceleration structure build instead.
            //
            // It writes every resource the frame reads, so the document is the
            // size of the scene on the device: the pine forest produces 7.7 GB.
            // Register allocation depends on the shader and its function
            // constants, not on how much geometry it traverses, so a small
            // scene with the same constants answers the same question for a few
            // megabytes.
            m_render->beginGpuCapture(m_config.capturePath);
            m_render->renderSync(outputBuf.get());
            m_render->endGpuCapture();
            fprintf(stdout, "\ncaptured one frame -> %s\n", m_config.capturePath.c_str());
            fflush(stdout);
        }
        if (!announced)
        {
            // A marker for anything sampling the GPU from outside.
            //
            // Hardware counters are a time average, and on a heavy scene the
            // loading, the texture cache and the acceleration structure build
            // are most of a short run -- sample across them and the numbers
            // describe a BVH build rather than a render. This is printed after
            // the *first* sample has completed, so everything expensive and
            // one-off is already behind it, and flushed because stdout is block
            // buffered when redirected: without the flush a profiler waiting on
            // this line waits forever.
            announced = true;
            fprintf(stdout, "\nSTRELKA_RENDER_BEGIN\n");
            fflush(stdout);
        }
        printProgress(static_cast<uint32_t>(m_sharedCtx->mSubframeIndex), m_config.spp, m_render->getLastRenderTimeMs());
    }
    const auto totalTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime);

    if (m_render->deviceError())
    {
        // Writing the file anyway would hand back a black image that looks like a
        // lighting problem; saying so and failing is the honest outcome.
        fprintf(stderr,
                "\nGPU command buffer failed -- the render is not valid. The scene most likely "
                "does not fit on the device.\nTry render.texture_downscale = 2 or "
                "render.texture_max_dim = 2048 in the config.\n");
        return 2;
    }

    saveOutput(outputBuf.get());

    fprintf(stdout, "\nDone: %u spp in %.1f s -> %s\n", m_config.spp, totalTime.count() / 1000.0,
            m_config.outputPath.c_str());
    return 0;
}

} // namespace oka
