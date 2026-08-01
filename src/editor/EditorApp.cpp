#include "EditorApp.h"

#include <strelka/sceneloader/sceneserializer.h>
#include <log.h>
#include <paths.h>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>
#include <vector>
#include <unistd.h>

#include <tinyexr.h>
#include <stb_image_write.h>

namespace oka
{

EditorApp::EditorApp(const std::string& sceneFile, const std::string& resourceSearchPath)
    : m_sceneFile(sceneFile), m_resourceSearchPath(resourceSearchPath)
{
    m_settingsManager = std::make_unique<SettingsManager>();

    m_scene = std::make_unique<Scene>();
    m_display = std::unique_ptr<Display>(DisplayFactory::createDisplay());
    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_sharedCtx = std::make_unique<SharedContext>();

    m_sceneLoader = std::make_unique<GltfLoader>();

    m_render->setScene(m_scene.get());
    m_render->setSettingsManager(m_settingsManager.get());
    m_render->setSharedContext(m_sharedCtx.get());

    prepare();

    m_render->init();
#ifdef __APPLE__
    m_display->setNativeDevice(m_render->getNativeDevicePtr());
    // Display creates its own command queue for independent frame pacing
#endif
    m_display->init(1024, 768, m_settingsManager.get());
    m_display->setResizeHandler(this);
}

void EditorApp::framebufferResize(int newWidth, int newHeight)
{
    m_settingsManager->setAs<uint32_t>("render/width", static_cast<uint32_t>(newWidth));
    m_settingsManager->setAs<uint32_t>("render/height", static_cast<uint32_t>(newHeight));
    m_resized = true;
}

glm::vec3 EditorApp::computeSceneFitPosition(float fovDegrees) const
{
    const auto& vertices = m_scene->getVertices();
    if (vertices.empty())
        return glm::vec3(0, 0, -10);

    // Compute AABB
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

    // Distance so the bounding sphere fits in the vertical FOV
    float halfFovRad = glm::radians(fovDegrees * 0.5f);
    float distance = radius / std::tan(halfFovRad);

    // Position camera along -Z looking at center (worldForward = (0,0,-1))
    return center + glm::vec3(0.0f, 0.0f, distance);
}

void EditorApp::prepare()
{
    m_sceneLoader->loadGltf(m_sceneFile, *m_scene);

    // Add a free-fly "Main" camera as the last entry
    oka::Camera camera;
    camera.name = "Main";
    camera.fov = 45.0f;
    camera.position = computeSceneFitPosition(camera.fov);
    camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
    camera.updateViewMatrix();
    m_scene->addCamera(camera);

    // Select first GLTF camera (index 0) by default
    m_selectedCamera = 0;
    setCameraDetached(false);

    m_cameraController = std::make_unique<CameraController>(m_scene->getCamera(m_selectedCamera), true);
    m_display->setInputHandler(m_cameraController.get());
    loadSettings();
}

void EditorApp::loadSettings()
{
    STRELKA_DEBUG("Resource search path {}", m_resourceSearchPath);

    const uint32_t imageWidth = 1024;
    const uint32_t imageHeight = 768;

    m_settingsManager->setAs<uint32_t>("render/width", imageWidth);
    m_settingsManager->setAs<uint32_t>("render/height", imageHeight);
    m_settingsManager->setAs<uint32_t>("render/pt/depth", 8);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 256);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/iteration", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/stratifiedSamplingType", 0); // 0 - none, 1 - random, 2 -
                                                                               // stratified sampling, 3 -
                                                                               // optimized stratified sampling
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 1); // 0 - None, 1 - Reinhard, 2 - ACES, 3 - Filmic
    m_settingsManager->setAs<uint32_t>("render/pt/debug", 0); // 0 - none, 1 - normals
    m_settingsManager->setAs<float>("render/cameraSpeed", 1.0f);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", 0.5f);
    // MetalFX is opt-in; keep native-resolution output as the default.
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<bool>("render/pt/enableTonemap", true);
    m_settingsManager->setAs<bool>("render/pt/isResized", false);
    m_settingsManager->setAs<uint32_t>("render/pt/rectLightSamplingMethod", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/misHeuristic", 0); // 0 = balance, 1 = power
    // Sobol with a blue-noise screen-space error distribution, handing over to
    // plain per-pixel Owen scrambling once the frame has enough samples that the
    // spectrum of the error matters less than how fast it shrinks.
    //
    // Not Halton, which was the default and is the reason 1024 samples did not
    // look like four times 256: its dimensions are told apart only by an offset
    // into a table of 32 bases, so any two that are 32 apart -- and a depth-8
    // path uses 117 -- are the same sequence read from two places. Measured on
    // vespa, its error at 1024 spp was no lower than at 512.
    m_settingsManager->setAs<uint32_t>("render/pt/samplerType", 4);
    // Measured crossover on vespa: blue noise wins on post-filter error up to
    // about 16 samples (-16% at 1 spp, -2% at 16) and loses past it (+6% at 32,
    // +7% at 64), because a toroidal shift is a weaker randomisation than a
    // scramble once there are enough samples for that to matter.
    m_settingsManager->setAs<uint32_t>("render/pt/blueNoiseSwitchSpp", 16);
    m_settingsManager->setAs<bool>("render/enableValidation", false);
    m_settingsManager->setAs<uint32_t>("render/selectedCamera", 0);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
    m_settingsManager->setAs<bool>("render/enableCameraMotionBlur", false);
    m_settingsManager->setAs<float>("render/motionBlur/shutterTime", 1.0f / 24.0f);
    m_settingsManager->setAs<uint32_t>("render/motionBlur/shutterMode", 1); // 0=centered, 1=leading, 2=trailing
    m_settingsManager->setAs<float>("render/animation/speed", 1.0f);
    // Wavefront by default: bit-identical output, 2.6x faster at depth 8. The
    // megakernel stays selectable so any change can still be A/B'd against it.
    m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", 1); // 0 = megakernel, 1 = wavefront
    m_settingsManager->setAs<uint32_t>("render/pt/splitSubmissions", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/writeAov", 0);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", 0);
    // 0 = device depth (clip z/w), 1 = view-space axial, 2 = distance to the eye.
    // MetalFX does not say which it wants; the default is whichever the audit
    // measures as closest to the reference. See kDenoiseDepth* in ShaderTypes.h.
    m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", 0);
    // Luminance ceiling for the denoiser's colour input, in exposed units: a
    // single unbounded sample gets smeared over many frames by a temporal filter.
    // 0 disables it.
    m_settingsManager->setAs<float>("render/pt/denoiseFireflyClamp", 8.0f);
    // Playback stays stable at shutter close unless path-traced blur is enabled;
    // render/pt/spp controls its sample count.
    m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", false);
    if (const char* dn = getenv("STRELKA_DENOISE"))
    {
        m_settingsManager->setAs<bool>("render/pt/denoise", atoi(dn) != 0);
    }
    m_settingsManager->setAs<uint32_t>("render/pt/metal4", 0);
    if (const char* m4 = getenv("STRELKA_METAL4"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/metal4", (uint32_t)atoi(m4));
    }
    if (const char* up = getenv("STRELKA_UPSCALE"))
    {
        const float f = (float)atof(up);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", f > 0.0f && f < 1.0f);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", f);
    }
    if (const char* aovEnv = getenv("STRELKA_AOV"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/writeAov", (uint32_t)atoi(aovEnv));
    }
    if (const char* dv = getenv("STRELKA_DEBUG_VIEW"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/debug", (uint32_t)atoi(dv));
    }
    // Static-geometry traversal. Forcing it off makes the wavefront traverse the
    // same structure the megakernel does, which is what the bit-identity check
    // needs: the two intersector types round intersection distances differently.
    m_settingsManager->setAs<uint32_t>("render/pt/staticTraversal", 1);
    if (const char* st = getenv("STRELKA_STATIC"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/staticTraversal", (uint32_t)atoi(st));
    }

    if (const char* tracer = getenv("STRELKA_TRACER"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", (uint32_t)atoi(tracer));
    }
    m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", 0);
    m_settingsManager->setAs<bool>("render/validate/analyticLights", true);
    m_settingsManager->setAs<std::string>("resource/searchPath", m_resourceSearchPath);
    // Postprocessing settings:
    m_settingsManager->setAs<float>("render/post/tonemapper/maxEDR", 1.0f); // refreshed per frame from the display
    m_settingsManager->setAs<float>("render/post/tonemapper/filmIso", 100.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/cm2_factor", 1.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/fStop", 4.0f);
    m_settingsManager->setAs<float>("render/post/tonemapper/shutterSpeed", 100.0f);

    m_settingsManager->setAs<float>("render/post/gamma", 2.4f); // 0.0f - off
    // Dev settings:
    m_settingsManager->setAs<float>("render/pt/dev/shadowRayTmin", 0.0f); // offset to avoid self-collision in
                                                                          // light sampling
    m_settingsManager->setAs<float>("render/pt/dev/materialRayTmin", 0.0f); // offset to avoid self-collision in

    loadAnimSettings();
}

void EditorApp::loadAnimSettings()
{
    // Erase all previous per-animation settings to avoid leaking keys from old scenes
    m_settingsManager->eraseByPrefix("render/animation/anim");

    char key[64];
    for (int i = 0; i < (int)m_scene->getAnimations().size(); ++i)
    {
        snprintf(key, sizeof(key), "render/animation/anim%d/state", i);
        m_settingsManager->setAs<bool>(key, false);

        snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
        m_settingsManager->setAs<float>(key, m_scene->getAnimations()[i].start);
    }
}

void EditorApp::checkLoadingComplete()
{
    if (!m_isLoading)
        return;
    if (m_loadingFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;
    if (m_render->isRenderBusy())
        return;

    auto new_scene = m_loadingFuture.get();
    m_isLoading = false;

    if (!new_scene)
        return;

    // Tear the old renderer down *before* the scene and shared context it points
    // at are replaced. ~MetalRender drains the GPU and waits for in-flight
    // completion handlers; running that after the Scene/SharedContext it holds
    // raw pointers to have already been freed is a use-after-free waiting for
    // the right timing.
    m_display->resetFrame();
    m_render.reset();

    m_scene = std::move(new_scene);

    oka::Camera camera;
    camera.name = "Main";
    camera.fov = 45.0f;
    camera.position = computeSceneFitPosition(camera.fov);
    camera.mOrientation = glm::quat(glm::vec3(0, 0, 0));
    camera.updateViewMatrix();
    m_scene->addCamera(camera);

    m_selectedCamera = 0;
    setCameraDetached(false);

    loadAnimSettings();

    m_sharedCtx = std::make_unique<SharedContext>();

    m_render = std::unique_ptr<Render>(RenderFactory::createRender());
    m_render->setSettingsManager(m_settingsManager.get());
    m_render->setSharedContext(m_sharedCtx.get());
    m_render->setScene(m_scene.get());
    m_render->init();

    m_cameraController->setCamera(m_scene->getCamera(m_selectedCamera));
    m_render->resetTemporalHistory(); // new scene, new everything
    m_display->setInputHandler(m_cameraController.get());
}

// GPU timing harness (STRELKA_BENCH=<frames>).
//
// Reports the median, not the mean: the first frames warm caches and the
// occasional frame is stretched by an unrelated compositor stall, and both would
// move a mean by more than the effect sizes being measured here. Accumulation is
// off so every frame does the full amount of work.
void EditorApp::runBenchmark()
{
    const uint32_t frames = std::max(4, atoi(getenv("STRELKA_BENCH")));
    const uint32_t warmup = std::max(4u, frames / 4);

    m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    // One submission per frame, so the number is the tracer's cost and not the
    // inter-band gaps of the responsiveness split.
    m_settingsManager->setAs<uint32_t>("render/pt/splitSubmissions", 0);
    if (const char* d = getenv("STRELKA_REF_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", (uint32_t)atoi(d));
    }
    if (const char* tracer = getenv("STRELKA_TRACER"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", (uint32_t)atoi(tracer));
    }
    if (getenv("STRELKA_STAGES"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/profileStages", 1);
    }

    // Playback changes the workload qualitatively — deforming geometry needs
    // two-keyframe acceleration structures — so it needs its own measurement.
    const bool play = getenv("STRELKA_PLAY") != nullptr;
    if (play)
    {
        for (size_t i = 0; i < m_scene->getAnimations().size(); ++i)
        {
            char key[64];
            snprintf(key, sizeof(key), "render/animation/anim%zu/state", i);
            m_settingsManager->setAs<bool>(key, true);
        }
    }

    std::vector<double> samples;
    samples.reserve(frames);
    // Wall clock too: the GPU render time excludes skinning and acceleration
    // structure work, which is most of what changes when playback is on.
    std::vector<double> wall;
    wall.reserve(frames);
    auto prevFrame = std::chrono::high_resolution_clock::now();
    double last = -1.0;
    for (uint32_t i = 0; i < warmup + frames && !m_display->windowShouldClose();)
    {
        m_display->pollEvents();
        if (play)
        {
            playAnimations(1.0 / 60.0);
        }
        if (!m_isLoading)
        {
            m_render->triggerRenderIfIdle();
        }
        const double t = m_render->getLastRenderTimeMs();
        if (t > 0.0 && t != last)
        {
            last = t;
            const auto now = std::chrono::high_resolution_clock::now();
            if (i >= warmup)
            {
                samples.push_back(t);
                wall.push_back(std::chrono::duration<double, std::milli>(now - prevFrame).count());
            }
            prevFrame = now;
            ++i;
        }
        usleep(200);
    }

    std::sort(samples.begin(), samples.end());
    std::sort(wall.begin(), wall.end());
    if (samples.empty())
    {
        STRELKA_INFO("BENCH  no frames measured");
        return;
    }
    const double median = samples[samples.size() / 2];
    STRELKA_INFO("BENCH  tracer={} depth={} frames={}  median={:.2f} ms  min={:.2f}  max={:.2f}  wall={:.2f} ms",
                 m_settingsManager->getAs<uint32_t>("render/pt/tracerMode"),
                 m_settingsManager->getAs<uint32_t>("render/pt/depth"),
                 samples.size(), median, samples.front(), samples.back(),
                 wall.empty() ? 0.0 : wall[wall.size() / 2]);
}

// Jitter-sign measurement (STRELKA_JITTER_TEST=<frames>).
//
// The MetalFX header documents jitterOffsetX in two sentences that imply
// opposite signs, and our own camera flips Y, so the correct value is measured
// rather than argued. A wrong sign does not shift the converged image -- the
// Halton sequence is zero-mean -- it makes each frame's samples land where the
// scaler believes the opposite subpixel position is. The result is a
// reconstruction that never settles: it swims.
//
// So: static camera, accumulation off, one sample per frame, and the denoiser
// left to do all the work. Two numbers per configuration, both over the tail of
// the run so the history is full:
//
//   swim  -- mean |I(t) - I(t-1)| relative to mean intensity. With a static
//            camera a settled reconstruction changes only by the noise it has
//            not yet filtered, so lower is better and the differences are large.
//   sharp -- mean gradient magnitude, relative to mean intensity. A mismatched
//            jitter averages neighbouring subpixel positions together, which is
//            blur, so higher is better.
void EditorApp::runJitterTest()
{
    const uint32_t frames = std::max(16, atoi(getenv("STRELKA_JITTER_TEST")));
    m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/splitSubmissions", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", 1);
    m_settingsManager->setAs<bool>("render/pt/denoise", true);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
    if (const char* up = getenv("STRELKA_UPSCALE"))
    {
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", (float)atof(up));
    }
    if (const char* d = getenv("STRELKA_REF_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", (uint32_t)atoi(d));
    }

    auto capture = [&](std::vector<float>& out, uint32_t& w, uint32_t& h) {
        const double before = m_render->getLastRenderTimeMs();
        for (int i = 0; i < 4000; ++i)
        {
            m_display->pollEvents();
            m_render->triggerRenderIfIdle();
            if (m_render->getLastRenderTimeMs() != before && m_render->getReadyBuffer())
            {
                break;
            }
            usleep(300);
        }
        usleep(4000); // let the denoise and tonemap that follow the trace land
        return m_render->readDisplayTexture(out, w, h);
    };

    // Ground truth: the same view at native resolution, converged, with neither
    // scaler nor denoiser in the path. Both images come out of the display
    // texture at output resolution, so they are directly comparable -- and this is
    // the metric that separates a jitter sign from a noise-versus-blur tradeoff,
    // because a misregistered reconstruction is further from the truth no matter
    // which side of that tradeoff it sits on.
    std::vector<float> truth;
    uint32_t tw = 0, th = 0;
    {
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
        m_settingsManager->setAs<bool>("render/pt/denoise", false);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_sharedCtx->mSubframeIndex = 0;
        while (m_sharedCtx->mSubframeIndex < 512 && !m_display->windowShouldClose())
        {
            m_display->pollEvents();
            m_render->triggerRenderIfIdle();
            usleep(300);
        }
        usleep(200000);
        m_render->readDisplayTexture(truth, tw, th);
        STRELKA_INFO("JITTER truth {}x{} spp={}", tw, th, (uint32_t)m_sharedCtx->mSubframeIndex);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", true);
        m_settingsManager->setAs<bool>("render/pt/denoise", true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
    }

    const char* names[4] = { "(+x,+y)", "(-x,+y)", "(+x,-y)", "(-x,-y)" };
    for (uint32_t sign = 0; sign < 4; ++sign)
    {
        m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", sign);
        m_render->resetTemporalHistory();
        m_sharedCtx->mSubframeIndex = 0;

        std::vector<float> prev, cur;
        uint32_t w = 0, h = 0;
        double swimSum = 0.0, lumSum = 0.0;
        uint32_t swimCount = 0;
        const uint32_t tail = frames / 2;
        for (uint32_t f = 0; f < frames && !m_display->windowShouldClose(); ++f)
        {
            if (!capture(cur, w, h) || cur.empty())
            {
                continue;
            }
            if (f >= tail && prev.size() == cur.size())
            {
                double d = 0.0, l = 0.0;
                for (size_t i = 0; i < cur.size(); i += 4)
                {
                    for (int k = 0; k < 3; ++k)
                    {
                        d += std::abs((double)cur[i + k] - (double)prev[i + k]);
                        l += (double)cur[i + k];
                    }
                }
                swimSum += d;
                lumSum += l;
                ++swimCount;
            }
            prev = cur;
        }

        // Sharpness of the final frame: forward differences, both axes.
        double grad = 0.0, mean = 0.0;
        if (w > 1 && h > 1 && !cur.empty())
        {
            for (uint32_t y = 0; y + 1 < h; ++y)
            {
                for (uint32_t x = 0; x + 1 < w; ++x)
                {
                    const size_t i = ((size_t)y * w + x) * 4;
                    for (int k = 0; k < 3; ++k)
                    {
                        grad += std::abs((double)cur[i + 4 + k] - (double)cur[i + k]) +
                                std::abs((double)cur[i + (size_t)w * 4 + k] - (double)cur[i + k]);
                        mean += (double)cur[i + k];
                    }
                }
            }
        }
        // Distance to the truth, and the shift at which that distance is
        // smallest. A minimum away from (0,0) is a reconstruction that sits in the
        // wrong place -- which is what a jitter sign error looks like when the
        // sequence happens not to average out.
        double bestRmse = -1.0;
        int bestDx = 0, bestDy = 0;
        if (!truth.empty() && tw == w && th == h && !cur.empty())
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    double se = 0.0;
                    size_t n = 0;
                    for (uint32_t y = 1; y + 1 < h; ++y)
                    {
                        for (uint32_t x = 1; x + 1 < w; ++x)
                        {
                            const size_t a = ((size_t)y * w + x) * 4;
                            const size_t b = ((size_t)(y + dy) * w + (x + dx)) * 4;
                            for (int k = 0; k < 3; ++k)
                            {
                                const double d = (double)cur[a + k] - (double)truth[b + k];
                                se += d * d;
                                ++n;
                            }
                        }
                    }
                    const double rmse = n ? std::sqrt(se / (double)n) : 0.0;
                    if (bestRmse < 0.0 || rmse < bestRmse)
                    {
                        bestRmse = rmse;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
            }
        }
        const double swim = (swimCount && lumSum > 0.0) ? swimSum / lumSum : -1.0;
        const double sharp = (mean > 0.0) ? grad / mean : -1.0;
        STRELKA_INFO("JITTER sign={} {}  rmse={:.5f} at shift({},{})  swim={:.5f}  sharp={:.5f}  ({}x{})",
                     sign, names[sign], bestRmse, bestDx, bestDy, swim, sharp, w, h);
    }
}

// --- Denoiser audit (STRELKA_DENOISE_AUDIT=<dir>) ---------------------------
//
// The denoiser's inputs are conventions, not derivations: which way depth grows,
// which space normals are in, which sign a jitter carries, where a motion vector
// points. Every one of them is wrong in a way that still produces a plausible
// picture, so "it looks fine" is not evidence. This measures them.
//
// Two kinds of check, and the first is the one that matters:
//
//   * Guide invariants -- read the exact textures MetalFX consumes and assert
//     things that must hold regardless of taste. A motion vector must be zero
//     when nothing moves and non-zero where something did; a normal must be unit
//     length; an albedo must be in [0,1]. These need no reference image, and they
//     name the broken input instead of reporting that the picture is worse.
//
//   * Reconstruction error -- RMSE against a converged reference, taken at the
//     best of nine one-pixel shifts. The shift is the interesting half: a
//     reconstruction that scores best somewhere other than (0,0) is sitting in
//     the wrong place, which is what a sign error looks like once the noise is
//     gone, and a plain RMSE hides it inside the noise floor.
namespace
{

struct AuditImage
{
    std::vector<float> px;
    uint32_t w = 0, h = 0;
    bool valid() const
    {
        return !px.empty() && w > 1 && h > 1 && px.size() == (size_t)w * h * 4;
    }
};

double auditMean(const AuditImage& img)
{
    double sum = 0.0;
    for (size_t i = 0; i < img.px.size(); i += 4)
        sum += img.px[i] + img.px[i + 1] + img.px[i + 2];
    return img.px.empty() ? 0.0 : sum / (double)(img.px.size() / 4 * 3);
}

// RMSE at the best of nine one-pixel shifts, plus which shift won.
double auditRmse(const AuditImage& a, const AuditImage& b, int& bestDx, int& bestDy)
{
    bestDx = bestDy = 0;
    if (!a.valid() || !b.valid() || a.w != b.w || a.h != b.h)
        return -1.0;
    double best = -1.0;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            double se = 0.0;
            size_t n = 0;
            for (uint32_t y = 1; y + 1 < a.h; ++y)
            {
                for (uint32_t x = 1; x + 1 < a.w; ++x)
                {
                    const size_t ia = ((size_t)y * a.w + x) * 4;
                    const size_t ib = ((size_t)(y + dy) * a.w + (x + dx)) * 4;
                    for (int k = 0; k < 3; ++k)
                    {
                        const double d = (double)a.px[ia + k] - (double)b.px[ib + k];
                        se += d * d;
                        ++n;
                    }
                }
            }
            const double rmse = n ? std::sqrt(se / (double)n) : 0.0;
            if (best < 0.0 || rmse < best)
            {
                best = rmse;
                bestDx = dx;
                bestDy = dy;
            }
        }
    }
    return best;
}

// Mean absolute frame-to-frame change, relative to intensity. With a still
// camera and a still scene a settled reconstruction only changes by the noise it
// has not filtered yet, so lower is better and the spread between right and
// wrong is large.
double auditSwim(const AuditImage& a, const AuditImage& b)
{
    if (!a.valid() || !b.valid() || a.px.size() != b.px.size())
        return -1.0;
    double d = 0.0, l = 0.0;
    for (size_t i = 0; i < a.px.size(); i += 4)
    {
        for (int k = 0; k < 3; ++k)
        {
            d += std::abs((double)a.px[i + k] - (double)b.px[i + k]);
            l += (double)a.px[i + k];
        }
    }
    return l > 0.0 ? d / l : -1.0;
}

// Mean forward-difference gradient, relative to intensity. A reconstruction that
// averages the wrong subpixel positions together is blurred, and blur shows here
// before it shows anywhere else.
double auditSharp(const AuditImage& img)
{
    if (!img.valid())
        return -1.0;
    double grad = 0.0, mean = 0.0;
    for (uint32_t y = 0; y + 1 < img.h; ++y)
    {
        for (uint32_t x = 0; x + 1 < img.w; ++x)
        {
            const size_t i = ((size_t)y * img.w + x) * 4;
            for (int k = 0; k < 3; ++k)
            {
                grad += std::abs((double)img.px[i + 4 + k] - (double)img.px[i + k]) +
                        std::abs((double)img.px[i + (size_t)img.w * 4 + k] - (double)img.px[i + k]);
                mean += (double)img.px[i + k];
            }
        }
    }
    return mean > 0.0 ? grad / mean : -1.0;
}

bool auditFinite(const AuditImage& img)
{
    for (float v : img.px)
    {
        if (!std::isfinite(v))
            return false;
    }
    return true;
}

} // namespace

void EditorApp::runDenoiseAudit()
{
    const char* outDir = getenv("STRELKA_DENOISE_AUDIT");
    const bool saveImages = outDir && outDir[0] && strchr(outDir, '/') != nullptr;
    // The reference is bounded by samples, not by time: past a couple of hundred
    // the estimator has converged and the renderer is only re-showing the same
    // picture, so waiting longer buys nothing.
    const uint32_t refSpp = (uint32_t)atoi(getenv("STRELKA_AUDIT_SPP") ? getenv("STRELKA_AUDIT_SPP") : "256");
    const uint32_t frames = (uint32_t)atoi(getenv("STRELKA_AUDIT_FRAMES") ? getenv("STRELKA_AUDIT_FRAMES") : "12");
    // A quarter of the editor's default pixel count. Every question here is about
    // whether a guide means what the denoiser thinks it means, and none of them
    // need a big image -- while the whole audit has to fit in the time a person
    // is willing to sit in front of it.
    const uint32_t auditW = (uint32_t)atoi(getenv("STRELKA_AUDIT_W") ? getenv("STRELKA_AUDIT_W") : "512");
    const uint32_t auditH = (uint32_t)atoi(getenv("STRELKA_AUDIT_H") ? getenv("STRELKA_AUDIT_H") : "384");
    const float upscale =
        (float)atof(getenv("STRELKA_UPSCALE") ? getenv("STRELKA_UPSCALE") : "0.5");
    // Wall-clock budgets. The audit drives the renderer through states it is not
    // known to survive -- that is the point of it -- so it must not be able to
    // wait forever on a frame that is never coming, and it must report whatever
    // it already measured when it runs out of time rather than nothing at all.
    const double budgetSec =
        atof(getenv("STRELKA_AUDIT_BUDGET") ? getenv("STRELKA_AUDIT_BUDGET") : "90");
    const double stepTimeoutSec =
        atof(getenv("STRELKA_AUDIT_STEP_SEC") ? getenv("STRELKA_AUDIT_STEP_SEC") : "10");
    const auto auditStart = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - auditStart).count();
    };
    // Distinguished on purpose: a closed window means the run was abandoned and
    // its numbers are partial, which is a different statement from having spent
    // the budget.
    auto outOfTime = [&]() { return elapsed() > budgetSec || m_display->windowShouldClose(); };

    // Every result goes to stdout as well as the log, unbuffered. spdlog batches
    // its writes, and a harness whose job is to walk the renderer into states
    // that might abort has to survive one: numbers that only exist in a buffer
    // that never got flushed are numbers that were never measured.
    auto report = [&](const std::string& line) {
        STRELKA_INFO("{}", line);
        std::fputs(line.c_str(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    };

    // Deterministic conditions for everything below. Motion blur off: it makes
    // the shutter, and therefore the pose the frame is rendered at, depend on
    // playback state, and the audit drives time by hand.
    m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", 1);
    m_settingsManager->setAs<uint32_t>("render/pt/splitSubmissions", 0);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
    m_settingsManager->setAs<uint32_t>("render/width", auditW);
    m_settingsManager->setAs<uint32_t>("render/height", auditH);

    auto& camera = m_scene->getCamera(m_selectedCamera);

    // --- frame pump ---------------------------------------------------------
    // One frame, submitted and landed. isRenderBusy() is what makes this exact:
    // sleeping a guessed interval instead would occasionally read the frame
    // before it, and every number here would carry that as noise.
    // Two phases, timed separately. triggerRenderIfIdle() encodes the frame on
    // this thread, and the first frame of a new configuration pays for MetalFX
    // building its network synchronously -- seconds, legitimately. Timing that
    // against a "did the frame land" budget reports a failure for work that was
    // proceeding normally, and every measurement taken afterwards is then of a
    // stale texture. Only the wait for the GPU is held to the timeout.
    auto step = [&]() -> bool {
        const size_t target = m_sharedCtx->mFrameNumber + 1;
        bool submitted = false;
        auto phaseStart = std::chrono::steady_clock::now();
        while (!m_display->windowShouldClose())
        {
            m_display->pollEvents();
            // Only until this frame is away. Triggering again on every iteration
            // starts the next frame the instant this one lands, so the renderer
            // is never observed idle and every read below sees whichever frame
            // happened to be finished -- not the one whose inputs were just set.
            if (!submitted)
            {
                m_render->triggerRenderIfIdle();
                if (m_sharedCtx->mFrameNumber >= target)
                {
                    submitted = true;
                    phaseStart = std::chrono::steady_clock::now();
                }
            }
            if (submitted && !m_render->isRenderBusy())
                return true;
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - phaseStart).count() >
                stepTimeoutSec)
            {
                report(fmt::format("AUDIT WARN frame did not {} within {:.0f}s (frame={} lastGpu={:.1f}ms)",
                                   submitted ? "complete" : "submit", stepTimeoutSec,
                                   m_sharedCtx->mFrameNumber, m_render->getLastRenderTimeMs()));
                return false;
            }
            usleep(200);
        }
        return false;
    };
    auto shown = [&](AuditImage& img) { return m_render->readDisplayTexture(img.px, img.w, img.h); };
    auto guide = [&](Render::Guide g, AuditImage& img) {
        return m_render->readGuideTexture(g, img.px, img.w, img.h);
    };

    const bool denoiseWithAcc = getenv("STRELKA_AUDIT_NO_ACC") == nullptr;
    auto setDenoise = [&](bool on) {
        m_settingsManager->setAs<bool>("render/pt/denoise", on);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", on);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", on ? denoiseWithAcc : true);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal",
                                           on ? (denoiseWithAcc ? refSpp : 1u) : refSpp);
    };

    // Converged, at native resolution, with neither scaler nor denoiser in the
    // path: the picture the reconstruction is trying to be.
    auto converge = [&](AuditImage& img) {
        setDenoise(false);
        m_sharedCtx->mSubframeIndex = 0;
        m_render->resetTemporalHistory();
        // Bounded by samples *and* by frames: the estimator restarts its
        // accumulation on any change it considers material, and a reference that
        // keeps being restarted would otherwise never finish.
        uint32_t guard = refSpp * 4 + 16;
        while (m_sharedCtx->mSubframeIndex < refSpp && guard-- > 0 && !outOfTime())
        {
            if (!step())
                break;
        }
        step(); // one more so the tonemap of the final accumulation lands
        return shown(img);
    };

    auto save = [&](const char* name, const AuditImage& img) {
        if (!saveImages || !img.valid())
            return;
        const char* err = nullptr;
        SaveEXR(img.px.data(), (int)img.w, (int)img.h, 4, 0,
                (std::string(outDir) + "/" + name + ".exr").c_str(), &err);
        if (err)
            FreeEXRErrorMessage(err);
    };

    // Yaw orbit about whatever the scene's own camera is already looking at.
    //
    // Deriving the pose from the vertex bounds instead looks reasonable and is
    // wrong: getVertices() is object space, so on any scene that places its
    // meshes with a node transform the camera ends up pointing at nothing. The
    // distance to the subject is therefore measured -- one frame, median of the
    // finite depths -- rather than assumed, which works on any scene without
    // knowing anything about it.
    const glm::vec3 startPos = camera.position;
    const glm::quat startOrient = camera.mOrientation;
    const glm::vec3 startFront = glm::conjugate(startOrient) * glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 orbitTarget = startPos + startFront;
    auto setOrbit = [&](float theta) {
        // view = rot(mOrientation) * translate(-p) and getFront() conjugates, so
        // the stored quaternion is the inverse of the camera's world rotation.
        const glm::quat yaw = glm::angleAxis(theta, glm::vec3(0.0f, 1.0f, 0.0f));
        camera.position = orbitTarget + yaw * (startPos - orbitTarget);
        camera.mOrientation = startOrient * glm::angleAxis(-theta, glm::vec3(0.0f, 1.0f, 0.0f));
        camera.updateViewMatrix();
    };

    const size_t animCount = m_scene->getAnimations().size();
    auto setAnimTime = [&](float t01) {
        char key[64];
        for (size_t i = 0; i < animCount; ++i)
        {
            const auto& a = m_scene->getAnimations()[i];
            snprintf(key, sizeof(key), "render/animation/anim%zu/time", i);
            m_settingsManager->setAs<float>(key, a.start + (a.end - a.start) * t01);
        }
    };

    report(fmt::format("AUDIT scene animations={} {}x{} refSpp={} frames={} upscale={:.2f} budget={:.0f}s",
                       animCount, auditW, auditH, refSpp, frames, upscale, budgetSec));

    // === Guide invariants ===================================================
    setOrbit(0.0f);
    setAnimTime(0.0f);
    setDenoise(true);
    m_render->resetTemporalHistory();
    for (int i = 0; i < 4; ++i)
        step();

    // Calibrate the orbit distance from what the camera can actually see.
    {
        AuditImage d;
        std::vector<float> finite;
        if (guide(Render::Guide::Depth, d))
        {
            for (size_t i = 0; i < d.px.size(); i += 4)
            {
                if (d.px[i] > 0.0f && d.px[i] < 1e6f)
                    finite.push_back(d.px[i]);
            }
        }
        if (finite.size() > d.px.size() / 40) // at least 10% of the frame is geometry
        {
            std::sort(finite.begin(), finite.end());
            orbitTarget = startPos + startFront * finite[finite.size() / 2];
        }
        report(fmt::format("AUDIT orbit target at {:.2f} units ({:.1f}% of frame is geometry)",
                           glm::length(orbitTarget - startPos),
                           d.px.empty() ? 0.0 : 100.0 * (double)finite.size() / (double)(d.px.size() / 4)));
        setOrbit(0.0f);
        step();
    }

    auto reportGuide = [&](const char* label, Render::Guide g, float lo, float hi, int channels) {
        AuditImage img;
        if (!guide(g, img))
        {
            report(fmt::format("AUDIT FAIL guide {:14s} UNAVAILABLE", label));
            return;
        }
        size_t bad = 0, nonFinite = 0;
        float mn = std::numeric_limits<float>::max(), mx = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < img.px.size(); i += 4)
        {
            for (int k = 0; k < channels; ++k)
            {
                const float v = img.px[i + k];
                if (!std::isfinite(v)) { ++nonFinite; continue; }
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                if (v < lo || v > hi) ++bad;
            }
        }
        report(fmt::format(
            "AUDIT guide {:14s} {}x{}  range[{:.4f},{:.4f}]  outside[{:.2f},{:.2f}]={}  nonfinite={}", label,
            img.w, img.h, mn, mx, lo, hi, bad, nonFinite));
    };
    reportGuide("color", Render::Guide::Color, 0.0f, 1e9f, 3);
    reportGuide("depth", Render::Guide::Depth, 0.0f, 1e9f, 1);
    reportGuide("diffuseAlbedo", Render::Guide::DiffuseAlbedo, 0.0f, 1.0f, 3);
    reportGuide("specAlbedo", Render::Guide::SpecularAlbedo, 0.0f, 1.0f, 3);
    reportGuide("roughness", Render::Guide::Roughness, 0.0f, 1.0f, 1);
    reportGuide("specHitDistance", Render::Guide::SpecularHitDistance, 0.0f, 1e9f, 1);
    reportGuide("reactive", Render::Guide::Reactive, 0.0f, 1.0f, 1);
    {
        AuditImage n;
        if (guide(Render::Guide::Normal, n))
        {
            size_t offUnit = 0;
            for (size_t i = 0; i < n.px.size(); i += 4)
            {
                const double len = std::sqrt((double)n.px[i] * n.px[i] + (double)n.px[i + 1] * n.px[i + 1] +
                                             (double)n.px[i + 2] * n.px[i + 2]);
                if (len < 0.9 || len > 1.1)
                    ++offUnit;
            }
            report(fmt::format("AUDIT guide {:14s} non-unit={} / {}", "normal", offUnit, n.px.size() / 4));
        }
    }

    // Motion vectors, the three cases that decide whether the denoiser can
    // reproject anything at all.
    auto motionStats = [&](const char* label, size_t& nonZero, size_t& total) {
        AuditImage mv;
        nonZero = total = 0;
        if (!guide(Render::Guide::Motion, mv))
        {
            report(fmt::format("AUDIT FAIL motion {:22s} UNAVAILABLE", label));
            return AuditImage{};
        }
        double maxMag = 0.0;
        for (size_t i = 0; i < mv.px.size(); i += 4)
        {
            const double m = std::abs((double)mv.px[i]) + std::abs((double)mv.px[i + 1]);
            maxMag = std::max(maxMag, m);
            if (m > 0.05) ++nonZero;
            ++total;
        }
        report(fmt::format("AUDIT motion {:22s} nonzero={:.1f}%  max={:.2f} px", label,
                           total ? 100.0 * (double)nonZero / (double)total : 0.0, maxMag));
        return mv;
    };

    size_t nz = 0, tot = 0;
    motionStats("still camera+scene", nz, tot);
    if (nz * 200 > tot) // more than 0.5% moving when nothing does
        report("AUDIT FAIL motion vectors non-zero with nothing moving");

    // Camera moved: everything on screen must move, background included.
    AuditImage beforeCam, afterCam;
    guide(Render::Guide::Color, beforeCam);
    setOrbit(0.05f);
    step();
    motionStats("camera moved", nz, tot);
    if (nz * 4 < tot * 3) // fewer than 75% of pixels carrying motion
        report("AUDIT FAIL camera motion missing from motion vectors (background?)");

    // Scene moved, camera still: the pixels whose shading changed are exactly the
    // ones that must carry a motion vector. Counting the ones that changed and
    // did not is a ghosting predictor that needs no reference image.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        m_render->resetTemporalHistory();
        for (int i = 0; i < 3; ++i)
            step();
        AuditImage a0;
        guide(Render::Guide::Depth, a0);
        // Control: one frame with nothing changed at all. The primary hit still
        // lands somewhere slightly different every frame because the jitter moves
        // it, so "depth differs" on its own marks the whole image as moving. The
        // control measures that floor and the test clears it.
        step();
        AuditImage ctrl;
        guide(Render::Guide::Depth, ctrl);
        // The 99th percentile, not the maximum: a pixel on a silhouette flips
        // between the surface and the background as the jitter moves it, and that
        // one pixel would otherwise set a threshold nothing can clear.
        std::vector<double> floorDeltas;
        if (a0.valid() && ctrl.valid())
        {
            for (size_t i = 0; i < ctrl.px.size(); i += 4)
            {
                const double d0 = (double)a0.px[i], d1 = (double)ctrl.px[i];
                if (d0 <= 0.0 || d0 >= 1e6 || d1 >= 1e6)
                    continue;
                floorDeltas.push_back(std::abs(d1 - d0) / d0);
            }
            std::sort(floorDeltas.begin(), floorDeltas.end());
        }
        const double floorDelta =
            floorDeltas.empty() ? 0.0 : floorDeltas[(size_t)(floorDeltas.size() * 0.99)];
        const double movedThreshold = std::max(4.0 * floorDelta, 0.01);

        setAnimTime(0.08f);
        step();
        AuditImage a1, mv;
        guide(Render::Guide::Depth, a1);
        guide(Render::Guide::Motion, mv);
        size_t changed = 0, changedNoMv = 0;
        if (a0.valid() && a1.valid() && a1.px.size() == mv.px.size())
        {
            for (size_t i = 0; i < a1.px.size(); i += 4)
            {
                // Depth, not colour: colour also changes with the noise seed.
                const double d0 = (double)a0.px[i], d1 = (double)a1.px[i];
                if (d0 <= 0.0 || d0 >= 1e6 || d1 >= 1e6)
                    continue; // background: nothing here to deform
                if (std::abs(d1 - d0) / d0 < movedThreshold)
                    continue;
                ++changed;
                if (std::abs((double)mv.px[i]) + std::abs((double)mv.px[i + 1]) <= 0.25)
                    ++changedNoMv;
            }
        }
        report(fmt::format("AUDIT motion {:22s} jitter floor={:.4f}, threshold={:.4f}", "scene moved",
                           floorDelta, movedThreshold));
        report(fmt::format("AUDIT motion {:22s} moved={} px, of which without motion vector={} ({:.1f}%)",
                           "scene moved", changed, changedNoMv,
                           changed ? 100.0 * (double)changedNoMv / (double)changed : 0.0));
        if (changed && changedNoMv * 10 > changed)
            report("AUDIT FAIL deforming geometry has no motion vectors");
    }

    // === S1 static camera ===================================================
    auto reconstruct = [&](const char* label, float thetaFrom, float thetaTo, float animFrom, float animTo,
                           AuditImage& last) {
        setDenoise(true);
        setOrbit(thetaFrom);
        setAnimTime(animFrom);
        m_render->resetTemporalHistory();
        // Fill the history before measuring anything. A temporal reconstruction
        // is only itself once it has one; measuring from the reset instead scores
        // whatever state the previous scenario happened to leave behind, which
        // moved this number by an order of magnitude between runs.
        for (uint32_t w = 0; w < 8 && !outOfTime(); ++w)
        {
            if (!step())
                break;
        }
        AuditImage prev;
        double swimSum = 0.0;
        double firstMean = -1.0;
        uint32_t swimCount = 0;
        for (uint32_t f = 0; f < frames && !outOfTime(); ++f)
        {
            const float u = frames > 1 ? (float)f / (float)(frames - 1) : 1.0f;
            setOrbit(thetaFrom + (thetaTo - thetaFrom) * u);
            setAnimTime(animFrom + (animTo - animFrom) * u);
            if (!step())
                break;
            AuditImage cur;
            if (!shown(cur))
                continue;
            if (firstMean < 0.0)
                firstMean = auditMean(cur);
            if (f >= frames / 2 && prev.valid())
            {
                const double s = auditSwim(prev, cur);
                if (s >= 0.0)
                {
                    swimSum += s;
                    ++swimCount;
                }
            }
            prev = std::move(cur);
        }
        last = prev;
        // Whether the picture was already wrong or drifted there matters: the
        // first tells you an input is wrong, the second that the accumulation is.
        report(fmt::format("AUDIT {:14s} mean over run: first={:.4f} last={:.4f}", label, firstMean,
                           auditMean(prev)));
        return swimCount ? swimSum / swimCount : -1.0;
    };

    // === Convention sweep (STRELKA_AUDIT_SWEEP=1) ===========================
    //
    // Jitter sign and depth encoding are conventions MetalFX does not document
    // unambiguously, and both fail the same way: the picture still looks like the
    // scene, it just never resolves. A still camera is the sharpest test there is
    // -- with nothing moving, a correct reconstruction converges to the reference
    // and a mismatched one averages neighbouring subpixel positions forever -- so
    // the sweep scores every combination against one converged reference.
    if (getenv("STRELKA_AUDIT_SWEEP"))
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        AuditImage truth;
        converge(truth);
        const double truthSharp = auditSharp(truth);
        report(fmt::format("AUDIT sweep reference sharp={:.5f}", truthSharp));

        for (uint32_t depthMode = 0; depthMode < 3 && !outOfTime(); ++depthMode)
        {
            for (uint32_t sign = 0; sign < 4 && !outOfTime(); ++sign)
            {
                m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", depthMode);
                m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", sign);
                AuditImage last;
                const double swim = reconstruct("sweep", 0.0f, 0.0f, 0.0f, 0.0f, last);
                int dx = 0, dy = 0;
                const double rmse = auditRmse(last, truth, dx, dy);
                static const char* kDepthNames[3] = { "device", "viewZ", "radial" };
                static const char* kSignNames[4] = { "(+x,+y)", "(-x,+y)", "(+x,-y)", "(-x,-y)" };
                report(fmt::format(
                    "AUDIT sweep depth={:6s} jitter={:7s} rmse={:.5f} shift({},{}) swim={:.5f} sharp={:.5f}",
                    kDepthNames[depthMode], kSignNames[sign], rmse, dx, dy, swim, auditSharp(last)));
            }
        }
        m_settingsManager->setAs<uint32_t>("render/pt/denoiseDepthMode", 0);
        m_settingsManager->setAs<uint32_t>("render/pt/jitterSign", 0);
    }

    struct Scenario
    {
        const char* name;
        float thetaFrom, thetaTo;
        float animFrom, animTo;
    };
    const float orbitSpan = 0.35f; // radians over the whole run
    std::vector<Scenario> scenarios = {
        { "S1 static", 0.0f, 0.0f, 0.0f, 0.0f },
        { "S2 orbit", -orbitSpan, orbitSpan, 0.0f, 0.0f },
    };
    if (animCount > 0)
    {
        scenarios.push_back({ "S3 animation", 0.0f, 0.0f, 0.0f, 0.25f });
        scenarios.push_back({ "S4 anim+orbit", -orbitSpan, orbitSpan, 0.0f, 0.25f });
    }

    for (const Scenario& s : scenarios)
    {
        if (outOfTime())
        {
            report(fmt::format("AUDIT WARN out of budget, skipping {} onwards", s.name));
            break;
        }
        AuditImage last;
        const double swim = reconstruct(s.name, s.thetaFrom, s.thetaTo, s.animFrom, s.animTo, last);

        // Reference at the pose and time the run ended on.
        setOrbit(s.thetaTo);
        setAnimTime(s.animTo);
        AuditImage truth;
        converge(truth);

        int dx = 0, dy = 0;
        const double rmse = auditRmse(last, truth, dx, dy);
        // Quantified, not just flagged: a best-shift away from centre only means
        // something if staying at centre actually costs more error.
        double rmse00 = -1.0;
        {
            double se = 0.0; size_t n = 0;
            if (last.valid() && truth.valid() && last.px.size() == truth.px.size())
            {
                for (uint32_t y = 1; y + 1 < last.h; ++y)
                    for (uint32_t x = 1; x + 1 < last.w; ++x)
                        for (int k = 0; k < 3; ++k)
                        {
                            const size_t i = ((size_t)y * last.w + x) * 4 + k;
                            const double d = (double)last.px[i] - (double)truth.px[i];
                            se += d * d; ++n;
                        }
                rmse00 = n ? std::sqrt(se / (double)n) : -1.0;
            }
        }
        report(fmt::format(
            "AUDIT {:14s} rmse@centre={:.5f} (best {:.5f} at shift {},{})", s.name, rmse00, rmse, dx, dy));
        report(fmt::format(
            "AUDIT {:14s} rmse={:.5f} at shift({},{})  swim={:.5f}  sharp={:.5f}/{:.5f}  mean={:.4f}/{:.4f}",
            s.name, rmse, dx, dy, swim, auditSharp(last), auditSharp(truth), auditMean(last),
            auditMean(truth)));
        // Only a real misregistration if centring costs more than a few percent.
        if (rmse >= 0.0 && (dx != 0 || dy != 0) && rmse00 > rmse * 1.05)
            report(fmt::format("AUDIT FAIL {} reconstruction is misregistered by ({},{}), centre costs {:.1f}%",
                               s.name, dx, dy, 100.0 * (rmse00 / rmse - 1.0)));
        // Uniformly darker and partly black are different faults with the same
        // mean, and only one of them is an exposure problem.
        if (last.valid() && truth.valid() && last.px.size() == truth.px.size())
        {
            std::vector<double> ratios;
            size_t nearBlack = 0, lit = 0;
            for (size_t i = 0; i < last.px.size(); i += 4)
            {
                const double t = (truth.px[i] + truth.px[i + 1] + truth.px[i + 2]) / 3.0;
                const double d = (last.px[i] + last.px[i + 1] + last.px[i + 2]) / 3.0;
                if (t <= 1e-4)
                    continue;
                ++lit;
                if (d < 0.1 * t)
                    ++nearBlack;
                ratios.push_back(d / t);
            }
            std::sort(ratios.begin(), ratios.end());
            report(fmt::format("AUDIT {:14s} vs truth: median ratio={:.4f}  pixels below 10% of truth={:.1f}%",
                               s.name, ratios.empty() ? 0.0 : ratios[ratios.size() / 2],
                               lit ? 100.0 * (double)nearBlack / (double)lit : 0.0));
        }
        save((std::string(s.name).substr(0, 2) + "_denoised").c_str(), last);
        save((std::string(s.name).substr(0, 2) + "_truth").c_str(), truth);
    }

    // === Playback ===========================================================
    //
    // Pressing play is not the same code path as scrubbing time: it sets the
    // animation state, which turns on motion blur, which swaps the scene onto
    // two-keyframe acceleration structures. Everything above drives time by hand
    // with motion blur off and never touches that.
    //
    // Two different measurements, because "the mesh is gone" has two different
    // causes that look identical on screen: geometry coverage says whether rays
    // still hit anything (is it in the acceleration structure?) and the displayed
    // image says whether what they hit is visible (is it being shaded and shown?).
    // Coverage alone was measured first and reported nothing wrong, which was the
    // wrong question.
    if (animCount > 0)
    {
        for (int denoiseOn = 1; denoiseOn >= 0; --denoiseOn)
        {
            setOrbit(0.0f);
            setAnimTime(0.0f);
            // Guides are only written when denoising, so coverage can only be read
            // in that mode; the displayed image can be read in both.
            setDenoise(denoiseOn != 0);
            m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
            m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
                step();

            auto coverage = [&]() {
                AuditImage d;
                if (!guide(Render::Guide::Depth, d) || d.px.empty())
                    return -1.0;
                size_t hits = 0, total = 0;
                for (size_t i = 0; i < d.px.size(); i += 4)
                {
                    ++total;
                    if (d.px[i] > 0.0f && d.px[i] < 1e6f)
                        ++hits;
                }
                return total ? 100.0 * (double)hits / (double)total : -1.0;
            };
            AuditImage still;
            shown(still);
            const double meanBefore = auditMean(still);
            const double covBefore = coverage();
            const float extentBefore = m_render->skinnedGeometryExtent();

            char key[64];
            for (size_t i = 0; i < animCount; ++i)
            {
                snprintf(key, sizeof(key), "render/animation/anim%zu/state", i);
                m_settingsManager->setAs<bool>(key, true);
            }

            // Driven through the editor's own playAnimations() at a realistic
            // frame time, not by setting normalised times. That is the only way
            // to exercise what actually happens on Play: animations of different
            // lengths drifting apart and wrapping independently.
            const int playFrames = 300;
            double worstCov = 1e9, worstMean = 1e9;
            float worstExtent = 1e9f;
            std::string trace;
            for (int i = 0; i < playFrames && !outOfTime(); ++i)
            {
                playAnimations(1.0f / 60.0f);
                if (!step())
                    break;
                AuditImage img;
                if (shown(img) && img.valid())
                {
                    const double m = auditMean(img);
                    if (m < worstMean)
                    {
                        worstMean = m;
                    }
                    if (i % 20 == 0)
                    {
                        trace += fmt::format("{:.2f} ", m);
                    }
                }
                const double c = coverage();
                if (c >= 0.0)
                    worstCov = std::min(worstCov, c);
                const float e = m_render->skinnedGeometryExtent();
                if (e >= 0.0f)
                    worstExtent = std::min(worstExtent, e);
            }
            for (size_t i = 0; i < animCount; ++i)
            {
                snprintf(key, sizeof(key), "render/animation/anim%zu/state", i);
                m_settingsManager->setAs<bool>(key, false);
            }

            // Is the character actually moving? A stable picture proves nothing
            // if the animation never reached the geometry -- that would look
            // exactly like "nothing broke".
            AuditImage endFrame;
            shown(endFrame);
            int mdx = 0, mdy = 0;
            const double movedBy = auditRmse(still, endFrame, mdx, mdy);
            const char* label = denoiseOn ? "playback denoise" : "playback plain  ";
            report(fmt::format("AUDIT {} image changed over playback by rmse={:.5f}", label, movedBy));
            report(fmt::format("AUDIT {} displayed mean over playback: {}", label, trace));
            report(fmt::format("AUDIT {} mean before={:.4f} worst={:.4f}   coverage before={:.1f}% worst={:.1f}%",
                               label, meanBefore, worstMean >= 1e9 ? -1.0 : worstMean, covBefore,
                               worstCov >= 1e9 ? -1.0 : worstCov));
            // The character itself, not the picture. A skinned mesh collapsing to
            // a point is invisible to every whole-frame metric.
            if (extentBefore >= 0.0f)
            {
                const float worst = worstExtent >= 1e9f ? extentBefore : worstExtent;
                report(fmt::format("AUDIT {} skinned geometry extent before={:.3f} worst={:.3f}", label,
                                   extentBefore, worst));
                // Absolute, not relative to the start of this scenario: by the
                // time playback runs the character may already have collapsed in
                // an earlier phase, and a ratio against zero notices nothing.
                if (worst < 1e-3f || extentBefore < 1e-3f)
                    report(fmt::format("AUDIT FAIL {} skinned geometry collapsed to a point", label));
                else if (worst < extentBefore * 0.25f)
                    report(fmt::format("AUDIT FAIL {} skinned geometry collapsed ({:.3f} -> {:.3f})", label,
                                       extentBefore, worst));
            }
            if (meanBefore > 0.0 && worstMean < 1e9 && worstMean < meanBefore * 0.6)
                report(fmt::format("AUDIT FAIL {} picture collapsed during playback ({:.4f} -> {:.4f})", label,
                                   meanBefore, worstMean));
        }
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    }

    // === Motion vector correctness ==========================================
    //
    // Everything above only asked whether motion vectors exist and are non-zero.
    // That is not the question. A vector is right when following it lands on the
    // same surface one frame earlier, and a wrong vector is non-zero too.
    //
    // Each axis is tested on its own, by displacing the camera along it. An
    // orbiting camera moves the image horizontally, so a test driven by one leaves
    // the vertical sign unexercised -- and a sign fixed on one axis while the
    // other is untested is a coin flip, not a fix.
    {
        setOrbit(0.0f);
        setAnimTime(0.10f);
        setDenoise(true);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", 1.0f);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);

        const glm::vec3 basePos = camera.position;
        const float nudge = 0.02f * glm::length(orbitTarget - startPos);
        const glm::vec3 right = glm::conjugate(camera.mOrientation) * glm::vec3(1.0f, 0.0f, 0.0f);
        const glm::vec3 up = glm::conjugate(camera.mOrientation) * glm::vec3(0.0f, 1.0f, 0.0f);

        for (int axis = 0; axis < 2 && !outOfTime(); ++axis)
        {
            camera.position = basePos;
            camera.updateViewMatrix();
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
                step();
            AuditImage d0;
            guide(Render::Guide::Depth, d0);

            camera.position = basePos + (axis == 0 ? right : up) * nudge;
            camera.updateViewMatrix();
            step();
            AuditImage d1, mv;
            guide(Render::Guide::Depth, d1);
            guide(Render::Guide::Motion, mv);

            size_t moved = 0, hitWithout = 0, hitVariant[4] = { 0, 0, 0, 0 };
            if (d0.valid() && d1.valid() && mv.valid() && d0.px.size() == d1.px.size())
            {
                const int w = (int)d1.w, h = (int)d1.h;
                auto depthAt = [&](const AuditImage& img, int x, int y) {
                    x = std::clamp(x, 0, w - 1);
                    y = std::clamp(y, 0, h - 1);
                    return (double)img.px[((size_t)y * w + x) * 4];
                };
                for (int y = 0; y < h; ++y)
                {
                    for (int x = 0; x < w; ++x)
                    {
                        const size_t p = (size_t)y * w + x;
                        const double now = d1.px[p * 4];
                        if (now <= 0.0 || now >= 1e6)
                            continue;
                        const double mx = mv.px[p * 4], my = mv.px[p * 4 + 1];
                        if (std::abs(mx) + std::abs(my) < 0.5)
                            continue;
                        ++moved;
                        auto agrees = [&](double d) {
                            return d > 0.0 && d < 1e6 && std::abs(d - now) / now < 0.02;
                        };
                        const double sx[4] = { mx, -mx, mx, -mx };
                        const double sy[4] = { my, my, -my, -my };
                        for (int v = 0; v < 4; ++v)
                        {
                            if (agrees(depthAt(d0, x + (int)std::lround(sx[v]),
                                               y + (int)std::lround(sy[v]))))
                                ++hitVariant[v];
                        }
                        if (agrees(depthAt(d0, x, y)))
                            ++hitWithout;
                    }
                }
            }
            const double pct = moved ? 100.0 / (double)moved : 0.0;
            report(fmt::format("AUDIT motioncheck camera {} : n={} ignoring={:.1f}%  (+x,+y)={:.1f}%  "
                               "(-x,+y)={:.1f}%  (+x,-y)={:.1f}%  (-x,-y)={:.1f}%",
                               axis == 0 ? "right" : "up   ", moved, hitWithout * pct,
                               hitVariant[0] * pct, hitVariant[1] * pct, hitVariant[2] * pct,
                               hitVariant[3] * pct));
            if (moved > 100 && hitVariant[0] <= hitWithout)
                report(fmt::format("AUDIT FAIL motion vectors ({} move) are no better than assuming "
                                   "nothing moved",
                                   axis == 0 ? "horizontal" : "vertical"));
        }
        camera.position = basePos;
        camera.updateViewMatrix();
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
    }

    // === Denoise on moving geometry =========================================
    //
    // A whole-frame RMSE cannot answer this. The animated character occupies a
    // small share of the pixels, so it can ghost badly while the number barely
    // moves -- the same blind spot that let a completely absent robot pass. So
    // the error is measured separately over the pixels that actually move and
    // over the ones that do not, and the interesting quantity is the ratio.
    //
    // The mask comes from two converged references at different times, i.e. from
    // ground truth, not from the thing under test.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        // Motion blur changes what a motion vector even means: the traced hit sits
        // at a random time inside the shutter while the previous pose is the last
        // frame's shutter close, so every sample's vector spans a slightly
        // different interval. The editor has it on during playback, so it has to
        // be measurable here.
        const bool mbOn = getenv("STRELKA_AUDIT_MB") != nullptr;
        m_settingsManager->setAs<bool>("render/enableMotionBlur", mbOn);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", mbOn);

        const float tA = 0.10f, tB = 0.16f;
        std::vector<uint8_t> depthMask;
        size_t depthMovedCount = 0;
        setAnimTime(tA);
        AuditImage truthA;
        converge(truthA);
        setAnimTime(tB);
        AuditImage truthB;
        converge(truthB);

        // Geometry, not lighting. A mask built from how the *picture* changes
        // between two animation times also catches the character's shadow and the
        // light it bounces, which are not the animated mesh and denoise
        // differently. The depth guide moves only where geometry does.
        {
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", 1.0f);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            setDenoise(true);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            setAnimTime(tA);
            for (int i = 0; i < 3 && !outOfTime(); ++i)
                step();
            AuditImage dA;
            guide(Render::Guide::Depth, dA);
            setAnimTime(tB);
            for (int i = 0; i < 3 && !outOfTime(); ++i)
                step();
            AuditImage dB;
            guide(Render::Guide::Depth, dB);
            if (dA.valid() && dB.valid() && dA.px.size() == dB.px.size())
            {
                depthMask.assign(dA.px.size() / 4, 2);
                for (size_t i = 0, p = 0; i < dA.px.size(); i += 4, ++p)
                {
                    const double a = dA.px[i], b = dB.px[i];
                    const bool bg = a <= 0.0 || a >= 1e6 || b >= 1e6;
                    if (!bg && std::abs(b - a) / std::max(a, 1e-6) > 0.01)
                    {
                        depthMask[p] = 1;
                        ++depthMovedCount;
                    }
                    else if (!bg)
                    {
                        depthMask[p] = 0;
                    }
                }
            }
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
        }

        // Self-calibrating: the strongest tenth of the change is the geometry that
        // actually moved, the weakest half is background. An absolute threshold
        // marks most of the frame instead, because converged references still
        // carry a little noise and the moving character relights everything
        // around it.
        std::vector<uint8_t> moving;
        size_t movingCount = 0;
        if (truthA.valid() && truthB.valid() && truthA.px.size() == truthB.px.size())
        {
            const size_t pixels = truthA.px.size() / 4;
            std::vector<double> change(pixels, 0.0);
            for (size_t i = 0, p = 0; i < truthA.px.size(); i += 4, ++p)
            {
                for (int k = 0; k < 3; ++k)
                    change[p] += std::abs((double)truthA.px[i + k] - (double)truthB.px[i + k]);
            }
            std::vector<double> sorted = change;
            std::sort(sorted.begin(), sorted.end());
            const double hi = sorted[(size_t)(pixels * 0.90)];
            const double lo = sorted[(size_t)(pixels * 0.50)];
            moving.assign(pixels, 2); // 2 = neither, excluded from both measures
            if (depthMask.size() == pixels)
            {
                moving = depthMask;
                movingCount = depthMovedCount;
            }
            else
            {
                for (size_t p = 0; p < pixels; ++p)
                {
                    if (change[p] >= hi)
                    {
                        moving[p] = 1;
                        ++movingCount;
                    }
                    else if (change[p] <= lo)
                    {
                        moving[p] = 0;
                    }
                }
            }
        }

        // The user's report is that the skinned mesh stays noisy even when nothing
        // is animating, so the headline case is a *held* time, not a moving one.
        // With the shutter open the geometry is still sampled at a random instant
        // every frame, which a 1-spp temporal reconstruction has no way to
        // reproject -- accumulation resolves it, the denoiser cannot.
        const bool holdTime = getenv("STRELKA_AUDIT_HOLD") != nullptr;
        AuditImage last;
        const double swim = holdTime ? reconstruct("held", 0.0f, 0.0f, tB, tB, last)
                                     : reconstruct("moving", 0.0f, 0.0f, tA, tB, last);

        auto maskedRmse = [&](const AuditImage& img, bool wantMoving) {
            if (!img.valid() || !truthB.valid() || img.px.size() != truthB.px.size() || moving.empty())
                return -1.0;
            double se = 0.0;
            size_t n = 0;
            for (size_t i = 0, p = 0; i < img.px.size(); i += 4, ++p)
            {
                if (moving[p] != (wantMoving ? 1 : 0))
                    continue;
                for (int k = 0; k < 3; ++k)
                {
                    const double d = (double)img.px[i + k] - (double)truthB.px[i + k];
                    se += d * d;
                    ++n;
                }
            }
            return n ? std::sqrt(se / (double)n) : -1.0;
        };
        // What the denoiser is actually told about those pixels. Guides are read
        // at 1:1 so the mask, which came from display-resolution references, maps
        // straight onto them.
        {
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", 1.0f);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            setDenoise(true);
            m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
            m_render->resetTemporalHistory();
            for (int i = 0; i < 4 && !outOfTime(); ++i)
                step();

            auto guideStats = [&](const char* name, Render::Guide g, int channels) {
                AuditImage img;
                if (!guide(g, img) || img.px.size() / 4 != moving.size())
                {
                    report(fmt::format("AUDIT guide-on-mesh {:14s} unavailable or size mismatch", name));
                    return;
                }
                double sumMesh = 0.0, sumRest = 0.0, magMesh = 0.0, magRest = 0.0;
                size_t nMesh = 0, nRest = 0;
                for (size_t p = 0; p < moving.size(); ++p)
                {
                    double len = 0.0, mag = 0.0;
                    for (int k = 0; k < channels; ++k)
                    {
                        const double v = img.px[p * 4 + k];
                        len += v * v;
                        mag += std::abs(v);
                    }
                    len = std::sqrt(len);
                    if (moving[p] == 1) { sumMesh += len; magMesh += mag; ++nMesh; }
                    else if (moving[p] == 0) { sumRest += len; magRest += mag; ++nRest; }
                }
                report(fmt::format("AUDIT guide-on-mesh {:14s} mesh len={:.4f} sum|v|={:.4f} | rest len={:.4f} "
                                   "sum|v|={:.4f}",
                                   name, nMesh ? sumMesh / nMesh : -1.0, nMesh ? magMesh / nMesh : -1.0,
                                   nRest ? sumRest / nRest : -1.0, nRest ? magRest / nRest : -1.0));
            };
            guideStats("normal", Render::Guide::Normal, 3);
            guideStats("diffuseAlbedo", Render::Guide::DiffuseAlbedo, 3);
            guideStats("specAlbedo", Render::Guide::SpecularAlbedo, 3);
            guideStats("roughness", Render::Guide::Roughness, 1);
            guideStats("depth", Render::Guide::Depth, 1);
            guideStats("motion", Render::Guide::Motion, 2);
            m_settingsManager->setAs<float>("render/pt/upscaleFactor", upscale);
        }

        const double rmseMoving = maskedRmse(last, true);
        const double rmseStatic = maskedRmse(last, false);

        // The denoiser's input, for scale. "Denoising does not work here" means
        // the output is no better than the noisy frame it was given, and without
        // this number the errors above have nothing to be compared against.
        setDenoise(false);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        setAnimTime(tB);
        m_render->resetTemporalHistory();
        step();
        AuditImage raw;
        shown(raw);
        const double rawMoving = maskedRmse(raw, true);
        const double rawStatic = maskedRmse(raw, false);
        report(fmt::format("AUDIT moving    motionBlur={} timeHeld={} {:.1f}% of pixels move   swim={:.5f}",
                           (int)mbOn, (int)holdTime,
                           moving.empty() ? 0.0 : 100.0 * (double)movingCount / (double)moving.size(), swim));
        report(fmt::format("AUDIT moving    on MOVING pixels: raw={:.5f} denoised={:.5f}  improvement x{:.2f}",
                           rawMoving, rmseMoving, rmseMoving > 0.0 ? rawMoving / rmseMoving : -1.0));
        report(fmt::format("AUDIT moving    on STATIC pixels: raw={:.5f} denoised={:.5f}  improvement x{:.2f}",
                           rawStatic, rmseStatic, rmseStatic > 0.0 ? rawStatic / rmseStatic : -1.0));
        // Denoising that leaves moving geometry no better than its own noisy
        // input is not denoising it at all.
        if (rawMoving > 0.0 && rmseMoving > 0.0 && rawMoving / rmseMoving < 1.2)
            report("AUDIT FAIL denoiser gives no improvement on moving geometry");
    }

    // === Pause mid-animation =================================================
    //
    // Pausing should freeze a frame of the film and go on refining it, and a
    // frame of the film has motion blur in it. The motion structures were tied to
    // playback rather than to whether the shutter actually spans two poses, so a
    // few frames after the pause the blur was dropped and a crisp still appeared.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        setDenoise(true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
        m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", true);
        m_settingsManager->setAs<bool>("render/enableMotionBlur", true);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", true);
        m_render->resetTemporalHistory();

        char key[64];
        for (size_t i = 0; i < animCount; ++i)
        {
            snprintf(key, sizeof(key), "render/animation/anim%zu/state", i);
            m_settingsManager->setAs<bool>(key, true);
        }
        for (int i = 0; i < 20 && !outOfTime(); ++i)
        {
            playAnimations(1.0f / 60.0f);
            if (!step())
                break;
        }
        const bool motionWhilePlaying = m_render->motionGeometryActive();

        // Pause: stop advancing time, keep rendering.
        for (size_t i = 0; i < animCount; ++i)
        {
            snprintf(key, sizeof(key), "render/animation/anim%zu/state", i);
            m_settingsManager->setAs<bool>(key, false);
        }
        const size_t subframeAtPause = m_sharedCtx->mSubframeIndex;
        bool motionHeld = true;
        AuditImage pausedDepthFirst;
        for (int i = 0; i < 30 && !outOfTime(); ++i)
        {
            if (!step())
                break;
            if (i == 0)
                guide(Render::Guide::Depth, pausedDepthFirst);
            motionHeld = motionHeld && m_render->motionGeometryActive();
        }
        const size_t subframeAfter = m_sharedCtx->mSubframeIndex;
        AuditImage pausedDepthLast;
        guide(Render::Guide::Depth, pausedDepthLast);
        const double pausedGuideDelta =
            auditSwim(pausedDepthFirst, pausedDepthLast);

        report(fmt::format("AUDIT pause     motion geometry while playing={} held across pause={}  "
                           "accumulation {} -> {} guideDelta={:.7f}",
                           (int)motionWhilePlaying, (int)motionHeld, subframeAtPause,
                           subframeAfter, pausedGuideDelta));
        if (motionWhilePlaying && !motionHeld)
            report("AUDIT FAIL pause dropped motion blur instead of refining the blurred frame");
        if (subframeAfter <= subframeAtPause)
            report("AUDIT FAIL pause stopped the estimator converging");
        if (pausedGuideDelta > 1e-6)
            report("AUDIT FAIL paused canonical depth guide is not stable");

        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
        m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", false);
    }

    // Frame-to-frame instability with nothing touched at all: no camera, no time,
    // no settings. Deliberately does not reset the animation clock, so whatever
    // state playback left behind -- including two pose keyframes that differ -- is
    // what gets measured.
    auto reconstructHeld = [&](const char* label, AuditImage& last) {
        m_render->resetTemporalHistory();
        for (int w = 0; w < 8 && !outOfTime(); ++w)
        {
            if (!step())
                break;
        }
        AuditImage prev;
        double swimSum = 0.0;
        uint32_t swimCount = 0;
        for (uint32_t f = 0; f < frames && !outOfTime(); ++f)
        {
            if (!step())
                break;
            AuditImage cur;
            if (!shown(cur))
                continue;
            if (prev.valid())
            {
                const double sw = auditSwim(prev, cur);
                if (sw >= 0.0)
                {
                    swimSum += sw;
                    ++swimCount;
                }
            }
            prev = std::move(cur);
        }
        last = prev;
        (void)label;
        return swimCount ? swimSum / swimCount : -1.0;
    };

    // === Shutter versus denoiser ============================================
    //
    // A stochastic shutter puts the geometry at a different instant every frame,
    // which a one-sample temporal reconstruction cannot follow. The symptom is a
    // still frame that shakes, and it is invisible to error-against-a-reference:
    // the shake averages out, so RMSE barely moves. Frame-to-frame instability is
    // the measure, and the test is that turning the shutter on does not change it.
    if (animCount > 0)
    {
        setOrbit(0.0f);
        setAnimTime(0.12f);
        double swimOff = -1.0, swimOn = -1.0;
        char akey[64];
        for (int mb = 0; mb < 2 && !outOfTime(); ++mb)
        {
            m_settingsManager->setAs<bool>("render/enableMotionBlur", mb != 0);
            m_settingsManager->setAs<bool>("render/isMotionBlurVisible", mb != 0);
            m_settingsManager->setAs<bool>(
                "render/pt/denoisePlaybackMotionBlur", mb != 0);
            // Play first, then hold. Holding from the start leaves both pose
            // keyframes identical, so the shutter has nothing to smear between and
            // the very defect being tested cannot appear -- which is exactly the
            // difference between "no shake at scene start" and "shake after
            // playing".
            setDenoise(true);
            for (size_t a = 0; a < animCount; ++a)
            {
                snprintf(akey, sizeof(akey), "render/animation/anim%zu/state", a);
                m_settingsManager->setAs<bool>(akey, true);
            }
            for (int i = 0; i < 12 && !outOfTime(); ++i)
            {
                playAnimations(1.0f / 60.0f);
                if (!step())
                    break;
            }
            for (size_t a = 0; a < animCount; ++a)
            {
                snprintf(akey, sizeof(akey), "render/animation/anim%zu/state", a);
                m_settingsManager->setAs<bool>(akey, false);
            }
            AuditImage last;
            const double swim = reconstructHeld(mb ? "shutter on" : "shutter off", last);
            (mb ? swimOn : swimOff) = swim;
        }
        m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
        m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
        m_settingsManager->setAs<bool>("render/pt/denoisePlaybackMotionBlur", false);
        report(fmt::format("AUDIT shutter   held frame instability: shutter off={:.5f} on={:.5f} ratio={:.1f}",
                           swimOff, swimOn, swimOff > 0.0 ? swimOn / swimOff : -1.0));
        if (swimOff > 0.0 && swimOn > swimOff * 3.0)
            report("AUDIT FAIL the shutter is shaking the denoised frame");
    }

    // === Denoising past the sample limit ====================================
    //
    // The sample cap stops the estimator. It must not stop the denoiser: a
    // temporal filter that is no longer fed has nothing to show, and with a
    // reduced-resolution render there is not even a display-sized image to fall
    // back on -- the screen simply empties.
    {
        setOrbit(0.0f);
        setAnimTime(0.0f);
        setDenoise(true);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", 4);
        m_render->resetTemporalHistory();
        // Every frame, not one of them. The display is double buffered, so a path
        // that stops writing updates only one of the pair: the picture then
        // alternates between the last good frame and whatever the other buffer
        // holds. A single sample lands on the good one half the time and reports
        // nothing wrong.
        double worstMean = 1e9;
        for (int i = 0; i < 24 && !outOfTime(); ++i)
        {
            if (!step())
                break;
            if (i < 8)
                continue; // let it cross the cap first
            AuditImage img;
            if (shown(img) && img.valid())
                worstMean = std::min(worstMean, auditMean(img));
        }
        const double meanAfterLimit = worstMean >= 1e9 ? -1.0 : worstMean;
        report(fmt::format("AUDIT sppcap    denoised image past a 4-sample cap: dimmest of 16 frames "
                           "mean={:.4f}",
                           meanAfterLimit));
        if (meanAfterLimit >= 0.0 && meanAfterLimit <= 1e-3)
            report("AUDIT FAIL denoised image vanished once the sample cap was reached");
        m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", refSpp);
    }

    // === S5 mode stress =====================================================
    // Every switch a user can reach, in the combinations that share state: a
    // scaler bound to one resolution, guide textures sized for another, a tracer
    // that writes no guides at all.
    setOrbit(0.0f);
    setAnimTime(0.0f);
    const uint32_t baseW = auditW;
    const uint32_t baseH = auditH;
    struct Mode
    {
        const char* name;
        bool denoise, upscale;
        float factor;
        uint32_t tracer, w, h;
    };
    const Mode modes[] = {
        { "denoise 0.50", true, true, 0.50f, 1, baseW, baseH },
        { "denoise 0.25", true, true, 0.25f, 1, baseW, baseH },
        { "denoise 0.75", true, true, 0.75f, 1, baseW, baseH },
        { "denoise 1.00", true, false, 1.00f, 1, baseW, baseH },
        // Odd sizes: the render resolution rounds to the same value while the
        // output does not, which is the case a guide-texture cache keyed on the
        // render size alone gets wrong.
        { "denoise odd+1", true, true, 0.50f, 1, baseW + 1, baseH + 1 },
        { "denoise odd+3", true, true, 0.50f, 1, baseW + 3, baseH + 3 },
        { "upscale only", false, true, 0.50f, 1, baseW, baseH },
        { "plain wavefront", false, false, 1.00f, 1, baseW, baseH },
        { "megakernel plain", false, false, 1.00f, 0, baseW, baseH },
        { "megakernel upscale", false, true, 0.50f, 0, baseW, baseH },
        { "denoise on megakernel", true, true, 0.50f, 0, baseW, baseH },
    };
    AuditImage prevModeImage;
    std::string prevModeKey;
    for (const Mode& m : modes)
    {
        if (outOfTime())
        {
            report(fmt::format("AUDIT WARN out of budget, skipping mode {} onwards", m.name));
            break;
        }
        // Named before it is entered: if a combination aborts the process, the
        // last line printed is the one that names the combination that did it.
        report(fmt::format("AUDIT mode  {:22s} entering ({}x{} tracer={} denoise={} upscale={} factor={:.2f})",
                           m.name, m.w, m.h, m.tracer, (int)m.denoise, (int)m.upscale, m.factor));
        m_settingsManager->setAs<uint32_t>("render/width", m.w);
        m_settingsManager->setAs<uint32_t>("render/height", m.h);
        m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", m.tracer);
        m_settingsManager->setAs<bool>("render/pt/denoise", m.denoise);
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", m.upscale);
        m_settingsManager->setAs<float>("render/pt/upscaleFactor", m.factor);
        m_settingsManager->setAs<bool>("render/pt/enableAcc", false);
        m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
        m_render->resetTemporalHistory();

        bool stepped = true;
        for (int i = 0; i < 4 && stepped; ++i)
            stepped = step();

        AuditImage img;
        const bool got = shown(img) && img.valid();
        const double mean = got ? auditMean(img) : 0.0;
        const bool finite = got && auditFinite(img);
        // Identical to the previous mode's picture means this mode wrote nothing
        // and the display is still showing the last one. That reads as a pass on
        // every other check, and it is exactly what a path that forgets to encode
        // its scaler looks like.
        // Compared only against a mode that should look different. Denoising is
        // ignored on the megakernel, so "megakernel + denoise" and "megakernel +
        // upscale" are the same configuration and produce the same pixels; that
        // is correct behaviour, not a mode that wrote nothing.
        const std::string modeKey =
            fmt::format("{}|{}|{:.2f}|{}x{}", m.tracer, (int)(m.denoise && m.tracer == 1),
                        m.upscale ? m.factor : 1.0f, m.w, m.h);
        const bool stale = got && prevModeImage.valid() && modeKey != prevModeKey &&
                           prevModeImage.px.size() == img.px.size() &&
                           std::memcmp(prevModeImage.px.data(), img.px.data(),
                                       img.px.size() * sizeof(float)) == 0;
        prevModeKey = modeKey;
        const char* verdict = !stepped     ? "NO FRAME"
                              : !got       ? "NO IMAGE"
                              : !finite    ? "NON-FINITE"
                              : mean <= 1e-6 ? "BLACK"
                              : stale      ? "STALE (mode wrote nothing)"
                                           : "ok";
        if (got)
            prevModeImage = img;
        if (std::strcmp(verdict, "ok") == 0)
        {
            report(fmt::format("AUDIT mode  {:22s} {}x{} mean={:.4f}  ok", m.name, img.w, img.h, mean));
        }
        else
        {
            report(fmt::format("AUDIT FAIL mode {:22s} {}", m.name, verdict));
        }
    }
    m_settingsManager->setAs<uint32_t>("render/width", baseW);
    m_settingsManager->setAs<uint32_t>("render/height", baseH);

    report(fmt::format("AUDIT done in {:.0f}s", elapsed()));
    m_display->requestClose();
}

// Reference capture / estimator self-consistency check (STRELKA_REF=<dir>).
void EditorApp::runReferenceCapture()
{
    const char* outDir = getenv("STRELKA_REF");
    const uint32_t spp = (uint32_t)atoi(getenv("STRELKA_REF_SPP") ? getenv("STRELKA_REF_SPP") : "512");

    struct C { const char* name; uint32_t estimator; bool analyticLights; };
    const C cases[] = {
        { "nee",           0, true  },
        { "bsdf_only",     1, true  },
        { "nee_envonly",   0, false },
        { "bsdf_envonly",  1, false },
    };

    // Linear output: the tone curve is irrelevant for comparing estimators and
    // would compress exactly the differences we are looking for.
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0);
    m_settingsManager->setAs<float>("render/post/gamma", 0.0f);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", spp);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", 1);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    // Full resolution unless asked otherwise: an upscaled capture is not what the
    // estimators are being compared at, and the buffer would hold a smaller image
    // than the EXR claims. The display EXR alongside it is the upscaled one.
    if (!getenv("STRELKA_UPSCALE"))
    {
        m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    }
    // Same harness for both tracers, so the wavefront rewrite can be checked
    // against the megakernel's recorded numbers without touching anything else.
    if (const char* d = getenv("STRELKA_REF_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", (uint32_t)atoi(d));
    }
    if (const char* tracer = getenv("STRELKA_TRACER"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/tracerMode", (uint32_t)atoi(tracer));
    }

    std::vector<std::vector<float>> images;
    for (const C& c : cases)
    {
        m_settingsManager->setAs<uint32_t>("render/validate/estimatorMode", c.estimator);
        m_settingsManager->setAs<bool>("render/validate/analyticLights", c.analyticLights);
        m_sharedCtx->mSubframeIndex = 0;

        while (m_sharedCtx->mSubframeIndex < spp && !m_display->windowShouldClose())
        {
            m_display->pollEvents();
            m_render->triggerRenderIfIdle();
            usleep(300);
        }
        // Let the last submission land.
        for (int i = 0; i < 2000 && m_render->getReadyBuffer() == nullptr; ++i) usleep(500);
        usleep(200000);

        oka::Buffer* rb = m_render->getReadyBuffer();
        std::vector<float> img;
        double meanLum = 0.0;
        if (rb)
        {
            const float* px = static_cast<const float*>(rb->getHostPointer());
            const size_t n = (size_t)rb->width() * rb->height() * 4;
            img.assign(px, px + n);
            for (size_t i = 0; i < n; i += 4)
                meanLum += 0.2126*px[i] + 0.7152*px[i+1] + 0.0722*px[i+2];
            meanLum /= (double)(n / 4);
            if (outDir)
            {
                saveScreenshot(rb, std::string(outDir) + "/" + c.name + ".exr");
            }
            // ...and what the screen actually shows, which after MetalFX is a
            // different image at a different resolution.
            std::vector<float> shown;
            uint32_t sw = 0, sh = 0;
            if (outDir && m_render->readDisplayTexture(shown, sw, sh))
            {
                const char* err = nullptr;
                SaveEXR(shown.data(), (int)sw, (int)sh, 4, 0,
                        (std::string(outDir) + "/" + c.name + "_display.exr").c_str(), &err);
            }
        }
        images.push_back(std::move(img));
        STRELKA_INFO("REF  {:14s} spp={} meanLum={:.6f}", c.name, (uint32_t)m_sharedCtx->mSubframeIndex, meanLum);
    }

    auto compare = [&](const char* label, size_t a, size_t b) {
        if (images[a].empty() || images[b].empty() || images[a].size() != images[b].size()) return;
        double se = 0.0, refEnergy = 0.0; size_t n = 0;
        double lumA = 0.0, lumB = 0.0;
        for (size_t i = 0; i < images[a].size(); i += 4)
        {
            for (int k = 0; k < 3; ++k)
            {
                const double d = images[a][i+k] - images[b][i+k];
                se += d * d;
                refEnergy += (double)images[a][i+k] * images[a][i+k];
            }
            lumA += 0.2126*images[a][i] + 0.7152*images[a][i+1] + 0.0722*images[a][i+2];
            lumB += 0.2126*images[b][i] + 0.7152*images[b][i+1] + 0.0722*images[b][i+2];
            ++n;
        }
        const double rmse = sqrt(se / (double)(n * 3));
        const double rel = refEnergy > 0 ? sqrt(se / refEnergy) : 0.0;
        STRELKA_INFO("REF  {:28s} RMSE={:.6f}  relative={:.3f}%  meanLum {:.6f} vs {:.6f}  bias={:+.2f}%",
                     label, rmse, 100.0*rel, lumA/n, lumB/n, 100.0*(lumB/lumA - 1.0));
    };
    compare("NEE vs BSDF-only (all)", 0, 1);
    compare("NEE vs BSDF-only (env only)", 2, 3);

    STRELKA_INFO("REF done");
    m_display->requestClose();
}

// ---------------------------------------------------------------------------
// Convergence sweep (STRELKA_CONV=<dir|1>).
//
// The question is whether the estimator's error actually falls as 1/sqrt(N).
// Eyeballing two renders cannot answer it: the difference between 256 and 1024
// samples is one stop of noise, which is easy to miss on a tonemapped image and
// impossible to miss in a number.
//
// Two metrics, because each has a blind spot:
//
//   * half-split noise. Snapshots at N and 2N give the mean of the first N
//     samples and of all 2N, so the second half's mean is 2*I(2N) - I(N). The
//     two halves are independent estimates of the same integral, so the RMS of
//     their difference over sqrt(2) is the error of an N-sample estimate --
//     with no reference image and therefore no reference bias. It does assume
//     the halves are independent, which for a QMC sequence they are not, so it
//     slightly *over*states QMC error. That is the safe direction.
//
//   * RMSE against the deepest snapshot. Sees any error, including the
//     correlated kind a half-split misses, but its own noise floor is the
//     reference's, so it flattens out near the end of the sweep by
//     construction. Only the first half of the table is meaningful.
//
// A healthy sampler halves each metric per 4x samples (slope 0.5 per doubling
// on a log2 scale). A sampler that has stopped bringing new information shows a
// slope heading to 0.
// ---------------------------------------------------------------------------
void EditorApp::runConvergenceSweep()
{
    const char* outDir = getenv("STRELKA_CONV");
    const bool saveImages = outDir && strchr(outDir, '/') != nullptr;
    const uint32_t maxSpp = (uint32_t)atoi(getenv("STRELKA_CONV_MAX") ? getenv("STRELKA_CONV_MAX") : "1024");
    const uint32_t firstSpp = (uint32_t)atoi(getenv("STRELKA_CONV_MIN") ? getenv("STRELKA_CONV_MIN") : "16");
    // Small on purpose. Every metric here is an average over pixels, so a
    // quarter-size image gives the same answer with four times less waiting --
    // and the whole sweep has to fit inside one run.
    const uint32_t convW = (uint32_t)atoi(getenv("STRELKA_CONV_W") ? getenv("STRELKA_CONV_W") : "320");
    const uint32_t convH = (uint32_t)atoi(getenv("STRELKA_CONV_H") ? getenv("STRELKA_CONV_H") : "240");
    // Samples per launch. Bigger is faster (fewer command buffers for the same
    // sample count) and must divide the checkpoints, so it is a power of two.
    const uint32_t sppPerLaunch =
        (uint32_t)atoi(getenv("STRELKA_CONV_STEP") ? getenv("STRELKA_CONV_STEP") : "8");
    const double budgetSec =
        atof(getenv("STRELKA_CONV_BUDGET") ? getenv("STRELKA_CONV_BUDGET") : "600");
    const char* samplersEnv =
        getenv("STRELKA_CONV_SAMPLERS") ? getenv("STRELKA_CONV_SAMPLERS") : "0,1,2";

    const auto startTime = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
    };
    auto outOfTime = [&]() { return elapsed() > budgetSec || m_display->windowShouldClose(); };
    auto report = [&](const std::string& line) {
        STRELKA_INFO("{}", line);
        std::fputs(line.c_str(), stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    };

    // Linear output and no reconstruction: a tone curve compresses exactly the
    // bright noise this is measuring, and a temporal upscaler would be the thing
    // under test instead of the estimator.
    m_settingsManager->setAs<uint32_t>("render/pt/tracerMode",
                                       getenv("STRELKA_TRACER") ? m_settingsManager->getAs<uint32_t>("render/pt/tracerMode") : 1);
    m_settingsManager->setAs<uint32_t>("render/pt/splitSubmissions", 0);
    m_settingsManager->setAs<uint32_t>("render/pt/tonemapperType", 0);
    m_settingsManager->setAs<float>("render/post/gamma", 0.0f);
    m_settingsManager->setAs<bool>("render/pt/denoise", false);
    m_settingsManager->setAs<bool>("render/pt/enableUpscale", false);
    m_settingsManager->setAs<bool>("render/enableMotionBlur", false);
    m_settingsManager->setAs<bool>("render/isMotionBlurVisible", false);
    m_settingsManager->setAs<bool>("render/pt/enableAcc", true);
    m_settingsManager->setAs<uint32_t>("render/pt/spp", sppPerLaunch);
    m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", maxSpp);
    m_settingsManager->setAs<uint32_t>("render/width", convW);
    m_settingsManager->setAs<uint32_t>("render/height", convH);
    if (const char* d = getenv("STRELKA_CONV_DEPTH"))
    {
        m_settingsManager->setAs<uint32_t>("render/pt/depth", (uint32_t)atoi(d));
    }

    std::vector<uint32_t> checkpoints;
    for (uint32_t n = firstSpp; n <= maxSpp; n *= 2)
    {
        checkpoints.push_back(n);
    }

    auto step = [&]() -> bool {
        const size_t target = m_sharedCtx->mFrameNumber + 1;
        bool submitted = false;
        while (!m_display->windowShouldClose())
        {
            m_display->pollEvents();
            if (!submitted)
            {
                m_render->triggerRenderIfIdle();
                submitted = m_sharedCtx->mFrameNumber >= target;
            }
            if (submitted && !m_render->isRenderBusy())
                return true;
            if (outOfTime())
                return false;
            usleep(200);
        }
        return false;
    };

    // The tonemap of an accumulation lands one frame after the samples do, so
    // every snapshot is taken after an extra pumped frame.
    auto snapshot = [&](std::vector<float>& out) -> bool {
        step();
        oka::Buffer* rb = m_render->getReadyBuffer();
        if (!rb)
            return false;
        const float* px = static_cast<const float*>(rb->getHostPointer());
        const size_t n = (size_t)rb->width() * rb->height() * 4;
        if (n == 0)
            return false;
        out.assign(px, px + n);
        return true;
    };

    // RMS over the three colour channels, and the same relative to the mean
    // level of `scale` -- an absolute RMSE says nothing without knowing how
    // bright the picture is.
    auto rms = [](const std::vector<float>& a, const std::vector<float>& b, double weight) {
        double se = 0.0;
        size_t n = 0;
        for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4)
        {
            for (int k = 0; k < 3; ++k)
            {
                const double d = ((double)a[i + k] - (double)b[i + k]) * weight;
                se += d * d;
                ++n;
            }
        }
        return n ? std::sqrt(se / (double)n) : -1.0;
    };
    auto mean = [](const std::vector<float>& a) {
        double s = 0.0;
        size_t n = 0;
        for (size_t i = 0; i + 3 < a.size(); i += 4)
        {
            s += 0.2126 * a[i] + 0.7152 * a[i + 1] + 0.0722 * a[i + 2];
            ++n;
        }
        return n ? s / (double)n : 0.0;
    };
    // A mean luminance is dominated by whichever pixels happen to hold the
    // brightest fireflies, so two samplers can disagree on it by percent while
    // agreeing everywhere a person would look. The median says which of those
    // two situations this is.
    auto median = [](const std::vector<float>& a) {
        std::vector<double> l;
        l.reserve(a.size() / 4);
        for (size_t i = 0; i + 3 < a.size(); i += 4)
            l.push_back(0.2126 * a[i] + 0.7152 * a[i + 1] + 0.0722 * a[i + 2]);
        if (l.empty())
            return 0.0;
        std::nth_element(l.begin(), l.begin() + l.size() / 2, l.end());
        return l[l.size() / 2];
    };

    // Error left after a small blur. A raw RMSE counts every frequency the same,
    // which is exactly the assumption a blue-noise sampler is built to violate:
    // it does not remove error, it moves error to high frequencies, where the
    // eye and any reconstruction filter throw it away. Two samplers with the
    // same RMSE and different spectra look very different, and only this number
    // says so.
    auto lowPassRms = [&](const std::vector<float>& a, const std::vector<float>& b, uint32_t w,
                          uint32_t h, double weight) {
        if (a.size() != b.size() || (size_t)w * h * 4 != a.size())
            return -1.0;
        // 5-tap binomial, separable, applied to the error field.
        static const double k[5] = { 1.0 / 16, 4.0 / 16, 6.0 / 16, 4.0 / 16, 1.0 / 16 };
        std::vector<double> err((size_t)w * h * 3), tmp((size_t)w * h * 3);
        for (size_t i = 0, j = 0; i + 3 < a.size(); i += 4, j += 3)
            for (int c = 0; c < 3; ++c)
                err[j + c] = ((double)a[i + c] - (double)b[i + c]) * weight;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    double s = 0.0;
                    for (int t = -2; t <= 2; ++t)
                    {
                        const uint32_t xx = (uint32_t)std::clamp((int)x + t, 0, (int)w - 1);
                        s += k[t + 2] * err[((size_t)y * w + xx) * 3 + c];
                    }
                    tmp[((size_t)y * w + x) * 3 + c] = s;
                }
        double se = 0.0;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    double s = 0.0;
                    for (int t = -2; t <= 2; ++t)
                    {
                        const uint32_t yy = (uint32_t)std::clamp((int)y + t, 0, (int)h - 1);
                        s += k[t + 2] * tmp[((size_t)yy * w + x) * 3 + c];
                    }
                    se += s * s;
                }
        return std::sqrt(se / (double)((size_t)w * h * 3));
    };

    static const char* kSamplerNames[] = { "Halton", "PCG", "Sobol", "SobolBN", "Hybrid" };
    // Deepest snapshot per sampler, kept for the cross-check at the end.
    std::vector<float> deepest[5];
    uint32_t deepestSpp[5] = { 0, 0, 0, 0, 0 };

    report(fmt::format("CONV {}x{} spp {}..{} step={} depth={} tracer={}", convW, convH, firstSpp, maxSpp,
                       sppPerLaunch, m_settingsManager->getAs<uint32_t>("render/pt/depth"),
                       m_settingsManager->getAs<uint32_t>("render/pt/tracerMode")));

    for (const char* p = samplersEnv; p && *p;)
    {
        const uint32_t samplerType = (uint32_t)atoi(p);
        while (*p && *p != ',')
            ++p;
        if (*p == ',')
            ++p;
        if (samplerType > 4 || outOfTime())
            continue;

        m_settingsManager->setAs<uint32_t>("render/pt/samplerType", samplerType);
        m_sharedCtx->mSubframeIndex = 0;
        m_render->resetTemporalHistory();

        std::vector<std::vector<float>> snaps(checkpoints.size());
        const auto samplerStart = std::chrono::steady_clock::now();
        for (size_t c = 0; c < checkpoints.size(); ++c)
        {
            // The renderer refuses to launch once the total is reached, so the
            // cap has to rise ahead of each target rather than being set once.
            m_settingsManager->setAs<uint32_t>("render/pt/sppTotal", checkpoints[c]);
            uint32_t guard = checkpoints[c] / sppPerLaunch + 32;
            while (m_sharedCtx->mSubframeIndex < checkpoints[c] && guard-- > 0 && !outOfTime())
            {
                if (!step())
                    break;
            }
            if (!snapshot(snaps[c]))
            {
                report(fmt::format("CONV {} spp={} no frame", kSamplerNames[samplerType], checkpoints[c]));
                break;
            }
            if (m_sharedCtx->mSubframeIndex != checkpoints[c])
            {
                report(fmt::format("CONV WARN {} wanted spp={} got {}", kSamplerNames[samplerType],
                                   checkpoints[c], (uint32_t)m_sharedCtx->mSubframeIndex));
            }
            if (saveImages)
            {
                const char* err = nullptr;
                SaveEXR(snaps[c].data(), (int)convW, (int)convH, 4, 0,
                        fmt::format("{}/{}_{:05d}.exr", outDir, kSamplerNames[samplerType], checkpoints[c])
                            .c_str(),
                        &err);
            }
        }
        const double sweepSec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - samplerStart).count();

        // Everything is reported relative to the image's own mean level, so the
        // three samplers -- and different scenes -- are on one scale.
        size_t last = 0;
        while (last + 1 < snaps.size() && !snaps[last + 1].empty())
            ++last;
        if (snaps[last].empty())
            continue;
        const double lvl = mean(snaps[last]);
        const double w = lvl > 1e-9 ? 1.0 / lvl : 1.0;

        deepest[samplerType] = snaps[last];
        deepestSpp[samplerType] = checkpoints[last];

        report(fmt::format("CONV --- sampler={} ({}) mean={:.5f} median={:.5f} sweep={:.1f}s ---", samplerType,
                           kSamplerNames[samplerType], lvl, median(snaps[last]), sweepSec));
        report("CONV   spp   halfSplitNoise  slope    rmseVsDeepest  slope    lowPassErr   lp/raw");
        double prevNoise = -1.0, prevRef = -1.0;
        for (size_t c = 0; c < snaps.size(); ++c)
        {
            if (snaps[c].empty())
                break;
            double noise = -1.0;
            if (c + 1 < snaps.size() && !snaps[c + 1].empty())
            {
                // Second half's mean: 2*I(2N) - I(N).
                std::vector<float> secondHalf(snaps[c].size());
                for (size_t i = 0; i < secondHalf.size(); ++i)
                {
                    secondHalf[i] = 2.0f * snaps[c + 1][i] - snaps[c][i];
                }
                noise = rms(snaps[c], secondHalf, w) / std::sqrt(2.0);
            }
            const double ref = c < last ? rms(snaps[c], snaps[last], w) : -1.0;
            // Same error field as rmseVsDeepest, after a blur -- what survives a
            // reconstruction filter, which is what a person actually sees.
            const double lp = c < last ? lowPassRms(snaps[c], snaps[last], convW, convH, w) : -1.0;
            auto slope = [](double prev, double cur) {
                return (prev > 0.0 && cur > 0.0) ? std::log2(prev / cur) : 0.0;
            };
            report(fmt::format("CONV {:6d}   {:12.5f}  {:+5.2f}    {:12.5f}  {:+5.2f}    {:10.5f}  {:6.3f}",
                               checkpoints[c], noise, slope(prevNoise, noise), ref, slope(prevRef, ref), lp,
                               (ref > 0.0 && lp > 0.0) ? lp / ref : 0.0));
            prevNoise = noise;
            prevRef = ref;
        }
    }

    // Cross-check. The samplers estimate the same integral, so at the deepest
    // sample count they must agree to within their own noise. A disagreement
    // larger than that is bias in one of them, and no amount of extra samples
    // will remove it -- which is a different problem from converging slowly and
    // has to be told apart from it.
    for (uint32_t a = 0; a < 5; ++a)
    {
        for (uint32_t b = a + 1; b < 5; ++b)
        {
            if (deepest[a].empty() || deepest[b].empty() || deepest[a].size() != deepest[b].size())
                continue;
            const double lvl = mean(deepest[a]);
            const double w = lvl > 1e-9 ? 1.0 / lvl : 1.0;
            report(fmt::format("CONV agree {:6s} vs {:6s} @spp {}/{}: rmse={:.5f}  mean {:.4f}/{:.4f} "
                               "({:+.2f}%)  median {:.4f}/{:.4f} ({:+.2f}%)",
                               kSamplerNames[a], kSamplerNames[b], deepestSpp[a], deepestSpp[b],
                               rms(deepest[a], deepest[b], w), mean(deepest[a]), mean(deepest[b]),
                               100.0 * (mean(deepest[b]) / mean(deepest[a]) - 1.0), median(deepest[a]),
                               median(deepest[b]),
                               100.0 * (median(deepest[b]) / median(deepest[a]) - 1.0)));
        }
    }

    report("CONV done");
    m_display->requestClose();
}

void EditorApp::run()
{
    if (getenv("STRELKA_CONV")) { runConvergenceSweep(); return; }
    if (getenv("STRELKA_REF")) { runReferenceCapture(); return; }
    if (getenv("STRELKA_JITTER_TEST")) { runJitterTest(); return; }
    if (getenv("STRELKA_DENOISE_AUDIT")) { runDenoiseAudit(); return; }
    if (getenv("STRELKA_BENCH")) { runBenchmark(); return; }
    auto prevTime = std::chrono::high_resolution_clock::now();

    while (!m_display->windowShouldClose())
    {
        m_display->pollEvents();

        auto currentTime = std::chrono::high_resolution_clock::now();
        const double deltaTime = std::chrono::duration<double>(currentTime - prevTime).count();

        const auto cameraSpeed = m_settingsManager->getAs<float>("render/cameraSpeed");
        m_cameraController->update(deltaTime, cameraSpeed);
        prevTime = currentTime;

        playAnimations(deltaTime);

        // Consumed every frame so a gesture cannot be acted on twice.
        const bool userMovedCamera = m_cameraController->consumeUserMovedCamera();

        auto& selectedCam = m_scene->getCamera(m_selectedCamera);
        if (selectedCam.node != -1 && !m_cameraDetached)
        {
            // GLTF camera in animation mode
            if (userMovedCamera)
            {
                setCameraDetached(true);
            }
            else
            {
                auto& ctrlCam = m_cameraController->getCamera();
                ctrlCam.position = selectedCam.position;
                ctrlCam.mOrientation = selectedCam.mOrientation;
                ctrlCam.updateViewMatrix();
            }
        }

        if (selectedCam.node == -1 || m_cameraDetached)
        {
            auto& ctrlCam = m_cameraController->getCamera();
            selectedCam.position = ctrlCam.position;
            selectedCam.mOrientation = ctrlCam.mOrientation;
            selectedCam.matrices = ctrlCam.matrices;
            selectedCam.updated = ctrlCam.updated;
            selectedCam.isDirty = ctrlCam.isDirty;
        }

        // The renderer refreshes the projection of the camera it draws with, but
        // the UI reads the same camera earlier in the frame: picking and the gizmo
        // run before the first render has happened, and with an unset projection
        // both fail without a trace (an inf pick ray, a gizmo that never draws).
        const uint32_t renderWidth = m_settingsManager->getAs<uint32_t>("render/width");
        const uint32_t renderHeight = m_settingsManager->getAs<uint32_t>("render/height");
        if (renderHeight != 0)
        {
            const float aspect = (float)renderWidth / (float)renderHeight;
            selectedCam.updateAspectRatio(aspect);
            m_cameraController->getCamera().updateAspectRatio(aspect);
        }

        checkLoadingComplete();

        if (m_resized)
        {
            m_resized = false;
            m_sharedCtx->mSubframeIndex = 0;
        }

        // getMaxEDR() crosses into AppKit; the value only changes when the window
        // moves between displays, so poll it a few times a second instead of
        // every frame.
        if (std::chrono::duration<double>(currentTime - m_lastEdrQuery).count() > 0.25)
        {
            m_lastEdrQuery = currentTime;
            m_settingsManager->setAs<float>("render/post/tonemapper/maxEDR", m_display->getMaxEDR());
        }

        // Display: always runs at vsync, independent of render
        m_display->onBeginFrame();

        oka::Buffer* readyBuf = m_render->getReadyBuffer();
        if (readyBuf)
        {
            oka::ImageBuffer outputImage;
            outputImage.deviceData = readyBuf->getDevicePointer();
            outputImage.height = readyBuf->height();
            outputImage.width = readyBuf->width();
            outputImage.pixel_format = oka::BufferFormat::FLOAT4;
            // Metal hands over the tonemapped texture directly; the buffer is
            // still there and still linear, which is what a screenshot wants.
            outputImage.deviceTexture = m_render->getReadyTexture();
            outputImage.dataSize = readyBuf->width() * readyBuf->height() * readyBuf->getElementSize();
            m_display->drawFrame(outputImage);
        }

        drawUI();

        // Process pending screenshot save
        if (!m_pendingScreenshotPath.empty() && readyBuf)
        {
            saveScreenshot(readyBuf, m_pendingScreenshotPath);
            m_pendingScreenshotPath.clear();
        }

        m_display->drawUI();
        m_display->onEndFrame();

        // Enqueue the next render pass only after this frame's presentation work
        // has been committed. The renderer runs on its own command queue, but the
        // GPU still executes submissions roughly in arrival order — submitting a
        // multi-second path-trace batch first would push the compositor's work
        // behind it and stall nextDrawable() on the following frame.
        m_render->triggerRenderIfIdle();

        // Window titles go through AppKit; refreshing at vsync is pure overhead
        // and the numbers are unreadable at 60+ Hz anyway.
        if (std::chrono::duration<double>(currentTime - m_lastTitleUpdate).count() > 0.25)
        {
            m_lastTitleUpdate = currentTime;
            char title[128];
            snprintf(title, sizeof(title), "Strelka [render: %.1f ms] [%zu spp]",
                     m_render->getLastRenderTimeMs(), m_sharedCtx->mSubframeIndex);
            m_display->setWindowTitle(title);
        }
    }
}

void EditorApp::playAnimations(const float deltaTime)
{
    const float speed = m_settingsManager->getAs<float>("render/animation/speed");
    const auto& animations = m_scene->getAnimations();
    char key[64];
    for (int i = 0; i < (int)animations.size(); ++i)
    {
        snprintf(key, sizeof(key), "render/animation/anim%d/state", i);
        const bool currAnimEnable = m_settingsManager->getAs<bool>(key);

        if (currAnimEnable)
        {
            snprintf(key, sizeof(key), "render/animation/anim%d/time", i);
            float currAnimTime = m_settingsManager->getAs<float>(key);

            const float currAnimStart = animations[i].start;
            const float currAnimEnd = animations[i].end;

            currAnimTime += deltaTime * speed;
            if (currAnimTime > currAnimEnd) currAnimTime -= (currAnimEnd - currAnimStart);
            if (currAnimTime < currAnimStart) currAnimTime = currAnimStart;
            m_settingsManager->setAs<float>(key, currAnimTime);
        }
    }
}

void EditorApp::saveScreenshot(Buffer* buf, const std::string& path)
{
    const uint32_t w = buf->width();
    const uint32_t h = buf->height();
    const float* data = static_cast<const float*>(buf->getHostPointer());

    auto dotPos = path.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? path.substr(dotPos) : "";

    if (ext == ".exr")
    {
        const char* err = nullptr;
        int ret = SaveEXR(data, w, h, 4, 0, path.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to save EXR: {}", err ? err : "unknown");
            if (err)
                FreeEXRErrorMessage(err);
        }
        else
        {
            STRELKA_INFO("Screenshot saved: {}", path);
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
        int ret = stbi_write_png(path.c_str(), w, h, 4, pixels.data(), w * 4);
        if (!ret)
        {
            STRELKA_ERROR("Failed to save PNG: {}", path);
        }
        else
        {
            STRELKA_INFO("Screenshot saved: {}", path);
        }
    }
    else
    {
        STRELKA_ERROR("Unsupported screenshot format: {}", ext);
    }
}

void EditorApp::setCameraDetached(bool detached)
{
    m_cameraDetached = detached;

    // Exactly one camera can be under manual control, and clearing all of them
    // first means a camera left detached and then switched away from does not
    // stay frozen out of playback.
    for (uint32_t i = 0; i < (uint32_t)m_scene->getCameraCount(); ++i)
    {
        m_scene->getCamera(i).manualControl = false;
    }
    if (detached && m_selectedCamera >= 0 && m_selectedCamera < (int)m_scene->getCameraCount())
    {
        m_scene->getCamera(m_selectedCamera).manualControl = true;
    }
}

void EditorApp::clearSelection()
{
    m_selectedNodeId = (uint32_t)-1;
    m_selectedInstanceId = (uint32_t)-1;
    m_selectedLightId = (uint32_t)-1;
    m_selectedMaterialId = (uint32_t)-1;
}

void EditorApp::markDocumentDirty()
{
    m_documentDirty = true;
}

void EditorApp::pushUndoLight(uint32_t lightId)
{
    if (lightId >= m_scene->getLightsDesc().size())
        return;
    UndoState s;
    s.kind = UndoState::Kind::Light;
    s.id = lightId;
    s.light = m_scene->getLightsDesc()[lightId];
    m_undoStack.push_back(s);
    m_redoStack.clear();
}

void EditorApp::pushUndoNode(uint32_t nodeId)
{
    if (nodeId >= m_scene->getNodes().size())
        return;
    const auto& n = m_scene->getNodes()[nodeId];
    UndoState s;
    s.kind = UndoState::Kind::Node;
    s.id = nodeId;
    s.translation = n.translation;
    s.rotation = n.rotation;
    s.scale = n.scale;
    m_undoStack.push_back(s);
    m_redoStack.clear();
}

void EditorApp::pushUndoMaterial(uint32_t materialId)
{
    if (materialId >= m_scene->getMaterials().size())
        return;
    UndoState s;
    s.kind = UndoState::Kind::Material;
    s.id = materialId;
    s.material = m_scene->getMaterials()[materialId];
    m_undoStack.push_back(s);
    m_redoStack.clear();
}

void EditorApp::undo()
{
    if (m_undoStack.empty())
        return;
    UndoState cur = m_undoStack.back();
    m_undoStack.pop_back();
    UndoState redo = cur;
    if (cur.kind == UndoState::Kind::Light && cur.id < m_scene->getLightsDesc().size())
    {
        redo.light = m_scene->getLightsDesc()[cur.id];
        m_scene->setLight(cur.id, cur.light);
    }
    else if (cur.kind == UndoState::Kind::Node && cur.id < m_scene->getNodes().size())
    {
        const auto& n = m_scene->getNodes()[cur.id];
        redo.translation = n.translation;
        redo.rotation = n.rotation;
        redo.scale = n.scale;
        m_scene->setNodeLocalTransform(cur.id, cur.translation, cur.rotation, cur.scale);
    }
    else if (cur.kind == UndoState::Kind::Material && cur.id < m_scene->getMaterials().size())
    {
        redo.material = m_scene->getMaterials()[cur.id];
        m_scene->setMaterial(cur.id, cur.material);
    }
    m_redoStack.push_back(redo);
    markDocumentDirty();
}

void EditorApp::redo()
{
    if (m_redoStack.empty())
        return;
    UndoState cur = m_redoStack.back();
    m_redoStack.pop_back();
    if (cur.kind == UndoState::Kind::Light)
        m_scene->setLight(cur.id, cur.light);
    else if (cur.kind == UndoState::Kind::Node)
        m_scene->setNodeLocalTransform(cur.id, cur.translation, cur.rotation, cur.scale);
    else if (cur.kind == UndoState::Kind::Material)
        m_scene->setMaterial(cur.id, cur.material);
    markDocumentDirty();
}

void EditorApp::applySelectionFromPick(const Scene::PickHit& hit)
{
    clearSelection();
    if (!hit.hit)
        return;
    m_selectedInstanceId = hit.instanceId;
    m_selectedNodeId = hit.nodeId;
    m_selectedLightId = hit.lightId;
    m_outlinerScrollToSelection = true;
    if (hit.instanceId < m_scene->getInstances().size())
        m_selectedMaterialId = m_scene->getInstances()[hit.instanceId].mMaterialId;
}

bool EditorApp::saveDocument(bool saveAs)
{
    if (saveAs || m_sceneFile.empty())
    {
        IGFD::FileDialogConfig config;
        config.path = m_resourceSearchPath.empty() ? "." : m_resourceSearchPath;
        ImGuiFileDialog::Instance()->OpenDialog("SaveSceneDlgKey", "Save Scene As", ".gltf,.glb", config);
        m_pendingSaveAs = true;
        return true;
    }

    const bool okGltf = saveGltf(*m_scene, m_sceneFile);
    const bool okLights = saveLightsJson(*m_scene, m_sceneFile);
    if (okGltf && okLights)
    {
        m_documentDirty = false;
        STRELKA_INFO("Saved scene: {}", m_sceneFile);
        return true;
    }
    return false;
}

void EditorApp::buildDefaultDockLayout(ImGuiID dockspaceId)
{
    ImGui::DockBuilderRemoveNode(dockspaceId);
    ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspaceId, ImGui::GetMainViewport()->WorkSize);

    ImGuiID center = dockspaceId;
    const ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.20f, nullptr, &center);
    const ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f, nullptr, &center);
    ImGuiID leftTop = left;
    const ImGuiID leftBottom = ImGui::DockBuilderSplitNode(leftTop, ImGuiDir_Down, 0.35f, nullptr, &leftTop);
    ImGuiID rightTop = right;
    const ImGuiID rightBottom = ImGui::DockBuilderSplitNode(rightTop, ImGuiDir_Down, 0.45f, nullptr, &rightTop);

    ImGui::DockBuilderDockWindow("Outliner", leftTop);
    ImGui::DockBuilderDockWindow("Animations", leftBottom);
    ImGui::DockBuilderDockWindow("Viewport", center);
    ImGui::DockBuilderDockWindow("Render Settings:", rightTop);
    ImGui::DockBuilderDockWindow("Properties", rightBottom);
    ImGui::DockBuilderDockWindow("Materials", rightBottom);
    ImGui::DockBuilderFinish(dockspaceId);

    STRELKA_INFO("Editor layout rebuilt from defaults");
}

void EditorApp::drawUI()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::BeginFrame();

    m_cameraController->setGizmoBlocksInput(ImGuizmo::IsOver() || ImGuizmo::IsUsing());

    ImGuiIO& io = ImGui::GetIO();

    // Hotkeys
    if (!io.WantTextInput)
    {
        if (ImGui::IsKeyPressed(ImGuiKey_W) && m_selectedNodeId != (uint32_t)-1)
            m_gizmoOperation = ImGuizmo::TRANSLATE;
        if (ImGui::IsKeyPressed(ImGuiKey_E))
            m_gizmoOperation = ImGuizmo::ROTATE;
        if (ImGui::IsKeyPressed(ImGuiKey_R))
            m_gizmoOperation = ImGuizmo::SCALE;
        if (ImGui::IsKeyPressed(ImGuiKey_Escape))
            clearSelection();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S))
            saveDocument(io.KeyShift);
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z))
            undo();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Y))
            redo();
    }

    const ImGuiID dockspaceId = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    // A layout saved by an older build has no entry for panels added since, and
    // ImGui then floats them in the top-left corner on top of everything else.
    // Rebuild the default arrangement in that case instead of leaving the user to
    // hunt for the windows.
    if (m_layoutRebuildPending || ImGui::FindWindowSettingsByID(ImHashStr("Outliner")) == nullptr)
    {
        m_layoutRebuildPending = false;
        buildDefaultDockLayout(dockspaceId);
    }

    // --- Main menu bar ---
    ImGui::BeginMainMenuBar();
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Open File", nullptr, false, !m_isLoading))
        {
            IGFD::FileDialogConfig config;
            config.path = ".";
            ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".gltf,.glb", config);
        }
        if (ImGui::MenuItem("Save", "Ctrl+S", false, !m_isLoading && !m_sceneFile.empty()))
            saveDocument(false);
        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S", false, !m_isLoading))
            saveDocument(true);
        ImGui::Separator();
        if (ImGui::MenuItem("Exit"))
            m_display->requestClose();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Edit"))
    {
        if (ImGui::MenuItem("Undo", "Ctrl+Z", false, !m_undoStack.empty()))
            undo();
        if (ImGui::MenuItem("Redo", "Ctrl+Y", false, !m_redoStack.empty()))
            redo();
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Window"))
    {
        ImGui::MenuItem("Outliner", nullptr, &m_showOutliner);
        ImGui::MenuItem("Properties", nullptr, &m_showProperties);
        ImGui::MenuItem("Materials", nullptr, &m_showMaterials);
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Layout"))
        {
            m_layoutRebuildPending = true;
            m_showOutliner = m_showProperties = m_showMaterials = true;
        }
        ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();

    // --- File dialog handling ---
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            std::string sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string resourceSearchPath = ImGuiFileDialog::Instance()->GetCurrentPath();
            STRELKA_DEBUG("Resource search path {}", resourceSearchPath);
            m_settingsManager->setAs<std::string>("resource/searchPath", resourceSearchPath);
            m_pendingResourcePath = resourceSearchPath;
            m_sceneFile = sceneFile;

            auto loader = m_sceneLoader.get();
            m_loadingFuture = std::async(std::launch::async, [loader, sceneFile]() -> std::unique_ptr<Scene> {
                auto scene = std::make_unique<Scene>();
                if (loader->loadGltf(sceneFile, *scene))
                    return scene;
                return nullptr;
            });
            m_isLoading = true;
            clearSelection();
            m_documentDirty = false;
            m_undoStack.clear();
            m_redoStack.clear();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("SaveSceneDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            m_sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
            m_resourceSearchPath = ImGuiFileDialog::Instance()->GetCurrentPath();
            m_settingsManager->setAs<std::string>("resource/searchPath", m_resourceSearchPath);
            saveDocument(false);
        }
        ImGuiFileDialog::Instance()->Close();
        m_pendingSaveAs = false;
    }

    // --- Save screenshot dialog handling ---
    if (ImGuiFileDialog::Instance()->Display("SaveScreenshotDlgKey"))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            m_pendingScreenshotPath = ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (m_isLoading)
    {
        ImGui::Begin("##Loading", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Loading scene...");
        ImGui::End();
    }

    // --- Panel draw calls (implementations in panels/*.cpp) ---
    drawViewportPanel();
    drawRenderSettingsPanel();
    drawAnimationPanel();
    if (m_showOutliner)
        drawOutlinerPanel();
    if (m_showProperties)
        drawPropertyPanel();
    if (m_showMaterials)
        drawMaterialPanel();

    // Rendering
    ImGui::Render();
}

} // namespace oka
