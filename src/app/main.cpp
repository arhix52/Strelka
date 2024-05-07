#include <display/Display.h>

#include <render/common.h>
#include <render/buffer.h>
#include <render/Render.h>

#include <sceneloader/gltfloader.h>

#include <log.h>
#include <logmanager.h>
#include <cxxopts.hpp>
#include <filesystem>


class CameraController : public oka::InputHandler
{
    oka::Camera mCam;
    glm::quat mOrientation;
    glm::float3 mPosition;
    glm::float3 mWorldUp;
    glm::float3 mWorldForward;

    float rotationSpeed = 0.025f;
    float movementSpeed = 1.0f;

    double pitch = 0.0;
    double yaw = 0.0;
    double max_pitch_rate = 5;
    double max_yaw_rate = 5;

public:
    virtual ~CameraController() = default;
    struct
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
        bool forward = false;
        bool back = false;
    } keys;
    struct MouseButtons
    {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    glm::float2 mMousePos;

    // // local cameras axis
    // glm::float3 getFront() const
    // {
    //     return mOrientation.Transform(GfVec3d(0.0, 0.0, -1.0));
    // }
    // glm::float3 getUp() const
    // {
    //     return mOrientation.Transform(GfVec3d(0.0, 1.0, 0.0));
    // }
    // glm::float3 getRight() const
    // {
    //     return mOrientation.Transform(GfVec3d(1.0, 0.0, 0.0));
    // }
    // // global camera axis depending on scene settings
    // GfVec3d getWorldUp() const
    // {
    //     return mWorldUp;
    // }
    // GfVec3d getWorldForward() const
    // {
    //     return mWorldForward;
    // }

    bool moving()
    {
        return keys.left || keys.right || keys.up || keys.down || keys.forward || keys.back || mouseButtons.right ||
               mouseButtons.left || mouseButtons.middle;
    }
    void update(double deltaTime, float speed)
    {
        mCam.rotationSpeed = rotationSpeed;
        mCam.movementSpeed = speed;
        mCam.update(deltaTime);
        // movementSpeed = speed;
        // if (moving())
        // {
        //     const float moveSpeed = deltaTime * movementSpeed;
        //     if (keys.up)
        //         mPosition += getWorldUp() * moveSpeed;
        //     if (keys.down)
        //         mPosition -= getWorldUp() * moveSpeed;
        //     if (keys.left)
        //         mPosition -= getRight() * moveSpeed;
        //     if (keys.right)
        //         mPosition += getRight() * moveSpeed;
        //     if (keys.forward)
        //         mPosition += getFront() * moveSpeed;
        //     if (keys.back)
        //         mPosition -= getFront() * moveSpeed;
        //     updateViewMatrix();
        // }
    }

    // void rotate(double rightAngle, double upAngle)
    // {
    //     GfRotation a(getRight(), upAngle * rotationSpeed);
    //     GfRotation b(getWorldUp(), rightAngle * rotationSpeed);

    //     GfRotation c = a * b;
    //     GfQuatd cq = c.GetQuat();
    //     cq.Normalize();
    //     mOrientation = cq * mOrientation;
    //     mOrientation.Normalize();
    //     updateViewMatrix();
    // }

    // void translate(GfVec3d delta)
    // {
    //     // mPosition += mOrientation.Transform(delta);
    //     // updateViewMatrix();
    //     mCam.translate()
    // }

    void updateViewMatrix()
    {
        // GfMatrix4d view(1.0);
        // view.SetRotateOnly(mOrientation);
        // view.SetTranslateOnly(mPosition);

        // mGfCam.SetTransform(view);
        mCam.updateViewMatrix();
    }

    oka::Camera& getCamera()
    {
        return mCam;
    }

    CameraController(oka::Camera& cam, bool isYup)
    {
        if (isYup)
        {
            cam.setWorldUp(glm::float3(0.0, 1.0, 0.0));
            cam.setWorldForward(glm::float3(0.0, 0.0, -1.0));
        }
        else
        {
            cam.setWorldUp(glm::float3(0.0, 0.0, 1.0));
            cam.setWorldForward(glm::float3(0.0, 1.0, 0.0));
        }
        mCam = cam;
        // GfMatrix4d xform = mGfCam.GetTransform();
        // xform.Orthonormalize();
        // mOrientation = xform.ExtractRotationQuat();
        // mOrientation.Normalize();
        // mPosition = xform.ExtractTranslation();
    }

    void keyCallback(int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) override
    {
        const bool keyState = ((GLFW_REPEAT == action) || (GLFW_PRESS == action)) ? true : false;
        switch (key)
        {
        case GLFW_KEY_W: {
            mCam.keys.forward = keyState;
            break;
        }
        case GLFW_KEY_S: {
            mCam.keys.back = keyState;
            break;
        }
        case GLFW_KEY_A: {
            mCam.keys.left = keyState;
            break;
        }
        case GLFW_KEY_D: {
            mCam.keys.right = keyState;
            break;
        }
        case GLFW_KEY_Q: {
            mCam.keys.up = keyState;
            break;
        }
        case GLFW_KEY_E: {
            mCam.keys.down = keyState;
            break;
        }
        default:
            break;
        }
    }

    void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods) override
    {
        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_PRESS)
            {
                mCam.mouseButtons.right = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mCam.mouseButtons.right = false;
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            if (action == GLFW_PRESS)
            {
                mCam.mouseButtons.left = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mCam.mouseButtons.left = false;
            }
        }
    }

    void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos) override
    {
        const float dx = mCam.mousePos[0] - xpos;
        const float dy = mCam.mousePos[1] - ypos;

        // ImGuiIO& io = ImGui::GetIO();
        // bool handled = io.WantCaptureMouse;
        // if (handled)
        //{
        //    camera.mousePos = glm::vec2((float)xpos, (float)ypos);
        //    return;
        //}

        if (mCam.mouseButtons.right)
        {
            mCam.rotate(-dx, -dy);
        }
        if (mCam.mouseButtons.left)
        {
            mCam.translate(glm::float3(-0.0, 0.0, -dy * .005 * movementSpeed));
        }
        if (mCam.mouseButtons.middle)
        {
            mCam.translate(glm::float3(-dx * 0.01, -dy * 0.01, 0.0f));
        }
        mCam.mousePos[0] = xpos;
        mCam.mousePos[1] = ypos;
    }
};

int main(int argc, const char* argv[])
{
    const oka::Logmanager loggerManager;
    cxxopts::Options options("Strelka -s <Scene path>", "commands");

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
    const std::string sceneFile(result["s"].as<std::string>());
    if (sceneFile.empty())
    {
        STRELKA_FATAL("Specify scene file name");
        return 1;
    }
    if (!std::filesystem::exists(sceneFile))
    {
        STRELKA_FATAL("Specified scene file: {} doesn't exist", sceneFile.c_str());
        return -1;
    }
    const std::filesystem::path sceneFilePath = { sceneFile.c_str() };
    const std::string resourceSearchPath = sceneFilePath.parent_path().string();
    STRELKA_DEBUG("Resource search path {}", resourceSearchPath);

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
    ctx->mSettingsManager->setAs<uint32_t>("render/pt/SamplingType", 0); // 0 - Uniform, 1 - Halton, 2 - Sobol, 3 - Blue Noise
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
    
    oka::GltfLoader sceneLoader;
    sceneLoader.loadGltf(sceneFilePath.string(), scene);

    CameraController cameraController(scene.getCamera(0), true);

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
    
    display->setInputHandler(&cameraController);

    while (!display->windowShouldClose())
    {
        auto start = std::chrono::high_resolution_clock::now();

        display->pollEvents();

        static auto prevTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        const double deltaTime = std::chrono::duration<double, std::milli>(currentTime - prevTime).count() / 1000.0;

        const auto cameraSpeed = ctx->mSettingsManager->getAs<float>("render/cameraSpeed");
        cameraController.update(deltaTime, cameraSpeed);
        prevTime = currentTime;

        scene.updateCamera(cameraController.getCamera(), 0);

        display->onBeginFrame();

        render->render(outputBuffer);
        outputBuffer->map();
        oka::ImageBuffer outputImage;        
        outputImage.data = outputBuffer->getHostPointer();
        outputImage.dataSize = outputBuffer->getHostDataSize();
        outputImage.height = outputBuffer->height();
        outputImage.width = outputBuffer->width();
        outputImage.pixel_format = oka::BufferFormat::FLOAT4;
        display->drawFrame(outputImage); // blit rendered image to swapchain
        display->drawUI(); // render ui to swapchain image in window resolution
        display->onEndFrame(); // submit command buffer and present

        outputBuffer->unmap();

        const uint32_t currentSpp = ctx->mSubframeIndex;
        auto finish = std::chrono::high_resolution_clock::now();
        const double frameTime = std::chrono::duration<double, std::milli>(finish - start).count();

        display->setWindowTitle((std::string("Strelka") + " [" + std::to_string(frameTime) + " ms]" + " [" +
                                 std::to_string(currentSpp) + " spp]")
                                    .c_str());    }
    display->destroy();
    return 0;
}
