

namespace oka
{
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

    bool moving() const
    {
        return keys.left || keys.right || keys.up || keys.down || keys.forward || keys.back || mouseButtons.right ||
               mouseButtons.left || mouseButtons.middle;
    }
    void update(double deltaTime, float speed)
    {
        mCam.rotationSpeed = rotationSpeed;
        mCam.movementSpeed = speed;
        mCam.update(deltaTime);
    }

    void updateViewMatrix()
    {
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

} // namespace oka
