#pragma once
#include <strelka/display/display.h>
#include <strelka/scene/camera.h>
#include <GLFW/glfw3.h>

#include "gamepad_camera.h"

#include <algorithm>
#include <cmath>

namespace oka
{
class CameraController : public oka::InputHandler
{
    Camera mCam;

    float rotationSpeed = 0.025f;
    float movementSpeed = 1.0f;
    float keyRotationSpeed = 60.0f; // degrees per second for arrow key rotation

    // Smoothing time constants, in seconds. Everything the user drives the camera
    // with is filtered through one of these, because the raw input is not the
    // problem -- the frame it lands on is. A path tracer's frame time swings by a
    // factor of several between the frame that restarts accumulation and the ones
    // that follow, so equal input per frame is unequal motion per frame, and that
    // is what reads as jerky. Filtering spreads each frame's share over the next
    // few, which costs a little latency and buys motion that does not step.
    //
    // Look is the tighter of the two: the hand expects the view to follow the
    // mouse, and past ~50 ms the lag is felt as the camera lagging rather than as
    // smoothness.
    static constexpr float kMoveSmoothingSec = 0.09f;
    static constexpr float kLookSmoothingSec = 0.045f;
    // A frame longer than this is a hitch -- a scene load, a window drag, a
    // breakpoint -- not slow rendering. Integrating it would teleport the camera
    // by whatever the pause happened to last, so it is treated as one slow frame.
    static constexpr float kMaxFrameSec = 0.1f;

    // Input the user has given that has not been applied to the camera yet, in the
    // units rotate()/translate() take. A buffer rather than a filtered value, so
    // nothing is lost or invented: every pixel of mouse travel still reaches the
    // camera, just spread over a few frames.
    float mPendingLookX = 0.0f, mPendingLookY = 0.0f;
    glm::float3 mPendingTranslate{ 0.0f };
    // E-folds of orthographic zoom per wheel notch: ~16% of the frame per click,
    // and it composes smoothly with the fractional deltas a trackpad sends.
    static constexpr float kWheelZoomRate = 0.15f;

    bool mIsViewportHovered = false;
    bool mGizmoBlocksInput = false;

    // Whether user input actually moved the camera, as opposed to a mouse button
    // merely being down: selecting in the viewport holds the left button, and a
    // click that picks an object must not read as taking the camera over.
    bool mUserMovedCamera = false;
    // Cursor travel with a button held, reset on press. Matched to the drag
    // threshold picking uses, so one gesture cannot both select and take over.
    float mDragDistance = 0.0f;
    static constexpr float kDragTakeoverPixels = 4.0f;

    struct RotateKeys
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } mRotateKeys;

public:
    ~CameraController() override = default;

    void setGizmoBlocksInput(bool blocks)
    {
        mGizmoBlocksInput = blocks;
        if (blocks)
        {
            mCam.mouseButtons.left = false;
            mCam.mouseButtons.right = false;
            mCam.mouseButtons.middle = false;
        }
    }

    /// Whether the user drove the camera since the last call, clearing the flag.
    bool consumeUserMovedCamera()
    {
        const bool moved = mUserMovedCamera;
        mUserMovedCamera = false;
        return moved;
    }

    void update(double deltaTime, float speed)
    {
        const float dt = std::min(static_cast<float>(deltaTime), kMaxFrameSec);
        mCam.rotationSpeed = rotationSpeed;
        mCam.movementSpeed = speed;
        mCam.movementSmoothing = kMoveSmoothingSec;
        if (mCam.keys.left || mCam.keys.right || mCam.keys.up || mCam.keys.down || mCam.keys.forward || mCam.keys.back)
        {
            mUserMovedCamera = true;
        }
        mCam.update(dt);
        // The camera can still be gliding after the key came up. That is the user's
        // motion finishing, not animation's turn to pose the camera back.
        if (mCam.isSettling())
        {
            mUserMovedCamera = true;
        }

        // Arrow key rotation. Queued rather than applied, so it goes through the
        // same filter the mouse does and a tap does not start and stop abruptly.
        if (mRotateKeys.left || mRotateKeys.right || mRotateKeys.up || mRotateKeys.down)
        {
            mUserMovedCamera = true;
            if (mRotateKeys.left)
                mPendingLookX -= keyRotationSpeed * dt;
            if (mRotateKeys.right)
                mPendingLookX += keyRotationSpeed * dt;
            if (mRotateKeys.up)
                mPendingLookY -= keyRotationSpeed * dt;
            if (mRotateKeys.down)
                mPendingLookY += keyRotationSpeed * dt;
        }

        applyPendingInput(dt);
    }

    /// Hand the queued look/translate input to the camera, a fixed fraction of
    /// what is left per unit of wall-clock time.
    ///
    /// Exponential rather than a fixed number of frames: the fraction depends on
    /// dt, so the camera arrives at the same place at the same time whether the
    /// scene renders at 15 fps or 120, and the feel of the control does not change
    /// with how expensive the scene is.
    void applyPendingInput(float dt)
    {
        // Below this a residual would take forever to reach zero (an exponential
        // never does) while keeping the camera nominally in motion, which keeps
        // accumulation restarting on a camera nobody is touching.
        constexpr float kLookEpsilon = 1e-3f; // degrees of raw mouse travel
        constexpr float kTranslateEpsilon = 1e-6f; // world units

        const float alpha = (dt > 0.0f) ? (1.0f - std::exp(-dt / kLookSmoothingSec)) : 1.0f;

        if (std::abs(mPendingLookX) > kLookEpsilon || std::abs(mPendingLookY) > kLookEpsilon)
        {
            const float dx = mPendingLookX * alpha;
            const float dy = mPendingLookY * alpha;
            mPendingLookX -= dx;
            mPendingLookY -= dy;
            mCam.rotate(dx, dy);
        }
        else
        {
            mPendingLookX = mPendingLookY = 0.0f;
        }

        if (glm::dot(mPendingTranslate, mPendingTranslate) > kTranslateEpsilon * kTranslateEpsilon)
        {
            const glm::float3 step = mPendingTranslate * alpha;
            mPendingTranslate -= step;
            mCam.translate(step);
        }
        else
        {
            mPendingTranslate = glm::float3(0.0f);
        }
    }

    /// Add a frame of gamepad input to the same queues the mouse and keys feed.
    ///
    /// Queued rather than applied, for the reason handleMouseMoveCallback is:
    /// everything the user drives the camera with goes through one filter, so a
    /// hand on the stick and a hand on the mouse compose instead of fighting, and
    /// the stick inherits the smoothing that makes motion survive a path tracer's
    /// uneven frame times.
    ///
    /// Gated on the viewport being hovered like the movement keys are -- a stick
    /// held while the user is in a text field must not fly the camera -- except
    /// that a pad has no cursor, so "hovered" here means the viewport is the
    /// thing the pointer is over, which is the editor's normal resting state.
    void applyGamepad(const gamepad::CameraInput& input)
    {
        if (mGizmoBlocksInput || !input.active)
        {
            return;
        }
        mUserMovedCamera = true;
        mPendingLookX += input.lookX;
        mPendingLookY += input.lookY;
        mPendingTranslate += input.translate;
        if (input.worldUp != 0.0f)
        {
            // The queue is in camera space and this lift is in world space, so it
            // is rotated into the queue's frame rather than the queue being split
            // in two. Camera::translate applies conjugate(orientation), so
            // orientation * v is the delta that comes back out as world v.
            mPendingTranslate += mCam.mOrientation * (mCam.getWorldUp() * input.worldUp);
        }
    }

    /// Drop queued input without applying it. Used where the camera's pose is
    /// being replaced outright, so half a gesture cannot arrive on top of the new
    /// one a frame later.
    void clearPendingInput()
    {
        mPendingLookX = mPendingLookY = 0.0f;
        mPendingTranslate = glm::float3(0.0f);
        mCam.mMoveInput = glm::float3(0.0f);
    }

    void updateViewMatrix()
    {
        mCam.updateViewMatrix();
    }

    void setViewportHovered(bool hovered)
    {
        mIsViewportHovered = hovered;
        if (!mIsViewportHovered)
        {
            mCam.keys.left = false;
            mCam.keys.right = false;
            mCam.keys.up = false;
            mCam.keys.down = false;
            mCam.keys.forward = false;
            mCam.keys.back = false;
            mRotateKeys = {};
        }
    }

    Camera& getCamera()
    {
        return mCam;
    }

    void setCamera(Camera& cam)
    {
        mCam = cam;
        mCam.movementSmoothing = kMoveSmoothingSec;
        clearPendingInput();
    }

    CameraController(Camera& cam, bool isYup)
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
        mCam.movementSmoothing = kMoveSmoothingSec;
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
        case GLFW_KEY_LEFT: {
            mRotateKeys.left = keyState;
            break;
        }
        case GLFW_KEY_RIGHT: {
            mRotateKeys.right = keyState;
            break;
        }
        case GLFW_KEY_UP: {
            mRotateKeys.up = keyState;
            break;
        }
        case GLFW_KEY_DOWN: {
            mRotateKeys.down = keyState;
            break;
        }
        default:
            break;
        }
    }

    void mouseButtonCallback(int button, int action, [[maybe_unused]] int mods, bool viewPortHovered) override
    {
        if (mGizmoBlocksInput)
        {
            if (action == GLFW_RELEASE)
            {
                mCam.mouseButtons.left = false;
                mCam.mouseButtons.right = false;
            }
            return;
        }

        if (action == GLFW_PRESS)
        {
            mDragDistance = 0.0f;
        }

        if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if (action == GLFW_PRESS && viewPortHovered)
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
            if (action == GLFW_PRESS && viewPortHovered)
            {
                mCam.mouseButtons.left = true;
            }
            else if (action == GLFW_RELEASE)
            {
                mCam.mouseButtons.left = false;
            }
        }
    }

    // Only an orthographic camera zooms on the wheel. A perspective one zooms by
    // moving, which the left-drag dolly and the movement keys already do, and
    // taking the wheel over for a second way to do it would change a control that
    // every existing scene is driven with.
    void scrollCallback([[maybe_unused]] double xoffset, double yoffset) override
    {
        if (mGizmoBlocksInput || yoffset == 0.0 || mCam.projection != Camera::ProjectionType::orthographic)
        {
            return;
        }
        mUserMovedCamera = true;
        mCam.zoomOrthographic(std::exp(-kWheelZoomRate * (float)yoffset));
    }

    void handleMouseMoveCallback([[maybe_unused]] double xpos, [[maybe_unused]] double ypos) override
    {
        if (mGizmoBlocksInput)
        {
            mCam.mousePos[0] = static_cast<float>(xpos);
            mCam.mousePos[1] = static_cast<float>(ypos);
            return;
        }

        const float dx = mCam.mousePos[0] - static_cast<float>(xpos);
        const float dy = mCam.mousePos[1] - static_cast<float>(ypos);

        if (mCam.mouseButtons.right || mCam.mouseButtons.left || mCam.mouseButtons.middle)
        {
            mDragDistance += std::abs(dx) + std::abs(dy);
            if (mDragDistance > kDragTakeoverPixels)
            {
                mUserMovedCamera = true;
            }
        }

        // Queued, not applied: mouse motion arrives in a burst of callbacks inside
        // one pollEvents, so applying it here puts a whole frame's travel into a
        // single step however long that frame turned out to be. applyPendingInput
        // pays it out against the clock instead. Nothing is dropped -- the queue
        // drains -- so a gesture still turns the camera by exactly as much.
        if (mCam.mouseButtons.right)
        {
            mPendingLookX += -dx;
            mPendingLookY += -dy;
        }
        if (mCam.mouseButtons.left)
        {
            mPendingTranslate.z += -dy * .005f * movementSpeed;
        }
        if (mCam.mouseButtons.middle)
        {
            mPendingTranslate.x += -dx * 0.01f;
            mPendingTranslate.y += -dy * 0.01f;
        }
        mCam.mousePos[0] = static_cast<float>(xpos);
        mCam.mousePos[1] = static_cast<float>(ypos);
    }
};

} // namespace oka
