#pragma once

#define GLM_FORCE_SILENT_WARNINGS
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/compatibility.hpp>

#include <string>

namespace oka
{

class Camera
{
public:
    std::string name = "Default camera";
    int node = -1;
    enum class CameraType : uint32_t
    {
        lookat,
        firstperson
    };
    CameraType type = CameraType::firstperson;

    /// Perspective or orthographic, matching glTF's two camera types.
    ///
    /// An orthographic camera is not a perspective one with a long lens: it has
    /// no centre of projection, so the primary ray's origin varies across the
    /// film and its direction does not. That is a branch in ray generation, not
    /// a different projection matrix -- which is why this is carried to the GPU
    /// as its own field rather than being left implicit in clipToView.
    enum class ProjectionType : uint32_t
    {
        perspective = 0,
        orthographic = 1
    };
    ProjectionType projection = ProjectionType::perspective;

    float fov = 45.0f;   // vertical, degrees

    // Orthographic half-extents in world units, i.e. glTF xmag / ymag. Blender's
    // `ortho_scale` is the full extent of the fitted axis, so it is half of that.
    float xmag = 1.0f;
    float ymag = 1.0f;
    // The frame aspect the fov was authored against, 0 when unknown. glTF's yfov
    // means nothing without it: a camera authored for 16:9 and rendered at 4:3
    // has to keep its *horizontal* angle, which is what every DCC does for a
    // landscape frame and what a renderer that keeps the vertical angle instead
    // gets wrong by exactly the ratio of the two aspects.
    float authoredAspect = 0.0f;
    float znear = 0.1f, zfar = 1000.0f;

    // Depth of field
    bool useDof = false;
    float focalDistance = 10.0f;
    float fStopDof = 2.8f;
    int apertureBlades = 0;
    float bladeRotation = 0.0f;
    float anamorphicRatio = 1.0f;

    // Sensor / lens
    float focalLengthMm = 50.0f;
    float sensorWidth = 36.0f;
    float sensorHeight = 24.0f;
    float shiftX = 0.0f;
    float shiftY = 0.0f;

    // View dir -Z
    glm::quat mOrientation = { 1.0f, 0.0f, 0.0f, 0.0f };
    glm::float3 position = { 0.0f, 0.0f, 10.0f };
    glm::float3 mWorldUp = {0.0, 1.0, 0.0};
    glm::float3 mWorldForward = {0.0, 0.0, -1.0};
    glm::quat getOrientation();

    float rotationSpeed = 0.025f;
    float movementSpeed = 5.0f;

    bool updated = false;
    bool isDirty = true;

    /// While set, animation leaves this camera's pose alone.
    ///
    /// A glTF camera is posed from its node, so playback and an editor that has
    /// handed the camera to the user are two owners of one transform: the frame
    /// gets rendered from the animated pose while the viewport overlay and
    /// picking use the user's, and the two disagree on screen.
    bool manualControl = false;

    struct MouseButtons
    {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    glm::float2 mousePos;

    struct Matrices
    {
        // Identity, not uninitialised: these are read by picking, gizmos and the
        // renderer, and a camera can reach any of them before setPerspective or
        // updateViewMatrix has run. Garbage here turns into inf/NaN rays that fail
        // silently instead of visibly.
        glm::float4x4 perspective{ 1.0f };
        glm::float4x4 invPerspective{ 1.0f };
        glm::float4x4 view{ 1.0f };
    };
    Matrices matrices;

    void updateViewMatrix();

    struct
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
        bool forward = false;
        bool back = false;
    } keys;

    glm::float3 getFront() const;
    glm::float3 getUp() const;
    glm::float3 getRight() const;
    bool moving() const;
    float getNearClip() const;
    float getFarClip() const;
    void setFov(float fov);
    void setPerspective(float fov, float aspect, float znear, float zfar);
    void setOrthographic(float xmag, float ymag, float znear, float zfar);
    /// Orthographic half-extents to render `aspect` with, given what the camera
    /// was authored for. Mirrors fovForAspect: the wider axis is the one held.
    void magForAspect(float aspect, float& halfWidth, float& halfHeight) const;
    void setWorldUp(const glm::float3 up);
    glm::float3 getWorldUp();
    void setWorldForward(const glm::float3 forward);
    glm::float3 getWorldForward();
    glm::float4x4& getPerspective();
    glm::float4x4 getView();
    void updateAspectRatio(float aspect);
    float fovForAspect(float aspect) const;
    void setPosition(glm::float3 position);
    glm::float3 getPosition();
    void setRotation(glm::quat rotation);
    void rotate(float, float);
    void setTranslation(glm::float3 translation);
    void translate(glm::float3 delta);
    void update(float deltaTime);
};

/// Primary ray through a point of the rendered image, where uv is normalised
/// image space with (0,0) at the top-left corner.
///
/// Kept next to the camera (and not in the editor) because it has to stay in
/// lockstep with generateCameraRay in the shaders: a CPU pick that maps pixels
/// differently than the renderer selects something other than what the user
/// clicked, and the mismatch is invisible until it is off by a mirrored axis.
void generatePickRay(const Camera& camera, const glm::float2& uv, glm::float3& origin, glm::float3& direction);

} // namespace oka
