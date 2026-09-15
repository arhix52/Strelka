#pragma once

#include <strelka/scene/glm_wrapper.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
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

    enum class ProjectionType : uint32_t
    {
        perspective = 0,
        orthographic = 1
    };
    ProjectionType projection = ProjectionType::perspective;

    float fov = 45.0f; // vertical, degrees

    // Orthographic half-extents in world units, i.e. glTF xmag / ymag. Blender's
    // `ortho_scale` is the full extent of the fitted axis, so it is half of that.
    float xmag = 1.0f;
    float ymag = 1.0f;
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
    glm::float3 mWorldUp = { 0.0, 1.0, 0.0 };
    glm::float3 mWorldForward = { 0.0, 0.0, -1.0 };
    glm::quat getOrientation();

    float rotationSpeed = 0.025f;
    float movementSpeed = 5.0f;

    float movementSmoothing = 0.0f;
    glm::float3 mMoveInput{ 0.0f };

    bool updated = false;
    bool isDirty = true;

    bool manualControl = false;

    struct MouseButtons
    {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    glm::float2 mousePos{ 0.0f };

    struct Matrices
    {
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
    /// Whether smoothed input is still carrying the camera after the keys were
    /// released. The editor has to keep treating that as user-driven motion, or
    /// animation poses the camera back mid-glide.
    bool isSettling() const;
    float getNearClip() const;
    float getFarClip() const;
    void setFov(float fov);
    void setPerspective(float fov, float aspect, float znear, float zfar);
    void setOrthographic(float xmag, float ymag, float znear, float zfar);
    /// Orthographic half-extents to render `aspect` with, given what the camera
    /// was authored for. Mirrors fovForAspect: the wider axis is the one held.
    void magForAspect(float aspect, float& halfWidth, float& halfHeight) const;
    /// Zoom an orthographic camera by scaling its film extents. `factor` below 1
    /// moves in. No-op on a perspective camera, which zooms by moving.
    void zoomOrthographic(float factor);
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

void generatePickRay(const Camera& camera, const glm::float2& uv, glm::float3& origin, glm::float3& direction);

} // namespace oka
