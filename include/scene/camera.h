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

    float fov = 45.0f;
    float znear = 0.1f, zfar = 1000.0f;

    // View dir -Z
    glm::quat mOrientation = { 1.0f, 0.0f, 0.0f, 0.0f };
    glm::float3 position = { 0.0f, 0.0f, 10.0f };
    glm::quat getOrientation();

    float rotationSpeed = 0.025f;
    float movementSpeed = 5.0f;

    bool updated = false;
    bool isDirty = true;

    struct MouseButtons
    {
        bool left = false;
        bool right = false;
        bool middle = false;
    } mouseButtons;

    glm::float2 mousePos;

    struct Matrices
    {
        glm::float4x4 perspective;
        glm::float4x4 invPerspective;
        glm::float4x4 view;
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

    glm::float3 getFront();
    glm::float3 getUp();
    glm::float3 getRight();
    bool moving();
    float getNearClip();
    float getFarClip();
    void setFov(float fov);
    void setPerspective(float fov, float aspect, float znear, float zfar);
    glm::float4x4& getPerspective();
    glm::float4x4 getView();
    void updateAspectRatio(float aspect);
    void setPosition(glm::float3 position);
    glm::float3 getPosition();
    void setRotation(glm::quat rotation);
    void rotate(float, float);
    void setTranslation(glm::float3 translation);
    void translate(glm::float3 delta);
    void update(float deltaTime);
};
} // namespace oka
