#include <strelka/scene/camera.h>

#include <glm/gtx/quaternion.hpp>

#include <strelka/scene/glm_wrapper.hpp>

namespace oka
{

void Camera::updateViewMatrix()
{
    const glm::float4x4 rotM{ mOrientation };
    const glm::float4x4 transM = glm::translate(glm::float4x4(1.0f), -position);
    if (type == CameraType::firstperson)
    {
        matrices.view = rotM * transM;
    }
    else
    {
        matrices.view = transM * rotM;
    }
    updated = true;
}

glm::float3 Camera::getFront() const
{
    return glm::conjugate(mOrientation) * glm::float3(0.0f, 0.0f, -1.0f);
}

glm::float3 Camera::getUp() const
{
    return glm::conjugate(mOrientation) * glm::float3(0.0f, 1.0f, 0.0f);
}

glm::float3 Camera::getRight() const
{
    return glm::conjugate(mOrientation) * glm::float3(1.0f, 0.0f, 0.0f);
}

bool Camera::moving() const
{
    return keys.left || keys.right || keys.up || keys.down || keys.forward || keys.back || mouseButtons.right || mouseButtons.left || mouseButtons.middle;
}

float Camera::getNearClip() const
{
    return znear;
}

float Camera::getFarClip() const
{
    return zfar;
}

void Camera::setFov(float fov)
{
    this->fov = fov;
}

// original implementation: https://vincent-p.github.io/notes/20201216234910-the_projection_matrix_in_vulkan/
glm::float4x4 perspective(float fov, float aspect_ratio, float n, float f, glm::float4x4* inverse)
{
    const float focal_length = 1.0f / std::tan(glm::radians(fov) / 2.0f);

    const float x = focal_length / aspect_ratio;
    const float y = focal_length;
    const float A = n / (f - n);
    const float B = f * A;

    //glm::float4x4 projection = glm::perspective(fov, aspect_ratio, n, f);
    //if (inverse)
    //{
    //    *inverse = glm::inverse(projection);
    //}


    glm::float4x4 projection({
        x,
        0.0f,
        0.0f,
        0.0f,

        0.0f,
        y,
        0.0f,
        0.0f,

        0.0f,
        0.0f,
        A,
        B,

        0.0f,
        0.0f,
        -1.0f,
        0.0f,
    });

    if (inverse)
    {
        *inverse = glm::transpose(glm::float4x4({ // glm inverse
            1 / x,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            1 / y,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            -1.0f,
            0.0f,
            0.0f,
            1 / B,
            A / B,
        }));
    }

    return glm::transpose(projection);
}

void Camera::setPerspective(float _fov, float _aspect, float _znear, float _zfar)
{
    fov = _fov;
    znear = _znear;
    zfar = _zfar;
    // swap near and far plane for reverse z
    matrices.perspective = perspective(fov, _aspect, zfar, znear, &matrices.invPerspective);
}

void Camera::setWorldUp(const glm::float3 up)
{
    mWorldUp = up;
}

glm::float3 Camera::getWorldUp()
{
    return mWorldUp;
}

void Camera::setWorldForward(const glm::float3 forward)
{
    mWorldForward = forward;
}

glm::float3 Camera::getWorldForward()
{
    return mWorldForward;
}

glm::float4x4& Camera::getPerspective()
{
    return matrices.perspective;
}

glm::float4x4 Camera::getView()
{
    return matrices.view;
}

void Camera::updateAspectRatio(float _aspect)
{
    setPerspective(fov, _aspect, znear, zfar);
}

void Camera::setPosition(glm::float3 _position)
{
    position = _position;
    updateViewMatrix();
}

glm::float3 Camera::getPosition()
{
    return position;
}

void Camera::setRotation(glm::quat rotation)
{
    mOrientation = rotation;
    updateViewMatrix();
}

void Camera::rotate(float rightAngle, float upAngle)
{
    const glm::quat a = glm::angleAxis(glm::radians(upAngle) * rotationSpeed, glm::float3(1.0f, 0.0f, 0.0f));
    const glm::quat b = glm::angleAxis(glm::radians(rightAngle) * rotationSpeed, glm::float3(0.0f, 1.0f, 0.0f));
    // const glm::quat a = glm::angleAxis(glm::radians(upAngle) * rotationSpeed, getRight());
    // const glm::quat b = glm::angleAxis(glm::radians(rightAngle) * rotationSpeed, getWorldUp());
    // auto c = a * b;
    // c = glm::normalize(c);
    mOrientation = glm::normalize(a * mOrientation * b);
    // mOrientation = glm::normalize(c * mOrientation);
    updateViewMatrix();
}

glm::quat Camera::getOrientation()
{
    return mOrientation;
}

void Camera::setTranslation(glm::float3 translation)
{
    position = translation;
    updateViewMatrix();
}

void Camera::translate(glm::float3 delta)
{
    position += glm::conjugate(mOrientation) * delta;
    updateViewMatrix();
}

void Camera::update(float deltaTime)
{
    updated = false;
    if (type == CameraType::firstperson)
    {
        if (moving())
        {
            float moveSpeed = deltaTime * movementSpeed;
            if (keys.up)
                position += getWorldUp() * moveSpeed;
            if (keys.down)
                position -= getWorldUp() * moveSpeed;
            if (keys.left)
                position -= getRight() * moveSpeed;
            if (keys.right)
                position += getRight() * moveSpeed;
            if (keys.forward)
                position += getFront() * moveSpeed;
            if (keys.back)
                position -= getFront() * moveSpeed;
            updateViewMatrix();
        }
    }
}

void generatePickRay(const Camera& camera, const glm::float2& uv, glm::float3& origin, glm::float3& direction)
{
    // Same chain as the shader: NDC with y up, unprojected through clipToView,
    // then rotated into world space. The w component is deliberately left alone;
    // only the direction matters here.
    const float ndcX = uv.x * 2.0f - 1.0f + camera.shiftX * 2.0f;
    const float ndcY = (1.0f - uv.y) * 2.0f - 1.0f + camera.shiftY * 2.0f;
    const glm::float4 viewSpace = camera.matrices.invPerspective * glm::float4(ndcX, ndcY, 1.0f, 1.0f);
    const glm::float4x4 viewToWorld = glm::inverse(camera.matrices.view);

    origin = glm::float3(viewToWorld * glm::float4(0.0f, 0.0f, 0.0f, 1.0f));
    direction = glm::normalize(glm::float3(viewToWorld * glm::float4(viewSpace.x, viewSpace.y, viewSpace.z, 0.0f)));
}

} // namespace oka
