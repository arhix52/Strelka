#pragma once

#include <strelka/scene/camera.h>

#include <glm/ext/matrix_relational.hpp>

namespace oka::metal::temporal_history
{

inline bool projectionChanged(const Camera::Matrices& current, const Camera::Matrices& previous)
{
    return glm::any(glm::notEqual(current.perspective, previous.perspective));
}

} // namespace oka::metal::temporal_history
