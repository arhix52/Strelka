#pragma once

#include <strelka/scene/camera.h>

#include <glm/ext/matrix_relational.hpp>

namespace oka::metal::temporal_history
{

/// A projection change invalidates temporal reprojection; a view change does not.
///
/// Camera translation and rotation are represented by motion vectors and the
/// current/previous transforms MetalFX receives. Treating their magnitude as a
/// cut discards the history during exactly the frames temporal reconstruction is
/// meant to stabilize. Actual cuts are signalled explicitly by the caller.
inline bool projectionChanged(const Camera::Matrices& current, const Camera::Matrices& previous)
{
    return glm::any(glm::notEqual(current.perspective, previous.perspective));
}

} // namespace oka::metal::temporal_history
