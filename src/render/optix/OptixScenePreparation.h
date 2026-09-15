#pragma once

#include <host/scene_preparation.h>

namespace oka::optix
{

using scene_preparation::buildProgressBefore;
using scene_preparation::BuildStage;
using scene_preparation::buildStageName;
using scene_preparation::SceneBuildHooks;
using OptixScenePreparation = scene_preparation::ScenePreparation;

} // namespace oka::optix
