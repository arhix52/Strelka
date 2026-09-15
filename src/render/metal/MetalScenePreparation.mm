#include "MetalScenePreparation.h"

#include <Metal/Metal.hpp>


namespace oka::metal
{

bool MetalScenePreparation::step(SceneBuildHooks& hooks, Buffer* output)
{
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();
    const bool done = mPreparation.step(hooks, output);
    pPool->release();
    return done;
}

void MetalScenePreparation::finish(SceneBuildHooks& hooks, Buffer* output)
{
    while (!isDone())
    {
        step(hooks, output);
    }
}

} // namespace oka::metal
