#include <strelka/render/render.h>
#ifdef __APPLE__
#    include "metal/MetalRender.h"
#else
#    include "optix/OptixRender.h"
#endif

using namespace oka;

Render* RenderFactory::createRender()
{
#ifdef __APPLE__
    return new MetalRender();
#else
    return new OptiXRender();
#endif
}
