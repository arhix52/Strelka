#include "render.h"
#ifdef __APPLE__
#include "metal/MetalRender.h"
#else
#include "optix/OptixRender.h"
#endif

using namespace oka;

Render* RenderFactory::createRender(const RenderType type)
{
#ifdef __APPLE__
    if (type == RenderType::eMetal)
    {
        return new MetalRender();
    }
    else
    {
        // unsupported
        return nullptr;
    }
#else
    if (type == RenderType::eOptiX)
    {
        return new OptiXRender();
    }
    else
    {
        return nullptr;
    }
#endif
}
