#include "render.h"
#include "metal/MetalRender.h"
#ifndef __APPLE__
#include "optix/OptixRender.h"
#endif

using namespace oka;

Render* RenderFactory::createRender(const RenderType type)
{
    if (type == RenderType::eMetal)
    {
        return new MetalRender();
    }
    else
    {
        // unsupported
        return nullptr;
    }
}
