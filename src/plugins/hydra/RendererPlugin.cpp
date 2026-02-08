#include "RendererPlugin.h"
#include "RenderDelegate.h"

#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/base/plug/plugin.h>
#include "pxr/base/plug/thisPlugin.h"

#include <log.h>

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(TfType)
{
    HdRendererPluginRegistry::Define<HdStrelkaRendererPlugin>();
}

HdStrelkaRendererPlugin::HdStrelkaRendererPlugin()
{
    m_isSupported = true;
}

HdStrelkaRendererPlugin::~HdStrelkaRendererPlugin()
{
}

HdRenderDelegate* HdStrelkaRendererPlugin::CreateRenderDelegate()
{
    HdRenderSettingsMap settingsMap = {};

    return new HdStrelkaRenderDelegate(settingsMap);
}

HdRenderDelegate* HdStrelkaRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
    return new HdStrelkaRenderDelegate(settingsMap);
}

void HdStrelkaRendererPlugin::DeleteRenderDelegate(HdRenderDelegate* renderDelegate)
{
    delete renderDelegate;
}

bool HdStrelkaRendererPlugin::IsSupported(bool gpuEnabled) const
{
    return m_isSupported;
}

PXR_NAMESPACE_CLOSE_SCOPE
