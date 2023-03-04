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
    const PlugPluginPtr plugin = PLUG_THIS_PLUGIN;

    const std::string& resourcePath = plugin->GetResourcePath();
    STRELKA_INFO("Resource path: {}", resourcePath.c_str());
    const std::string shaderPath = resourcePath + "/shaders";
    const std::string mtlxmdlPath = resourcePath + "/mtlxmdl";
    const std::string mtlxlibPath = resourcePath + "/mtlxlib";

    // m_translator = std::make_unique<MaterialNetworkTranslator>(mtlxlibPath);
    const char* envUSDPath = std::getenv("USD_DIR");
    if (!envUSDPath)
    {
        STRELKA_FATAL("Please, set USD_DIR variable\n");
        assert(0);
        m_isSupported = false;
    }
    else
    {
        const std::string USDPath(envUSDPath);
        m_translator = std::make_unique<MaterialNetworkTranslator>(USDPath + "./libraries");
        m_isSupported = true;
    }
}

HdStrelkaRendererPlugin::~HdStrelkaRendererPlugin()
{
}

HdRenderDelegate* HdStrelkaRendererPlugin::CreateRenderDelegate()
{
    HdRenderSettingsMap settingsMap = {};

    return new HdStrelkaRenderDelegate(settingsMap, *m_translator);
}

HdRenderDelegate* HdStrelkaRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
    return new HdStrelkaRenderDelegate(settingsMap, *m_translator);
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
