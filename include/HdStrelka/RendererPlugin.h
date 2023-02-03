#pragma once

#include <pxr/imaging/hd/rendererPlugin.h>

#include <memory>

#include "MaterialNetworkTranslator.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaRendererPlugin final : public HdRendererPlugin
{
public:
    HdStrelkaRendererPlugin();

    ~HdStrelkaRendererPlugin() override;

public:
    HdRenderDelegate* CreateRenderDelegate() override;

    HdRenderDelegate* CreateRenderDelegate(const HdRenderSettingsMap& settingsMap) override;

    void DeleteRenderDelegate(HdRenderDelegate* renderDelegate) override;

    bool IsSupported(bool gpuEnabled = true) const override;

private:
    std::unique_ptr<MaterialNetworkTranslator> m_translator;
    bool m_isSupported;
};

PXR_NAMESPACE_CLOSE_SCOPE
