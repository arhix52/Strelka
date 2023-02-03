#pragma once

#include <pxr/imaging/hd/renderDelegate.h>

#include "MaterialNetworkTranslator.h"

#include <render/common.h>
#include <scene/scene.h>
#include <render/render.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaRenderDelegate final : public HdRenderDelegate
{
public:
    HdStrelkaRenderDelegate(const HdRenderSettingsMap& settingsMap, const MaterialNetworkTranslator& translator);

    ~HdStrelkaRenderDelegate() override;

public:
    void SetDrivers(HdDriverVector const& drivers) override;

    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;

public:
    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index, const HdRprimCollection& collection) override;

    HdResourceRegistrySharedPtr GetResourceRegistry() const override;

    void CommitResources(HdChangeTracker* tracker) override;

    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, const SdfPath& id) override;

    void DestroyInstancer(HdInstancer* instancer) override;

    HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;

public:
    /* Rprim */
    const TfTokenVector& GetSupportedRprimTypes() const override;

    HdRprim* CreateRprim(const TfToken& typeId, const SdfPath& rprimId) override;

    void DestroyRprim(HdRprim* rPrim) override;

    /* Sprim */
    const TfTokenVector& GetSupportedSprimTypes() const override;

    HdSprim* CreateSprim(const TfToken& typeId, const SdfPath& sprimId) override;

    HdSprim* CreateFallbackSprim(const TfToken& typeId) override;

    void DestroySprim(HdSprim* sprim) override;

    /* Bprim */
    const TfTokenVector& GetSupportedBprimTypes() const override;

    HdBprim* CreateBprim(const TfToken& typeId, const SdfPath& bprimId) override;

    HdBprim* CreateFallbackBprim(const TfToken& typeId) override;

    void DestroyBprim(HdBprim* bprim) override;

public:
    TfToken GetMaterialBindingPurpose() const override;

    // In a USD file, there can be multiple networks associated with a material:
    //   token outputs:mdl:surface.connect = </Root/Glass.outputs:out>
    //   token outputs:surface.connect = </Root/GlassPreviewSurface.outputs:surface>
    // This function returns the order of preference used when selecting one for rendering.
    TfTokenVector GetMaterialRenderContexts() const override;

    TfTokenVector GetShaderSourceTypes() const override;

public:
    oka::SharedContext& getSharedContext();

private:
    const MaterialNetworkTranslator& m_translator;
    HdRenderSettingDescriptorList m_settingDescriptors;
    HdResourceRegistrySharedPtr m_resourceRegistry;

    oka::SharedContext* mSharedCtx;
    oka::Scene mScene;
    oka::Render* mRenderer;
};

PXR_NAMESPACE_CLOSE_SCOPE
