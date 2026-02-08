#pragma once

#include <pxr/imaging/hd/renderDelegate.h>

#include <strelka/render/common.h>
#include <strelka/scene/scene.h>
#include <strelka/render/render.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaRenderDelegate final : public HdRenderDelegate
{
public:
    HdStrelkaRenderDelegate(const HdRenderSettingsMap& settingsMap);

    ~HdStrelkaRenderDelegate() override;

    void SetDrivers(HdDriverVector const& drivers) override;

    HdRenderSettingDescriptorList GetRenderSettingDescriptors() const override;

    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index, const HdRprimCollection& collection) override;

    HdResourceRegistrySharedPtr GetResourceRegistry() const override;

    void CommitResources(HdChangeTracker* tracker) override;

    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, const SdfPath& id) override;

    void DestroyInstancer(HdInstancer* instancer) override;

    HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override;

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

    TfToken GetMaterialBindingPurpose() const override;

    TfTokenVector GetMaterialRenderContexts() const override;

    TfTokenVector GetShaderSourceTypes() const override;

    oka::SharedContext& getSharedContext();

private:
    HdRenderSettingDescriptorList m_settingDescriptors;
    HdResourceRegistrySharedPtr m_resourceRegistry;

    const TfTokenVector SUPPORTED_BPRIM_TYPES = { HdPrimTypeTokens->renderBuffer };
    const TfTokenVector SUPPORTED_RPRIM_TYPES = { HdPrimTypeTokens->mesh, HdPrimTypeTokens->basisCurves };
    const TfTokenVector SUPPORTED_SPRIM_TYPES = {
        HdPrimTypeTokens->camera,    HdPrimTypeTokens->material,  HdPrimTypeTokens->light,
        HdPrimTypeTokens->rectLight, HdPrimTypeTokens->diskLight, HdPrimTypeTokens->sphereLight,
        HdPrimTypeTokens->distantLight,
    };

    oka::SharedContext* mSharedCtx;
    oka::Scene mScene;
    oka::Render* mRenderer;
};

PXR_NAMESPACE_CLOSE_SCOPE
