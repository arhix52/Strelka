#pragma once

#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/pxr.h>

#include <render/common.h>
#include <render/buffer.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdStrelkaRenderBuffer final : public HdRenderBuffer
{
public:
    HdStrelkaRenderBuffer(const SdfPath& id, oka::SharedContext* ctx);

    ~HdStrelkaRenderBuffer() override;

public:
    bool Allocate(const GfVec3i& dimensions, HdFormat format, bool multiSamples) override;

public:
    unsigned int GetWidth() const override;

    unsigned int GetHeight() const override;

    unsigned int GetDepth() const override;

    HdFormat GetFormat() const override;

    bool IsMultiSampled() const override;

    VtValue GetResource(bool multiSampled) const override;

public:
    bool IsConverged() const override;

    void SetConverged(bool converged);

public:
    void* Map() override;

    bool IsMapped() const override;

    void Unmap() override;

    void Resolve() override;

protected:
    void _Deallocate() override;

private:
    // oka::Image* mResult = nullptr;
    oka::SharedContext* mCtx = nullptr;
    oka::CUDAOutputBuffer<uchar4>* mResult = nullptr;

    void* m_bufferMem = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    HdFormat m_format;
    bool m_isMultiSampled;
    bool m_isMapped;
    bool m_isConverged;
};

PXR_NAMESPACE_CLOSE_SCOPE
