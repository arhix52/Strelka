#pragma once

#include <strelka/render/render.h>

#include <optix.h>

#include "OptixRenderParams.h"

#include <strelka/scene/scene.h>

#include "cuda_checks.h"
#include <strelka/render/common.h>
#include "OptixBuffer.h"

struct Texture;

namespace oka
{

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle ias_handle;
    CUdeviceptr d_instances = 0;
    size_t d_instances_size = 0;

    OptixModuleCompileOptions module_compile_options = {};
    OptixModule ptx_module = 0;
    OptixModule closest_hit_module = 0; // Loaded from OptixRender_closest_hit.cu.optixir
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipeline pipeline = 0;
    OptixModule m_catromCurveModule = 0;

    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup radiance_miss_group = 0;
    OptixProgramGroup occlusion_miss_group = 0;
    OptixProgramGroup radiance_default_hit_group = 0;
    std::vector<OptixProgramGroup> radiance_hit_groups;
    OptixProgramGroup occlusion_hit_group = 0;
    OptixProgramGroup light_hit_group = 0;
    CUstream stream = 0;
    Params params = {};
    Params prevParams = {};

    std::unique_ptr<OptixBuffer> mParamsBuffer;

    OptixShaderBindingTable sbt = {};
};

class OptiXRender : public Render
{
private:
    struct Mesh
    {
        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output_buffer = 0;
        ~Mesh()
        {
            CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
        }
    };

    struct Curve
    {
        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr d_gas_output_buffer = 0;
        ~Curve()
        {
            CUDA_CHECK(cudaFree((void*)d_gas_output_buffer));
        }
    };

    struct Instance
    {
        OptixInstance instance;
    };

    // Per-material data (host-side; GPU data uploaded to shared device buffers)
    struct Material
    {
        MaterialParams params; // Resolved material parameters (with texture indices set)
    };

    struct View
    {
        oka::Camera::Matrices mCamMatrices;
    };

    struct DeviceSkinningPtrs
    {
        sutil::Matrix4x4* d_jointMats = nullptr;
        ~DeviceSkinningPtrs()
        {
            if (d_jointMats)
                cudaFree(d_jointMats);
        }
    };
    DeviceSkinningPtrs mSkinningPtrs;
    std::vector<int> mJointMatOffsets;

    std::vector<oka::Instance> mPrevInstances;

    View mPrevView;

    PathTracerState mState;
    bool mEnableValidation;
    bool mEnableMotionBlur;

    // Previous-frame settings for change detection (replaces static locals in render())
    uint32_t mPrevRectLightSamplingMethod = 0;
    bool mPrevEnableAccumulation = false;
    uint32_t mPrevSspTotal = 0;

    // Device buffers for per-material data (indexed by materialId)
    std::unique_ptr<OptixBuffer> mMaterialParamsBuffer; // MaterialParams[] on device
    uint32_t mMaterialCount = 0;

    void allocJointMatrices();
    std::unique_ptr<Mesh> createMesh(const oka::Mesh& mesh);
    void updateMesh(const oka::Mesh& mesh, int optixMeshesId);
    std::unique_ptr<Curve> createCurve(const oka::Curve& curve);
    bool compactAccel(CUdeviceptr& buffer, OptixTraversableHandle& handle, CUdeviceptr result, size_t outputSizeInBytes);

    std::vector<std::unique_ptr<Mesh>> mOptixMeshes;
    std::vector<std::unique_ptr<Curve>> mOptixCurves;

    std::unique_ptr<OptixBuffer> mVertexBuffer;
    std::unique_ptr<OptixBuffer> mPrevVertexBuffer;
    const int NUM_MOTION_KEYS = 2;
    std::unique_ptr<OptixBuffer> mVertexSkinDataBuffer;
    std::unique_ptr<OptixBuffer> mIndexBuffer;
    std::unique_ptr<OptixBuffer> mLightBuffer;
    // TODO: move to raii buffers
    std::unique_ptr<OptixBuffer> mPointsBuffer;
    std::unique_ptr<OptixBuffer> mWidthsBuffer;

    std::vector<std::shared_ptr<OptixBuffer>> mMotionTransformBuffers; // used for motion blur

    std::unique_ptr<OptixBuffer> mTlasBuffer;
    
    std::unique_ptr<OptixBuffer> mTexturesDataBuffer; // Consolidated GPU texture object array

    // Temporary buffers for GAS building
    // These buffers are reused across multiple GAS builds to reduce allocations
    // They are automatically resized if needed but never shrink
    std::unique_ptr<OptixBuffer> mTempAccelBuffer;        // Temporary buffer for acceleration structure building
    std::unique_ptr<OptixBuffer> mCompactedSizeBuffer;  // Buffer for storing compaction size results
    std::unique_ptr<OptixBuffer> mSegmentIndicesBuffer; // Buffer for curve segment indices

    void createVertexBuffer();
    void createPrevBuffers();
    void createVertexSkinDataBuffer();
    void createIndexBuffer();

    // curve utils
    void createPointsBuffer();
    void createWidthsBuffer();

    void createLightBuffer();

    Texture loadTextureFromFile(const std::string& fileName);

    bool createOptixMaterials();
    void destroyTextures();

    std::vector<Material> mMaterials;

    // Texture resource tracking for cleanup
    std::vector<cudaArray_t> mTextureArrays;
    std::vector<cudaTextureObject_t> mTextureObjects;

    void updatePathtracerParams(const uint32_t width, const uint32_t height);

public:
    OptiXRender(/* args */);
    ~OptiXRender();

    void init() override;
    void render(Buffer* output_buffer) override;
    Buffer* createBuffer(const BufferDesc& desc) override;

    void applySkinning();
    void createContext();
    void createBottomLevelAccelerationStructures();
    void updateBottomLevelAccelerationStructures();
    void createTopLevelAccelerationStructure();
    void updateTopLevelAccelerationStructure();
    void resolveInstanceGeometry(OptixInstance& oi, const oka::Instance& instance) const;
    void uploadInstancesToDevice(const std::vector<OptixInstance>& optixInstances);
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSbt();

};

} // namespace oka
