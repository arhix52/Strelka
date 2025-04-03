#pragma once

#include "render.h"

#include <optix.h>

#include "OptixRenderParams.h"

#include <scene/scene.h>

#include <cuda_checks.h>
#include "common.h"
#include "OptixBuffer.h"

#include <materialmanager.h>

struct Texture;

namespace oka
{

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle ias_handle;
    CUdeviceptr d_instances = 0;
    size_t d_instances_size = 0;

    OptixModuleCompileOptions module_compile_options = {};
    OptixModule ptx_module = 0;
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

    // optix material
    struct Material
    {
        OptixProgramGroup programGroup;
        CUdeviceptr d_argData = 0;
        size_t d_argDataSize = 0;
        CUdeviceptr d_roData = 0;
        size_t d_roSize = 0;
        CUdeviceptr d_textureHandler;
    };

    struct View
    {
        oka::Camera::Matrices mCamMatrices;
    };

    View mPrevView;

    PathTracerState mState;
    bool mEnableValidation;

    Mesh* createMesh(const oka::Mesh& mesh);
    Curve* createCurve(const oka::Curve& curve);
    bool compactAccel(CUdeviceptr& buffer, OptixTraversableHandle& handle, CUdeviceptr result, size_t outputSizeInBytes);

    std::vector<std::unique_ptr<Mesh>> mOptixMeshes;
    std::vector<std::unique_ptr<Curve>> mOptixCurves;

    std::unique_ptr<OptixBuffer> mVertexBuffer;
    std::unique_ptr<OptixBuffer> mIndexBuffer;
    std::unique_ptr<OptixBuffer> mLightBuffer;
    // TODO: move to raii buffers
    std::unique_ptr<OptixBuffer> mPointsBuffer;
    std::unique_ptr<OptixBuffer> mWidthsBuffer;

    std::unique_ptr<OptixBuffer> mMaterialRoDataBuffer;
    std::unique_ptr<OptixBuffer> mMaterialArgDataBuffer;
    std::unique_ptr<OptixBuffer> mTexturesHandlerBuffer;
    std::unique_ptr<OptixBuffer> mTexturesDataBuffer;

    void createVertexBuffer();
    void createIndexBuffer();

    // curve utils
    void createPointsBuffer();
    void createWidthsBuffer();

    void createLightBuffer();

    Texture loadTextureFromFile(const std::string& fileName);

    bool createOptixMaterials();
    Material& getMaterial(int id);

    MaterialManager mMaterialManager;
    std::vector<Material> mMaterials;

    void updatePathtracerParams(const uint32_t width, const uint32_t height);

public:
    OptiXRender(/* args */);
    ~OptiXRender();

    void init() override;
    void render(Buffer* output_buffer) override;
    Buffer* createBuffer(const BufferDesc& desc) override;

    void createContext();
    void createBottomLevelAccelerationStructures();
    void createTopLevelAccelerationStructure();
    void createModule();
    void createProgramGroups();
    void createPipeline();
    void createSbt();

    OptixProgramGroup createRadianceClosestHitProgramGroup(PathTracerState& state,
                                                           char const* module_code,
                                                           size_t module_size);
};

} // namespace oka
