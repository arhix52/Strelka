#include "OptixRender.h"

#include "OptixBuffer.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <glm/glm.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/ext/matrix_relational.hpp>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <vector_types.h>
#include <vector_functions.h>

#include <sutil/vec_math_adv.h>
#include <sutil/Matrix.h>

#include "texture_support_cuda.h"

#include <filesystem>
#include <array>
#include <string>
#include <fstream>
#include <memory>
#include <cstdlib>

#include <log.h>
#include <paths.h>

#include <postprocessing/Tonemappers.h>
#include <postprocessing/DenoiseGuides.h>
#include <skinning/skinning.h>
#include <env_cdf.h>

#include <strelka/render/Camera.h>

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    switch (level)
    {
    case 1:
        STRELKA_FATAL("OptiX [{0}]: {1}", tag, message);
        break;
    case 2:
        STRELKA_ERROR("OptiX [{0}]: {1}", tag, message);
        break;
    case 3:
        STRELKA_WARNING("OptiX [{0}]: {1}", tag, message);
        break;
    case 4:
        STRELKA_INFO("OptiX [{0}]: {1}", tag, message);
        break;
    default:
        break;
    }
}

static inline void optixCheck(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        const char* errorName = optixGetErrorName(res);
        const char* errorString = optixGetErrorString(res);
        STRELKA_ERROR("OptiX call {0} failed: {1}:{2} with [{3}] - [{4}]", call, file, line, errorName, errorString);
        std::abort();
    }
}

static inline void optixCheckLog(OptixResult res,
                                 const char* log,
                                 size_t sizeof_log,
                                 size_t sizeof_log_returned,
                                 const char* call,
                                 const char* file,
                                 unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        const char* errorName = optixGetErrorName(res);
        const char* errorString = optixGetErrorString(res);
        // OptiX reports how much log it wanted to write. Saying so matters here:
        // a module that fails to compile produces far more than the 16 KB buffer
        // holds, and silently printing the first 16 KB has sent people looking at
        // the wrong error more than once.
        const char* truncated = (sizeof_log_returned > sizeof_log) ? " [log truncated]" : "";
        STRELKA_FATAL("OptiX call {0} failed: {1}:{2} : result={3} ({4}) log={5}{6}", call, file, line, errorName,
                      errorString, log, truncated);
        std::abort();
    }
}

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------
#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call) optixCheckLog(call, log, sizeof(log), sizeof_log, #call, __FILE__, __LINE__)


using namespace oka;
namespace fs = std::filesystem;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static bool readSourceFile(std::string& str, const fs::path& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

OptiXRender::OptiXRender() = default;

OptiXRender::~OptiXRender()
{
    // Destroy texture objects and arrays
    destroyTextures();

    // Free SBT records
    if (mState.sbt.raygenRecord)
        cudaFree(reinterpret_cast<void*>(mState.sbt.raygenRecord));
    if (mState.sbt.missRecordBase)
        cudaFree(reinterpret_cast<void*>(mState.sbt.missRecordBase));
    if (mState.sbt.hitgroupRecordBase)
        cudaFree(reinterpret_cast<void*>(mState.sbt.hitgroupRecordBase));

    // Destroy pipeline
    if (mState.pipeline)
        optixPipelineDestroy(mState.pipeline);

    // Destroy program groups
    if (mState.raygen_prog_group)
        optixProgramGroupDestroy(mState.raygen_prog_group);
    if (mState.radiance_miss_group)
        optixProgramGroupDestroy(mState.radiance_miss_group);
    if (mState.occlusion_miss_group)
        optixProgramGroupDestroy(mState.occlusion_miss_group);
    if (mState.radiance_default_hit_group)
        optixProgramGroupDestroy(mState.radiance_default_hit_group);
    for (auto& pg : mState.radiance_hit_groups)
        if (pg) optixProgramGroupDestroy(pg);
    if (mState.occlusion_hit_group)
        optixProgramGroupDestroy(mState.occlusion_hit_group);
    if (mState.light_hit_group)
        optixProgramGroupDestroy(mState.light_hit_group);

    // Destroy modules
    if (mState.ptx_module)
        optixModuleDestroy(mState.ptx_module);
    if (mState.closest_hit_module)
        optixModuleDestroy(mState.closest_hit_module);
    if (mState.m_catromCurveModule)
        optixModuleDestroy(mState.m_catromCurveModule);

    // Free raw device pointers in Params
    if (mState.params.accum)
        cudaFree(mState.params.accum);
    if (mState.params.diffuse)
        cudaFree(mState.params.diffuse);
    if (mState.params.diffuseCounter)
        cudaFree(mState.params.diffuseCounter);
    if (mState.params.specular)
        cudaFree(mState.params.specular);
    if (mState.params.specularCounter)
        cudaFree(mState.params.specularCounter);

    // Free instance device memory
    if (mState.d_instances)
        cudaFree(reinterpret_cast<void*>(mState.d_instances));

    // Destroy CUDA stream
    if (mState.stream)
        cudaStreamDestroy(mState.stream);

    // Destroy OptiX device context (must be last)
    if (mState.context)
        optixDeviceContextDestroy(mState.context);
}

void OptiXRender::createContext()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamCreate(&mState.stream));

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    if (mEnableValidation)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        options.logCallbackLevel = 4;
    }
    else
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        options.logCallbackLevel = 2; // error
    }
    CUcontext cu_ctx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &mState.context));

    mState.mParamsBuffer.reset(new OptixBuffer(sizeof(Params)));
}

bool OptiXRender::compactAccel(CUdeviceptr& buffer,
                               OptixTraversableHandle& handle,
                               CUdeviceptr result,
                               size_t outputSizeInBytes)
{
    // Get compacted size from device
    size_t compactedSize;
    CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)result, sizeof(size_t), cudaMemcpyDeviceToHost));

    // Only compact if it saves space
    if (compactedSize >= outputSizeInBytes)
    {
        return false;
    }

    // Allocate compacted buffer
    CUdeviceptr compactedBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedBuffer), compactedSize));

    // Compact acceleration structure into new buffer
    OPTIX_CHECK(optixAccelCompact(mState.context, 0, handle, compactedBuffer, compactedSize, &handle));

    // Free original buffer and update pointer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
    buffer = compactedBuffer;

    return true;
}

std::unique_ptr<OptiXRender::Curve> OptiXRender::createCurve(const oka::Curve& curve)
{
    auto rcurve = std::make_unique<Curve>();
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t pointsCount = mScene->getCurvesPoint().size(); // total points count in points buffer
    const int degree = 3;
    const uint32_t numCurves = curve.mVertexCountsCount;

    std::vector<int> segmentIndices;
    uint32_t offsetInsideCurveArray = 0;
    for (uint32_t curveIndex = 0; curveIndex < numCurves; ++curveIndex)
    {
        const std::vector<uint32_t>& vertexCounts = mScene->getCurvesVertexCounts();
        const uint32_t numControlPoints = vertexCounts[curve.mVertexCountsStart + curveIndex];
        const int segmentsCount = numControlPoints - degree;
        for (int i = 0; i < segmentsCount; ++i)
        {
            int index = curve.mPointsStart + offsetInsideCurveArray + i;
            segmentIndices.push_back(index);
        }
        offsetInsideCurveArray += numControlPoints;
    }

    const size_t segmentIndicesSize = sizeof(int) * segmentIndices.size();
    // Reuse existing buffer if large enough, otherwise allocate new one
    if (!mSegmentIndicesBuffer || mSegmentIndicesBuffer->size() < segmentIndicesSize)
    {
        mSegmentIndicesBuffer.reset(new OptixBuffer(segmentIndicesSize));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mSegmentIndicesBuffer->getPtr()), segmentIndices.data(),
                          segmentIndicesSize, cudaMemcpyHostToDevice));

    OptixBuildInput curve_input = {};
    curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
    switch (degree)
    {
    case 1:
        curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        break;
    case 2:
        curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
        break;
    case 3:
        curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        break;
    }

    curve_input.curveArray.numPrimitives = segmentIndices.size();
    CUdeviceptr vertexBuffers[] = { mPointsBuffer->getPtr() };
    curve_input.curveArray.vertexBuffers = vertexBuffers;
    curve_input.curveArray.numVertices = pointsCount;
    curve_input.curveArray.vertexStrideInBytes = sizeof(glm::float3);
    CUdeviceptr widthBuffers[] = { mWidthsBuffer->getPtr() };
    curve_input.curveArray.widthBuffers = widthBuffers;
    curve_input.curveArray.widthStrideInBytes = sizeof(float);
    curve_input.curveArray.normalBuffers = 0;
    curve_input.curveArray.normalStrideInBytes = 0;
    curve_input.curveArray.indexBuffer = mSegmentIndicesBuffer->getPtr();
    curve_input.curveArray.indexStrideInBytes = sizeof(int);
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curve_input.curveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &curve_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));

    // Reuse temporary buffers if large enough, otherwise allocate new ones
    // These buffers are kept alive and reused for future GAS builds
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < gas_buffer_sizes.tempSizeInBytes)
    {
        mTempAccelBuffer.reset(new OptixBuffer(gas_buffer_sizes.tempSizeInBytes));
    }
    if (!mCompactedSizeBuffer || mCompactedSizeBuffer->size() < sizeof(uint64_t))
    {
        mCompactedSizeBuffer.reset(new OptixBuffer(sizeof(uint64_t)));
    }

    CUdeviceptr d_gas_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = mCompactedSizeBuffer->getPtr();

    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &curve_input,
                                1, // num build inputs
                                mTempAccelBuffer->getPtr(), gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &rcurve->gas_handle,
                                &property, // emitted property list
                                1)); // num emitted properties

    compactAccel(d_gas_output_buffer, rcurve->gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);

    rcurve->d_gas_output_buffer = d_gas_output_buffer;
    return rcurve;
}

std::unique_ptr<OptiXRender::Mesh> OptiXRender::createMesh(const oka::Mesh& mesh)
{
    bool isSkeletal = mesh.isSkeletal;

    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;

    OptixAccelBuildOptions accel_options = {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = isSkeletal ?
        (OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE) :
        (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE);

    constexpr int PREV_VB = 0;
    constexpr int CURR_VB = 1;
    // vertexBuffer[0] - previous vertex state (t=0), vertexBuffer[1] - current vertex state (t=1)
    CUdeviceptr vertexBuffer[2];
    if (mEnableMotionBlur)
    {
        vertexBuffer[PREV_VB] = mPrevVertexBuffer->getPtr() + mesh.mVbOffset * sizeof(oka::Scene::Vertex);
    }
    else
    {
        vertexBuffer[PREV_VB] = 0;
    }
    vertexBuffer[CURR_VB] = mVertexBuffer->getPtr() + mesh.mVbOffset * sizeof(oka::Scene::Vertex);

    const CUdeviceptr indexBuffer = mIndexBuffer->getPtr() + mesh.mIndex * sizeof(uint32_t);

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mesh.mVertexCount;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(oka::Scene::Vertex);
    triangle_input.triangleArray.indexBuffer = indexBuffer;
    triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    triangle_input.triangleArray.numIndexTriplets = mesh.mCount / 3;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    if (mEnableMotionBlur && isSkeletal)
    {
        // Motion options
        OptixMotionOptions motion_options = {};
        motion_options.numKeys = NUM_MOTION_KEYS;
        motion_options.timeBegin = 0.0f;
        motion_options.timeEnd = 1.0f;
        motion_options.flags = OPTIX_MOTION_FLAG_NONE;
        accel_options.motionOptions = motion_options;

        triangle_input.triangleArray.vertexBuffers = vertexBuffer;
    }
    else
    {
        triangle_input.triangleArray.vertexBuffers = &vertexBuffer[CURR_VB];
    }

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    // Reuse temporary buffers if large enough, otherwise allocate new ones
    // These buffers are kept alive and reused for future GAS builds
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < gas_buffer_sizes.tempSizeInBytes)
    {
        mTempAccelBuffer.reset(new OptixBuffer(gas_buffer_sizes.tempSizeInBytes));
    }
    if (!mCompactedSizeBuffer || mCompactedSizeBuffer->size() < sizeof(uint64_t))
    {
        mCompactedSizeBuffer.reset(new OptixBuffer(sizeof(uint64_t)));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    if (isSkeletal)
    {
        OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &triangle_input, 1,
                                    mTempAccelBuffer->getPtr(), gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                    gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
    }
    else
    {
        OptixAccelEmitDesc property = {};
        property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        property.result = mCompactedSizeBuffer->getPtr();

        OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &triangle_input,
                                    1, // num build inputs
                                    mTempAccelBuffer->getPtr(), gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                    gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                    &property, // emitted property list
                                    1)); // num emitted properties

        compactAccel(d_gas_output_buffer, gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);
    }

    auto rmesh = std::make_unique<Mesh>();
    rmesh->d_gas_output_buffer = d_gas_output_buffer;
    rmesh->gas_handle = gas_handle;
    return rmesh;
}

void OptiXRender::createBottomLevelAccelerationStructures()
{
    // Clear existing acceleration structures
    mOptixMeshes.clear();
    mOptixCurves.clear();

    // Create BLAS for meshes
    const auto& meshes = mScene->getMeshes();
    mOptixMeshes.reserve(mScene->getMeshes().size());
    for (const auto& mesh : meshes)
    {
        mOptixMeshes.emplace_back(createMesh(mesh));
    }

    // Create BLAS for curves
    const auto& curves = mScene->getCurves();
    mOptixCurves.reserve(curves.size());
    for (const auto& curve : curves)
    {
        mOptixCurves.emplace_back(createCurve(curve));
    }
}

void OptiXRender::updateMesh(const oka::Mesh& mesh, int optixMeshesId)
{
    OptixTraversableHandle& gas_handle = mOptixMeshes[optixMeshesId]->gas_handle;
    CUdeviceptr& d_gas_output_buffer = mOptixMeshes[optixMeshesId]->d_gas_output_buffer;

    // Configure acceleration structure build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    const CUdeviceptr vertexBuffer = mVertexBuffer->getPtr() + mesh.mVbOffset * sizeof(oka::Scene::Vertex);
    const CUdeviceptr indexBuffer = mIndexBuffer->getPtr() + mesh.mIndex * sizeof(uint32_t);

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = mesh.mVertexCount;
    triangle_input.triangleArray.vertexBuffers = &vertexBuffer;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(oka::Scene::Vertex);
    triangle_input.triangleArray.indexBuffer = indexBuffer;
    triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
    triangle_input.triangleArray.numIndexTriplets = mesh.mCount / 3;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    // Calculate memory requirements
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &triangle_input, 1,
                                mTempAccelBuffer->getPtr(), gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
}

void OptiXRender::updateBottomLevelAccelerationStructures()
{
    // update BLAS for meshes
    const auto& meshes = mScene->getMeshes();
    int index = 0;
    for (const auto& mesh : meshes)
    {
        if (mesh.isSkeletal)
        {
            updateMesh(mesh, index);
        }
        ++index;
    }
}

void OptiXRender::resolveInstanceGeometry(OptixInstance& oi, const oka::Instance& instance) const
{
    switch (instance.type)
    {
    case oka::Instance::Type::eMesh:
        oi.traversableHandle = mOptixMeshes[instance.mMeshId]->gas_handle;
        oi.visibilityMask = GEOMETRY_MASK_TRIANGLE;
        break;
    case oka::Instance::Type::eCurve:
        oi.traversableHandle = mOptixCurves[instance.mCurveId]->gas_handle;
        oi.visibilityMask = GEOMETRY_MASK_CURVE;
        break;
    case oka::Instance::Type::eLight:
        oi.traversableHandle = mOptixMeshes[instance.mMeshId]->gas_handle;
        oi.visibilityMask = GEOMETRY_MASK_LIGHT;
        break;
    default:
        STRELKA_ERROR("Unknown instance type");
        std::abort();
        break;
    }
}

void OptiXRender::uploadInstancesToDevice(const std::vector<OptixInstance>& optixInstances)
{
    const size_t instancesSize = sizeof(OptixInstance) * optixInstances.size();
    if (instancesSize != mState.d_instances_size)
    {
        if (mState.d_instances)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mState.d_instances)));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.d_instances), instancesSize));
        mState.d_instances_size = instancesSize;
    }
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(mState.d_instances), optixInstances.data(), instancesSize, cudaMemcpyHostToDevice));
}

void OptiXRender::createTopLevelAccelerationStructure()
{
    mMotionTransformBuffers.clear();

    const std::vector<oka::Instance>& instances = mScene->getInstances();

    // Build OptixInstance array
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(instances.size());

    for (size_t instID = 0; instID < instances.size(); ++instID)
    {
        const auto& instance = instances[instID];
        OptixInstance oi = {};
        resolveInstanceGeometry(oi, instance);

        // If instance is animated, create linear matrix motion transform; else set transform directly
        if (mEnableMotionBlur && instance.isAnimated)
        {
            OptixMatrixMotionTransform matrixMotionTransform = {};
            OptixTraversableHandle matrixMotionTransformHandle;

            matrixMotionTransform.child = oi.traversableHandle;
            matrixMotionTransform.motionOptions.numKeys = NUM_MOTION_KEYS;
            matrixMotionTransform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
            matrixMotionTransform.motionOptions.timeBegin = 0.0f;
            matrixMotionTransform.motionOptions.timeEnd = 1.0f;

            memcpy(matrixMotionTransform.transform[0],
                   glm::value_ptr(glm::float3x4(glm::rowMajor4(mPrevInstances[instID].transform))), sizeof(float) * 12);
            memcpy(matrixMotionTransform.transform[1],
                   glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);

            auto motionTransformBuffer = std::make_shared<OptixBuffer>(sizeof(OptixMatrixMotionTransform));
            CUDA_CHECK(cudaMemcpy(motionTransformBuffer->getNativePtr(), &matrixMotionTransform,
                                  sizeof(OptixMatrixMotionTransform), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixConvertPointerToTraversableHandle(mState.context, motionTransformBuffer->getPtr(),
                                                               OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                                                               &matrixMotionTransformHandle));

            mMotionTransformBuffers.push_back(motionTransformBuffer);

            // No transform on the instance - the motion transform handles it
            const float trafoIdentity[12] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
            memcpy(oi.transform, trafoIdentity, sizeof(float) * 12);

            oi.traversableHandle = matrixMotionTransformHandle;
        }
        else
        {
            memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);
        }

        oi.sbtOffset = static_cast<unsigned int>(optixInstances.size() * RAY_TYPE_COUNT);
        optixInstances.push_back(oi);
    }

    uploadInstancesToDevice(optixInstances);

    // Setup IAS build input
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = static_cast<int>(optixInstances.size());

    // Setup IAS build options
    OptixAccelBuildOptions iasOptions = {};
    if (mEnableMotionBlur)
        iasOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    else
        iasOptions.buildFlags =
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    // Allocate buffers
    size_t outputBufferSize = iasBufferSizes.outputSizeInBytes;
    if (!mTlasBuffer || mTlasBuffer->size() < outputBufferSize)
    {
        mTlasBuffer.reset(new OptixBuffer(outputBufferSize));
    }

    // Reuse temporary buffers if large enough, otherwise allocate new ones
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < iasBufferSizes.tempSizeInBytes)
    {
        mTempAccelBuffer.reset(new OptixBuffer(iasBufferSizes.tempSizeInBytes));
    }
    if (!mCompactedSizeBuffer || mCompactedSizeBuffer->size() < sizeof(uint64_t))
    {
        mCompactedSizeBuffer.reset(new OptixBuffer(sizeof(uint64_t)));
    }

    // Setup compaction property
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = mCompactedSizeBuffer->getPtr();

    // Build IAS
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &iasOptions, &iasInput,
                                1, // num build inputs
                                mTempAccelBuffer->getPtr(), iasBufferSizes.tempSizeInBytes, mTlasBuffer->getPtr(),
                                outputBufferSize, &mState.ias_handle, &property,
                                1 // num emitted properties
                                ));

    // Compact acceleration structure
    size_t compactedSize;
    CUDA_CHECK(cudaMemcpy(&compactedSize, (void*)property.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    // Only compact if it saves space
    if (compactedSize < outputBufferSize)
    {
        // Create new buffer for compacted data
        std::unique_ptr<OptixBuffer> compactedBuffer(new OptixBuffer(compactedSize));

        // Compact acceleration structure into new buffer
        OPTIX_CHECK(optixAccelCompact(mState.context, 0, mState.ias_handle,
                                     compactedBuffer->getPtr(), compactedSize, &mState.ias_handle));

        // Replace old buffer with compacted one
        mTlasBuffer = std::move(compactedBuffer);
    }
}

void oka::OptiXRender::updateTopLevelAccelerationStructure()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();

    // Build OptixInstance array (no motion blur for refit)
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(instances.size());

    for (const auto& instance : instances)
    {
        OptixInstance oi = {};
        resolveInstanceGeometry(oi, instance);
        memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);
        oi.sbtOffset = static_cast<unsigned int>(optixInstances.size() * RAY_TYPE_COUNT);
        optixInstances.push_back(oi);
    }

    uploadInstancesToDevice(optixInstances);

    // Setup IAS build (refit) input
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = static_cast<int>(optixInstances.size());

    // Setup IAS build (refit) options
    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    // Reuse temporary buffers if large enough, otherwise allocate new ones
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < iasBufferSizes.tempSizeInBytes)
    {
        mTempAccelBuffer.reset(new OptixBuffer(iasBufferSizes.tempSizeInBytes));
    }

    // Build (refit) IAS
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &iasOptions, &iasInput,
                                1, // num build inputs
                                mTempAccelBuffer->getPtr(), iasBufferSizes.tempSizeInBytes, mTlasBuffer->getPtr(),
                                mTlasBuffer->size(), &mState.ias_handle, nullptr,
                                0 // num emitted properties
                                ));
}

void OptiXRender::createModule()
{
    // Setup module compilation options
    OptixModuleCompileOptions moduleOptions = {};
    if (mEnableValidation)
    {
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    }
    else
    {
        moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    }

    // Setup pipeline compilation options
    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.usesMotionBlur = mEnableMotionBlur;
    pipelineOptions.traversableGraphFlags = mEnableMotionBlur ?
                                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY :
                                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = 2;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.exceptionFlags =
        mEnableValidation ?
            (OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW) :
            OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
    pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(Params);

    // Load and create main module (raygen, miss, occlusion, light hit).
    //
    // Resolved against the executable, not the working directory: the build system
    // drops the OPTIXIR next to the binary (see CMAKE_RUNTIME_OUTPUT_DIRECTORY in
    // the root CMakeLists), and resolving against the CWD meant the renderer only
    // started when launched from the build root -- so a debugger, Finder, or any
    // harness that cd'd elsewhere got an unexplained "failed to open" instead.
    // This is the same resolveResourcePath() the Metal backend uses for metallibs.
    const fs::path optixPath = oka::resolveResourcePath("optix/strelka_shaders_generated_OptixRender.cu.optixir");
    std::string optixSource;
    readSourceFile(optixSource, optixPath);

    char log[16384];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(mState.context, &moduleOptions, &pipelineOptions, optixSource.c_str(),
                                      optixSource.size(), log, &sizeof_log, &mState.ptx_module));

    // Load closest-hit module (radiance closest hit with BSDF evaluation)
    const fs::path closestHitPath =
        oka::resolveResourcePath("optix/strelka_shaders_generated_OptixRender_closest_hit.cu.optixir");
    std::string closestHitSource;
    readSourceFile(closestHitSource, closestHitPath);

    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(mState.context, &moduleOptions, &pipelineOptions, closestHitSource.c_str(),
                                      closestHitSource.size(), log, &sizeof_log, &mState.closest_hit_module));

    // Store options for later use
    mState.pipeline_compile_options = pipelineOptions;
    mState.module_compile_options = moduleOptions;

    // Create curve module
    OptixBuiltinISOptions builtinOptions = {};
    builtinOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    builtinOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;

    OPTIX_CHECK(optixBuiltinISModuleGet(
        mState.context, &moduleOptions, &pipelineOptions, &builtinOptions, &mState.m_catromCurveModule));
}

void OptiXRender::createProgramGroups()
{
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = mState.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &raygen_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = mState.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &miss_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.radiance_miss_group));

    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = nullptr; // NULL miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &miss_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.occlusion_miss_group));

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mState.closest_hit_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hit_prog_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr; // auto for built-in
    sizeof_log = sizeof(log);
    OptixProgramGroup radiance_hit_group;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &radiance_hit_group));
    mState.radiance_default_hit_group = radiance_hit_group;

    OptixProgramGroupDesc light_hit_prog_group_desc = {};
    light_hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    light_hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    light_hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__light";
    sizeof_log = sizeof(log);
    OptixProgramGroup light_hit_group;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &light_hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &light_hit_group));
    mState.light_hit_group = light_hit_group;

    memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";

    hit_prog_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = 0; // automatically supplied for built-in module

    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                        1, // num program groups
                                        &program_group_options, log, &sizeof_log, &mState.occlusion_hit_group));
}

void OptiXRender::createPipeline()
{
    OptixPipeline pipeline = nullptr;
    const uint32_t max_trace_depth = 2;
    std::vector<OptixProgramGroup> program_groups = {};

    program_groups.push_back(mState.raygen_prog_group);
    program_groups.push_back(mState.radiance_miss_group);
    program_groups.push_back(mState.radiance_default_hit_group);
    program_groups.push_back(mState.occlusion_miss_group);
    program_groups.push_back(mState.occlusion_hit_group);
    program_groups.push_back(mState.light_hit_group);

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(mState.context, &mState.pipeline_compile_options, &pipeline_link_options,
                                        program_groups.data(), program_groups.size(), log, &sizeof_log, &pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDepth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    int maxTraversableDepth = mEnableMotionBlur ? 3 : 2;
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          maxTraversableDepth));
    mState.pipeline = pipeline;
}

void OptiXRender::createSbt()
{
    // Free previous SBT records if they exist
    if (mState.sbt.raygenRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mState.sbt.raygenRecord)));
    if (mState.sbt.missRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mState.sbt.missRecordBase)));
    if (mState.sbt.hitgroupRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mState.sbt.hitgroupRecordBase)));
    mState.sbt = {};

    // Create raygen record
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));

    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    // Create miss records
    CUdeviceptr miss_record;
    const size_t miss_record_size = sizeof(MissSbtRecord) * RAY_TYPE_COUNT;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));

    std::vector<MissSbtRecord> miss_records(RAY_TYPE_COUNT);

    // Radiance miss record
    MissSbtRecord& radiance_miss = miss_records[RAY_TYPE_RADIANCE];
    radiance_miss.data.bg_color = { 0.0f, 0.0f, 0.0f };
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_miss_group, &radiance_miss));

    // Occlusion miss record
    MissSbtRecord& occlusion_miss = miss_records[RAY_TYPE_OCCLUSION];
    // Named rather than brace-initialised: MissData wraps a float3, so `{0,0,0}`
    // needs a nested brace and only ever compiled by accident.
    occlusion_miss.data.bg_color = make_float3(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_miss_group, &occlusion_miss));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(miss_record), miss_records.data(), miss_record_size, cudaMemcpyHostToDevice));

    // Create hit group records
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    const uint32_t hit_group_count = std::max(1u, static_cast<uint32_t>(instances.size())) * RAY_TYPE_COUNT;
    const size_t hit_group_size = sizeof(HitGroupSbtRecord) * hit_group_count;

    std::vector<HitGroupSbtRecord> hit_groups(hit_group_count);
    CUdeviceptr hit_group_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hit_group_record), hit_group_size));

    if (instances.empty())
    {
        // Create default hit groups when no instances exist
        HitGroupSbtRecord& radiance_hit = hit_groups[RAY_TYPE_RADIANCE];
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_default_hit_group, &radiance_hit));

        HitGroupSbtRecord& occlusion_hit = hit_groups[RAY_TYPE_OCCLUSION];
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_hit_group, &occlusion_hit));
    }
    else
    {
        // Create hit groups for each instance
        const std::vector<oka::Mesh>& meshes = mScene->getMeshes();

        for (size_t i = 0; i < instances.size(); i++)
        {
            const oka::Instance& instance = instances[i];
            const uint32_t material_idx = (instance.mMaterialId == (uint32_t)-1) ? 0u : instance.mMaterialId;

            // Radiance hit group
            HitGroupSbtRecord& radiance_hit = hit_groups[i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE];

            if (instance.type == oka::Instance::Type::eLight)
            {
                radiance_hit.data.lightId = instance.mLightId;
                OPTIX_CHECK(optixSbtRecordPackHeader(mState.light_hit_group, &radiance_hit));
            }
            else
            {
                OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_default_hit_group, &radiance_hit));
                radiance_hit.data.lightId = -1;
            }

            // Material is looked up from device buffer by materialId
            radiance_hit.data.materialId = material_idx;

            // Set mesh data if applicable
            if (instance.type == oka::Instance::Type::eMesh)
            {
                const oka::Mesh& mesh = meshes[instance.mMeshId];
                radiance_hit.data.indexCount = mesh.mCount;
                radiance_hit.data.indexOffset = mesh.mIndex;
                radiance_hit.data.vertexOffset = mesh.mVbOffset;
            }

            // Occlusion hit group
            HitGroupSbtRecord& occlusion_hit = hit_groups[i * RAY_TYPE_COUNT + RAY_TYPE_OCCLUSION];
            OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_hit_group, &occlusion_hit));
        }
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(hit_group_record), hit_groups.data(), hit_group_size, cudaMemcpyHostToDevice));

    // Create final SBT
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase = hit_group_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = hit_group_count;

    mState.sbt = sbt;
}

void OptiXRender::updatePathtracerParams(const uint32_t width, const uint32_t height)
{
    bool needRealloc = false;
    if (mState.params.image_width != width || mState.params.image_height != height)
    {
        // new dimensions!
        needRealloc = true;
        // reset rendering
        getSharedContext().mSubframeIndex = 0;
        // The caller's buffer is the display image, and a resolution change is
        // exactly when the caller replaces it. Holding the old pointer would
        // have readDisplayTexture() read a freed allocation.
        mDisplayImage = nullptr;
        mDisplayWidth = 0;
        mDisplayHeight = 0;
    }
    mState.params.image_width = width;
    mState.params.image_height = height;
    if (needRealloc)
    {
        if (mState.params.accum)
            CUDA_CHECK(cudaFree((void*)mState.params.accum));
        if (mState.params.diffuse)
            CUDA_CHECK(cudaFree((void*)mState.params.diffuse));
        if (mState.params.diffuseCounter)
            CUDA_CHECK(cudaFree((void*)mState.params.diffuseCounter));
        if (mState.params.specular)
            CUDA_CHECK(cudaFree((void*)mState.params.specular));
        if (mState.params.specularCounter)
            CUDA_CHECK(cudaFree((void*)mState.params.specularCounter));
        const size_t frameSize = mState.params.image_width * mState.params.image_height;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.accum), frameSize * sizeof(float4)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.diffuse), frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMemset(mState.params.diffuse, 0, frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.diffuseCounter), frameSize * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(mState.params.diffuseCounter, 0, frameSize * sizeof(uint16_t)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.specular), frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMemset(mState.params.specular, 0, frameSize * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.specularCounter), frameSize * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(mState.params.specularCounter, 0, frameSize * sizeof(uint16_t)));
    }
}

// The production sizes the plan is told about have to be the sizes the device
// actually uses, or the buffers are allocated for a struct that is not the one
// being written. Same discipline as src/render/metal/integrator_buffer_sizes.h.
static_assert(sizeof(AovSample) == 64, "AovSample is written once per pixel per frame; keep an eye on the size");
static_assert(sizeof(AovSample) == sizeof(float) * 16, "AovSample must stay a whole number of floats to read back");

void OptiXRender::updateGuideBuffers(const DenoisePlan& plan)
{
    const bool wantGuides = plan.writeAov;
    if (!wantGuides)
    {
        // Held rather than freed only while they are in use: a scene that is not
        // denoising should not pay for a per-pixel record it never reads.
        mAovBuffer.reset();
        mDenoiseColorBuffer.reset();
        mDenoiseAlbedoBuffer.reset();
        mDenoiseNormalBuffer.reset();
        mDenoiseFlowBuffer.reset();
        mDenoiseFlowTrustBuffer.reset();
        mRenderImageBuffer.reset();
        return;
    }

    const DenoiseBufferLayout layout = denoiseBufferLayout(plan, sizeof(AovSample));

    auto ensure = [](std::unique_ptr<OptixBuffer>& buffer, size_t bytes) {
        if (!buffer || buffer->size() != bytes)
        {
            buffer = std::make_unique<OptixBuffer>(bytes);
        }
    };

    ensure(mAovBuffer, layout.aovBytes);
    ensure(mDenoiseColorBuffer, layout.colorBytes);
    ensure(mDenoiseAlbedoBuffer, layout.albedoBytes);
    ensure(mDenoiseNormalBuffer, layout.normalBytes);
    ensure(mDenoiseFlowBuffer, layout.flowBytes);
    ensure(mDenoiseFlowTrustBuffer, layout.flowTrustBytes);
    if (plan.upscale)
    {
        ensure(mRenderImageBuffer, layout.colorBytes);
    }
    else
    {
        mRenderImageBuffer.reset();
    }
}

bool OptiXRender::readDisplayTexture(std::vector<float>& out, uint32_t& width, uint32_t& height)
{
    if (mDisplayImage == nullptr || mDisplayWidth == 0 || mDisplayHeight == 0)
    {
        return false;
    }
    width = mDisplayWidth;
    height = mDisplayHeight;
    out.resize(static_cast<size_t>(width) * height * 4);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), mDisplayImage, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return true;
}

bool OptiXRender::readGuideTexture(Guide guide, std::vector<float>& out, uint32_t& width, uint32_t& height)
{
    // Unpacked on the host rather than by a kernel per guide. This is an
    // inspection path -- a few frames, by hand or by a test -- and a kernel per
    // channel would be nine more places for the packing to be got wrong.
    const uint32_t w = mDenoisePlan.renderWidth;
    const uint32_t h = mDenoisePlan.renderHeight;

    if (guide == Guide::Denoised)
    {
        if (!mDenoiser.hasOutput())
        {
            return false;
        }
        width = mDenoiser.outputWidth();
        height = mDenoiser.outputHeight();
        out.resize(static_cast<size_t>(width) * height * 4);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), reinterpret_cast<const void*>(mDenoiser.output()),
                              out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return true;
    }

    if (guide == Guide::Color)
    {
        if (!mDenoiseColorBuffer || w == 0 || h == 0)
        {
            return false;
        }
        width = w;
        height = h;
        out.resize(static_cast<size_t>(w) * h * 4);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), mDenoiseColorBuffer->getNativePtr(), out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        return true;
    }

    if (!mAovBuffer || w == 0 || h == 0)
    {
        return false;
    }

    std::vector<AovSample> records(static_cast<size_t>(w) * h);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(records.data(), mAovBuffer->getNativePtr(), records.size() * sizeof(AovSample),
                          cudaMemcpyDeviceToHost));

    width = w;
    height = h;
    out.assign(static_cast<size_t>(w) * h * 4, 0.0f);
    for (size_t i = 0; i < records.size(); ++i)
    {
        const AovSample& a = records[i];
        float* px = out.data() + i * 4;
        switch (guide)
        {
        case Guide::Depth:
            px[0] = a.depth;
            break;
        case Guide::Motion:
            px[0] = a.motionX;
            px[1] = a.motionY;
            break;
        case Guide::DiffuseAlbedo:
            px[0] = a.diffuseAlbedo.x;
            px[1] = a.diffuseAlbedo.y;
            px[2] = a.diffuseAlbedo.z;
            px[3] = 1.0f;
            break;
        case Guide::SpecularAlbedo:
            px[0] = a.specularAlbedo.x;
            px[1] = a.specularAlbedo.y;
            px[2] = a.specularAlbedo.z;
            px[3] = 1.0f;
            break;
        case Guide::Normal:
            px[0] = a.normal.x;
            px[1] = a.normal.y;
            px[2] = a.normal.z;
            break;
        case Guide::Roughness:
            px[0] = a.roughness;
            break;
        case Guide::SpecularHitDistance:
            px[0] = a.specularHitDistance;
            break;
        case Guide::Reactive:
            px[0] = a.reactive;
            break;
        default:
            return false;
        }
    }
    return true;
}

void OptiXRender::applySkinning()
{
    // joint matrices
    std::vector<glm::mat4> jointMat;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            auto jointCount = mScene->mSkines[node.skin].joints.size();
            std::vector<glm::mat4> currJointMats;
            mScene->computeJointMatrices(&currJointMats, jointCount, node.skin);

            jointMat.insert(jointMat.end(), currJointMats.begin(), currJointMats.end());
        }
    }

    size_t jointMatSize = jointMat.size();
    std::vector<sutil::Matrix4x4> cudaMatrices(jointMatSize);
    for (size_t i = 0; i < jointMatSize; ++i)
    {
        // glm is column-major, sutil::Matrix4x4 is row-major — transpose and copy
        const glm::mat4 transposed = glm::transpose(jointMat[i]);
        memcpy(cudaMatrices[i].getData(), glm::value_ptr(transposed), 16 * sizeof(float));
    }

    CUDA_CHECK(cudaMemcpy(mSkinningPtrs.d_jointMats, cudaMatrices.data(), jointMatSize * sizeof(sutil::Matrix4x4),
                          cudaMemcpyHostToDevice));

    // apply skinning
    int index = 0;
    int jointMatOffset = 0;
    if (mEnableMotionBlur)
        mVertexBuffer.swap(mPrevVertexBuffer);
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            if (index > 0)
            {
                jointMatOffset += mJointMatOffsets[index - 1];
            }
            index++;

            for (const auto instId : node.instanceIds)
            {
                auto& mesh = mScene->mMeshes[mScene->mInstances[instId].mMeshId];

                cuApplySkinning(256, mesh.mVbOffset, mesh.mSbOffset, reinterpret_cast<void*>(mVertexBuffer->getPtr()),
                                reinterpret_cast<void*>(mVertexSkinDataBuffer->getPtr()), mSkinningPtrs.d_jointMats,
                                jointMatOffset, mesh.mVertexCount);
            }
        }
    }
}

void OptiXRender::allocJointMatrices()
{
    size_t jointMatSize = 0;
    for (auto& node : mScene->mNodes)
    {
        if (node.skin != -1 && node.type == oka::Scene::Node::NodeType::mesh)
        {
            auto jointCount = mScene->mSkines[node.skin].joints.size();
            std::vector<glm::mat4> currJointMats;
            mScene->computeJointMatrices(&currJointMats, jointCount, node.skin);

            jointMatSize += currJointMats.size();
            mJointMatOffsets.push_back(currJointMats.size());
        }
    }
    CUDA_CHECK(cudaMalloc(&mSkinningPtrs.d_jointMats, jointMatSize * sizeof(sutil::Matrix4x4)));
}

void OptiXRender::render(Buffer* output)
{
    if (getSharedContext().mFrameNumber == 0)
    {
        createOptixMaterials();

        // Load environment map if specified
        const auto& envLight = mScene->getEnvLight();
        if (envLight.has_value() && !envLight->texturePath.empty())
        {
            const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
            const fs::path envTexPath = fs::path(resourcePathStr) / envLight->texturePath;
            loadEnvMap(envTexPath.string());
            mState.params.envMapIntensity = mEnvMapAutoScale * envLight->intensity;
            mState.params.envMapRotation = envLight->rotationY * (M_PI / 180.0f);
            mState.params.envMapColorTint = make_float3(envLight->color.x, envLight->color.y, envLight->color.z);
        }
        else
        {
            mState.params.hasEnvMap = false;
        }

        createVertexBuffer();
        if (mEnableMotionBlur)
        {
            createPrevBuffers();
        }
        createIndexBuffer();
        createVertexSkinDataBuffer();
        allocJointMatrices();
        // upload all curve data
        createPointsBuffer();
        createWidthsBuffer();
        createBottomLevelAccelerationStructures();
        createTopLevelAccelerationStructure();
        createSbt();
        createLightBuffer();
    }

    const ChangeBits changes = mScene->peekChanges();
    if (any(changes & ChangeBits::Lights))
    {
        createLightBuffer();
        createTopLevelAccelerationStructure();
    }
    if (any(changes & ChangeBits::Transforms))
    {
        // Instance transforms changed outside the animation path
        createTopLevelAccelerationStructure();
    }
    if (any(changes & (ChangeBits::Lights | ChangeBits::Transforms | ChangeBits::Materials)))
    {
        mScene->consumeChanges();
    }

    SettingsManager& settings = *getSettings();
    bool settingsChanged = false;
    bool animStateChanged = false;

    // Animation changes
    std::vector<oka::Scene::Animation>& animations = mScene->getAnimations();
    bool accelStructureDirty = false;
    if (mEnableMotionBlur)
        mPrevInstances.swap(mScene->getInstances());
    for (size_t i = 0; i < animations.size(); ++i)
    {
        const std::string scrollNameStr = "render/animation/anim" + std::to_string(i) + "/time";
        const char* scrollName = scrollNameStr.c_str();
        float currAnimTime = settings.getAs<float>(scrollName);

        const float EPSILON = 1e-6f; // 0.000001

        if (std::abs(animations[i].current - currAnimTime) > EPSILON)
        {
            animStateChanged = true;
            animations[i].current = currAnimTime;
            accelStructureDirty |= mScene->applyAnimation(i);
        }
    }

    // AS refit/reduild
    if (animStateChanged)
    {
        if (accelStructureDirty)
        {
            // blas refit/reduild + tlas rebuild
            applySkinning();
            if (mScene->blasUpdateCount < 10)
            {
                if (mEnableMotionBlur)
                    createBottomLevelAccelerationStructures();
                else
                    updateBottomLevelAccelerationStructures();
                mScene->blasUpdateCount++;
            }
            else
            {
                createBottomLevelAccelerationStructures();
                mScene->blasUpdateCount = 0;
            }
            createTopLevelAccelerationStructure();
            mScene->tlasUpdateCount = 0;
        }
        else
        {
            // tlas refit/reduild, blas untouched
            if (mScene->tlasUpdateCount < 10)
            {
                if (mEnableMotionBlur)
                    createTopLevelAccelerationStructure();
                else
                    updateTopLevelAccelerationStructure();
                mScene->tlasUpdateCount++;
            }
            else
            {
                createTopLevelAccelerationStructure();
                mScene->tlasUpdateCount = 0;
            }
        }
    }

    const uint32_t outputWidth = output->width();
    const uint32_t outputHeight = output->height();

    // What the frame is asked to produce, resolved before anything is sized:
    // upscaling renders at half the caller's resolution and lets the 2x model
    // put the missing pixels back, so the launch dimensions, the accumulation
    // buffers and the guides all follow from this rather than from the output.
    const uint32_t debugMode = settings.getAs<uint32_t>("render/pt/debug");
    const DenoisePlan plan = denoisePlan(settings.getAs<bool>("render/pt/denoise"),
                                         settings.getAs<bool>("render/pt/enableUpscale"),
                                         settings.getAs<uint32_t>("render/pt/upscaleMode"), debugMode, outputWidth,
                                         outputHeight);
    const bool planChanged = plan.kind != mDenoisePlan.kind || plan.renderWidth != mDenoisePlan.renderWidth ||
                             plan.renderHeight != mDenoisePlan.renderHeight || plan.writeAov != mDenoisePlan.writeAov;
    settingsChanged |= planChanged;
    mDenoisePlan = plan;

    const uint32_t width = plan.renderWidth;
    const uint32_t height = plan.renderHeight;

    updatePathtracerParams(width, height);
    updateGuideBuffers(plan);

    const uint32_t selectedCameraIdx = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCameraIdx < mScene->getCameraCount() ? selectedCameraIdx : 0);
    camera.updateAspectRatio(outputWidth / (float)outputHeight);
    camera.updateViewMatrix();

    View currView = {};

    currView.mCamMatrices = camera.matrices;

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
        getSharedContext().mSubframeIndex = 0;
    }

    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    settingsChanged |= (mPrevRectLightSamplingMethod != rectLightSamplingMethod);
    mPrevRectLightSamplingMethod = rectLightSamplingMethod;

    bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    settingsChanged |= (mPrevEnableAccumulation != enableAccumulation);
    mPrevEnableAccumulation = enableAccumulation;

    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    settingsChanged |= (mPrevSspTotal > sspTotal); // reset only if new spp less than already accumulated
    mPrevSspTotal = sspTotal;

    const float gamma = settings.getAs<float>("render/post/gamma");
    const ToneMapperType tonemapperType = (ToneMapperType)settings.getAs<uint32_t>("render/pt/tonemapperType");

    Params& params = mState.params;
    params.scene.vb = (Vertex*)mVertexBuffer->getPtr();

    if (mEnableMotionBlur)
        params.scene.vb_prev = (Vertex*)mPrevVertexBuffer->getPtr();
    params.enableMotionBlur = mEnableMotionBlur;
    settingsChanged |= (params.isMotionBlurVisible != settings.getAs<bool>("render/isMotionBlurVisible"));
    params.isMotionBlurVisible = settings.getAs<bool>("render/isMotionBlurVisible");

    if (settingsChanged || animStateChanged)
    {
        getSharedContext().mSubframeIndex = 0;
    }

    params.scene.ib = (uint32_t*)mIndexBuffer->getPtr();
    params.scene.lights = (UniformLight*)mLightBuffer->getPtr();
    params.scene.numLights = mScene->getLights().size();

    // When the 2x model is upscaling, the caller's buffer is twice the size the
    // path tracer runs at, so the tracer writes into its own buffer and the
    // denoiser is what fills the caller's.
    params.image = plan.upscale ? (float4*)mRenderImageBuffer->getNativePtr() :
                                  (float4*)((OptixBuffer*)output)->getNativePtr();
    params.samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    params.handle = mState.ias_handle;
    params.max_depth = settings.getAs<uint32_t>("render/pt/depth");

    params.rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    params.enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    params.debug = settings.getAs<uint32_t>("render/pt/debug");
    params.shadowRayTmin = settings.getAs<float>("render/pt/dev/shadowRayTmin");
    params.materialRayTmin = settings.getAs<float>("render/pt/dev/materialRayTmin");
    params.misHeuristic = settings.getAs<uint32_t>("render/pt/misHeuristic");

    memcpy(params.viewToWorld, glm::value_ptr(glm::transpose(glm::inverse(camera.matrices.view))),
           sizeof(params.viewToWorld));
    memcpy(params.clipToView, glm::value_ptr(glm::transpose(camera.matrices.invPerspective)), sizeof(params.clipToView));

    // World to clip, this frame's and the previous frame's. sutil::Matrix4x4 is
    // row major and glm is column major, hence the transpose -- the same one the
    // two matrices above take.
    const glm::mat4 worldToClip = camera.matrices.perspective * camera.matrices.view;
    const glm::mat4 prevWorldToClip = mPrevView.mCamMatrices.perspective * mPrevView.mCamMatrices.view;
    memcpy(params.worldToClip, glm::value_ptr(glm::transpose(worldToClip)), sizeof(params.worldToClip));
    memcpy(params.prevWorldToClip, glm::value_ptr(glm::transpose(prevWorldToClip)), sizeof(params.prevWorldToClip));

    // --- Guides ----------------------------------------------------------
    params.aov = mAovBuffer ? (AovSample*)mAovBuffer->getNativePtr() : nullptr;
    params.writeAov = plan.writeAov && params.aov != nullptr;
    const bool guidePrimaryHit = settings.getAs<uint32_t>("render/pt/guidePrimaryHit") != 0;
    if (mPrevGuidePrimaryHit != guidePrimaryHit)
    {
        // The reset check upstream has already run for this frame, so this one
        // resets directly rather than feeding a flag that nothing will read.
        getSharedContext().mSubframeIndex = 0;
        mPrevGuidePrimaryHit = guidePrimaryHit;
    }
    params.guidePrimaryHit = guidePrimaryHit;
    params.denoiseDepthMode = settings.getAs<uint32_t>("render/pt/denoiseDepthMode");
    // The first frame has no predecessor, and neither does the frame after a cut.
    // Telling the denoiser otherwise makes it reproject from an image that has
    // nothing to do with this one.
    params.hasPrevFramePose = !mResetTemporalHistory && getSharedContext().mFrameNumber > 0;

    // Depth of field params
    params.useDof = camera.useDof ? 1 : 0;
    params.focalDistance = camera.focalDistance;
    params.apertureBlades = camera.apertureBlades;
    params.bladeRotation = camera.bladeRotation;
    params.anamorphicRatio = camera.anamorphicRatio;
    params.shiftX = camera.shiftX;
    params.shiftY = camera.shiftY;
    params.lensRadius = camera.useDof ? camera.focalLengthMm / (2.0f * camera.fStopDof * 1000.0f) : 0.0f;

    params.subframe_index = getSharedContext().mSubframeIndex;
    // Photometric Units from iray documentation
    // Controls the sensitivity of the "camera film" and is expressed as an index; the ISO number of the film, also
    // known as "film speed." The higher this value, the greater the exposure. If this is set to a non-zero value,
    // "Photographic" mode is enabled. If this is set to 0, "Arbitrary" mode is enabled, and all color scaling is then
    // strictly defined by the value of cm^2 Factor.
    float filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    // The candela per meter square factor
    float cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    // The fractional aperture number; e.g., 11 means aperture "f/11." It adjusts the size of the opening of the "camera
    // iris" and is expressed as a ratio. The higher this value, the lower the exposure.
    float fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    // Controls the duration, in fractions of a second, that the "shutter" is open; e.g., the value 100 means that the
    // "shutter" is open for 1/100th of a second. The higher this value, the greater the exposure
    float shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
    // Specifies the main color temperature of the light sources; the color that will be mapped to "white" on output,
    // e.g., an incoming color of this hue/saturation will be mapped to grayscale, but its intensity will remain
    // unchanged. This is similar to white balance controls on digital cameras.
    float3 whitePoint{ 1.0f, 1.0f, 1.0f };
    float3 exposureValue = all(whitePoint) ? 1.0f / whitePoint : make_float3(1.0f);
    const float lum = dot(exposureValue, make_float3(0.299f, 0.587f, 0.114f));
    if (filmIso > 0.0f)
    {
        // See https://www.nayuki.io/page/the-photographic-exposure-equation
        exposureValue *= cm2_factor * filmIso / (shutterSpeed * fStop * fStop) / 100.0f;
    }
    else
    {
        exposureValue *= cm2_factor;
    }
    exposureValue /= lum;

    params.exposure = exposureValue;

    const uint32_t totalSpp = settings.getAs<uint32_t>("render/pt/sppTotal");
    const uint32_t samplesPerLaunch = settings.getAs<uint32_t>("render/pt/spp");
    const int32_t leftSpp = totalSpp - getSharedContext().mSubframeIndex;
    // if accumulation is off then launch selected samples per pixel
    uint32_t samplesThisLaunch = enableAccumulation ? std::min((int32_t)samplesPerLaunch, leftSpp) : samplesPerLaunch;
    // not to trace rays if there is no geometry
    if (mScene->getIndices().empty())
    {
        samplesThisLaunch = 0;
    }

    // The two single-hit debug views describe one surface, so one sample of it
    // is the whole answer and accumulating it says nothing new.
    if (params.debug == (uint32_t)DebugMode::eNormal || params.debug == (uint32_t)DebugMode::eMotionBlur)
    {
        samplesThisLaunch = 1;
        enableAccumulation = false;
    }
    // A guide view needs the guides written this frame, and they are only
    // written by a launch. Without this the view freezes at whatever the last
    // launch before convergence produced -- which looks exactly like a correct
    // static guide, and is how a stale record goes unnoticed.
    if (params.debug >= DEBUG_MODE_FIRST_AOV && samplesThisLaunch == 0 && !mScene->getIndices().empty())
    {
        samplesThisLaunch = 1;
    }

    params.samples_per_launch = samplesThisLaunch;
    params.enableAccumulation = enableAccumulation;
    params.maxSampleCount = totalSpp;

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(mState.mParamsBuffer->getPtr()), &params, sizeof(params), cudaMemcpyHostToDevice));

    if (samplesThisLaunch != 0)
    {
        // Launch OptiX path tracer
        OPTIX_CHECK(optixLaunch(mState.pipeline, mState.stream, mState.mParamsBuffer->getPtr(), sizeof(Params),
                                &mState.sbt, width, height,
                                /*depth=*/1));

        // Update subframe index for accumulation
        getSharedContext().mSubframeIndex =
            enableAccumulation ? getSharedContext().mSubframeIndex + samplesThisLaunch : 0;
    }
    else
    {
        // Nothing was traced this frame -- the render has converged, or the scene
        // is empty. Re-present the accumulated image so the caller still gets a
        // picture rather than whatever was in its buffer.
        if (params.debug == 0)
        {
            const size_t imageSize = mState.params.image_width * mState.params.image_height * sizeof(float4);
            CUDA_CHECK(cudaMemcpy(params.image, params.accum, imageSize, cudaMemcpyDeviceToDevice));
        }
    }

    // --- Denoise ---------------------------------------------------------
    //
    // Before tonemapping, on linear radiance: the network was trained on light
    // rather than on a display curve, and exposure is a viewing decision that
    // comes after it. The result goes back into the image the display and the
    // EXR writer read, which is what makes `render.denoise = true` mean
    // something in a headless run.
    float4* displayImage = (float4*)((OptixBuffer*)output)->getNativePtr();
    mDenoiserFallback = false;
    // A guide view is looked at instead of the denoised image, not through it,
    // so the network is not run -- and the plan handed over is the empty one, so
    // its state and scratch memory go back to the device while it is not needed.
    const bool runDenoiser = plan.enabled() && params.debug < DEBUG_MODE_FIRST_AOV;
    {
        const bool ready = mDenoiser.configure(mState.context, mState.stream, runDenoiser ? plan : DenoisePlan{});
        if (runDenoiser && ready)
        {
            if (mResetTemporalHistory)
            {
                mDenoiser.resetHistory();
            }
            resolveDenoiseGuides((const AovSample*)mAovBuffer->getNativePtr(), params.image, width, height,
                                 params.exposure, settings.getAs<float>("render/pt/denoiseFireflyClamp"),
                                 (float4*)mDenoiseColorBuffer->getNativePtr(),
                                 (float4*)mDenoiseAlbedoBuffer->getNativePtr(),
                                 (float4*)mDenoiseNormalBuffer->getNativePtr(),
                                 (float2*)mDenoiseFlowBuffer->getNativePtr(),
                                 (float*)mDenoiseFlowTrustBuffer->getNativePtr());
            const bool denoised =
                mDenoiser.denoise(mState.stream, mDenoiseColorBuffer->getPtr(), mDenoiseAlbedoBuffer->getPtr(),
                                  mDenoiseNormalBuffer->getPtr(), mDenoiseFlowBuffer->getPtr(),
                                  mDenoiseFlowTrustBuffer->getPtr());
            if (denoised)
            {
                copyDenoisedToImage((const float4*)mDenoiser.output(), displayImage, outputWidth, outputHeight);
            }
            else
            {
                mDenoiserFallback = true;
            }
        }
        else if (runDenoiser)
        {
            mDenoiserFallback = true;
        }
    }
    // An upscaling plan that could not run leaves the caller's buffer holding
    // nothing at all, since the tracer wrote a half-size image somewhere else.
    // A nearest-neighbour blow-up is not a good picture, but it is a picture of
    // the right scene at the right size, which a black frame is not.
    if (plan.upscale && (mDenoiserFallback || params.debug >= DEBUG_MODE_FIRST_AOV))
    {
        upscalePointSample(params.image, width, height, displayImage, outputWidth, outputHeight);
    }
    mResetTemporalHistory = false;

    // Apply tonemapping except for the single-hit debug views, which are already
    // in display units and would only be crushed by a curve.
    if (params.debug != (uint32_t)DebugMode::eNormal && params.debug != (uint32_t)DebugMode::eMotionBlur)
    {
        float maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");
        exposureValue *= maxEDR;
        tonemap(tonemapperType, exposureValue, gamma, displayImage, outputWidth, outputHeight);
    }

    mDisplayImage = displayImage;
    mDisplayWidth = outputWidth;
    mDisplayHeight = outputHeight;

    getSharedContext().mFrameNumber++;

    mPrevView = currView;
    mState.prevParams = mState.params;
}

void OptiXRender::init()
{
    mEnableValidation = getSettings()->getAs<bool>("render/enableValidation");
    mEnableMotionBlur = getSettings()->getAs<bool>("render/enableMotionBlur");

    // Add a default material (standard PBR, white)
    {
        oka::Scene::MaterialDescription defaultMaterial{};
        defaultMaterial.name = "default_material";
        defaultMaterial.params.material_type = MATERIAL_TYPE_STANDARD_PBR;
        defaultMaterial.params.base_color = {1.0f, 1.0f, 1.0f};
        defaultMaterial.params.roughness = 0.5f;
        defaultMaterial.params.metallic = 0.0f;
        defaultMaterial.params.ior = 1.5f;
        defaultMaterial.params.specular = 0.5f;
        defaultMaterial.params.normal_scale = 1.0f;
        defaultMaterial.params.occlusion_strength = 1.0f;
        defaultMaterial.params.alpha_cutoff = 0.5f;
        defaultMaterial.params.base_color_tex = -1;
        defaultMaterial.params.metallic_roughness_tex = -1;
        defaultMaterial.params.normal_tex = -1;
        defaultMaterial.params.emission_tex = -1;
        defaultMaterial.params.occlusion_tex = -1;
        defaultMaterial.params.transmission_tex = -1;
        mScene->addMaterial(defaultMaterial);
    }

    createContext();
    createModule();
    createProgramGroups();
    createPipeline();
}

Buffer* OptiXRender::createBuffer(const BufferDesc& desc)
{
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);

    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size));

    return new OptixBuffer(devicePtr, desc.format, desc.width, desc.height);
}

template <typename T>
void createOrUpdateBuffer(std::unique_ptr<OptixBuffer>& buffer, const std::vector<T>& data)
{
    const size_t bufferSize = data.size() * sizeof(T);

    if (buffer == nullptr)
    {
        buffer.reset(new OptixBuffer(bufferSize));
    }
    if (buffer->size() != bufferSize)
    {
        buffer->realloc(bufferSize);
    }
    if (bufferSize > 0)
    {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer->getPtr()), data.data(), bufferSize, cudaMemcpyHostToDevice));
    }
}

void OptiXRender::createPointsBuffer()
{
    const auto& scenePoints = mScene->getCurvesPoint();
    if (scenePoints.empty())
    {
        return;
    }

    // Convert glm points to CUDA float3 format
    std::vector<float3> devicePoints;
    devicePoints.reserve(scenePoints.size());
    for (const auto& p : scenePoints)
    {
        devicePoints.push_back(make_float3(p.x, p.y, p.z));
    }

    createOrUpdateBuffer(mPointsBuffer, devicePoints);
}

void OptiXRender::createWidthsBuffer()
{
    createOrUpdateBuffer(mWidthsBuffer, mScene->getCurvesWidths());
}

void OptiXRender::createVertexBuffer()
{
    createOrUpdateBuffer(mVertexBuffer, mScene->getVertices());
}

void OptiXRender::createPrevBuffers()
{
    mPrevInstances = mScene->getInstances();
    createOrUpdateBuffer(mPrevVertexBuffer, mScene->getVertices());
}

void OptiXRender::createVertexSkinDataBuffer()
{
    createOrUpdateBuffer(mVertexSkinDataBuffer, mScene->getVerticesSkinData());
}

void OptiXRender::createIndexBuffer()
{
    createOrUpdateBuffer(mIndexBuffer, mScene->getIndices());
}

void OptiXRender::createLightBuffer()
{
    createOrUpdateBuffer(mLightBuffer, mScene->getLights());
}

Texture OptiXRender::loadTextureFromFile(const std::string& fileName)
{
    int texWidth, texHeight, texChannels;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!data)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return Texture();
    }
    // TODO: add compression here to save gpu mem

    const void* dataPtr = data;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));

    cudaArray_t device_tex_array;
    CUDA_CHECK(cudaMallocArray(&device_tex_array, &channel_desc, texWidth, texHeight));

    CUDA_CHECK(cudaMemcpy2DToArray(device_tex_array, 0, 0, dataPtr, texWidth * sizeof(char) * 4,
                                   texWidth * sizeof(char) * 4, texHeight, cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = device_tex_array;

    // Create filtered texture object
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    cudaTextureAddressMode addr_mode = cudaAddressModeWrap;
    tex_desc.addressMode[0] = addr_mode;
    tex_desc.addressMode[1] = addr_mode;
    tex_desc.addressMode[2] = addr_mode;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    if (res_desc.resType == cudaResourceTypeMipmappedArray)
    {
        tex_desc.mipmapFilterMode = cudaFilterModeLinear;
        tex_desc.maxAnisotropy = 16;
        tex_desc.minMipmapLevelClamp = 0.f;
        tex_desc.maxMipmapLevelClamp = 1000.f; // default value in OpenGL
    }
    cudaTextureObject_t tex_obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));
    // Create unfiltered texture object if necessary (cube textures have no texel functions)
    cudaTextureObject_t tex_obj_unfilt = 0;
    // if (texture_shape != mi::neuraylib::ITarget_code::Texture_shape_cube)
    {
        // Use a black border for access outside of the texture
        tex_desc.addressMode[0] = cudaAddressModeBorder;
        tex_desc.addressMode[1] = cudaAddressModeBorder;
        tex_desc.addressMode[2] = cudaAddressModeBorder;
        tex_desc.filterMode = cudaFilterModePoint;

        CUDA_CHECK(cudaCreateTextureObject(&tex_obj_unfilt, &res_desc, &tex_desc, nullptr));
    }
    stbi_image_free(data);

    // Track resources for cleanup
    mTextureArrays.push_back(device_tex_array);
    mTextureObjects.push_back(tex_obj);
    mTextureObjects.push_back(tex_obj_unfilt);

    return Texture(tex_obj, tex_obj_unfilt, make_uint3(texWidth, texHeight, 1));
}

void OptiXRender::loadEnvMap(const std::string& texturePath)
{
    int width = 0, height = 0;
    float* pixelData = nullptr;

    const std::string ext = fs::path(texturePath).extension().string();
    if (ext == ".exr" || ext == ".EXR")
    {
        const char* err = nullptr;
        int ret = LoadEXR(&pixelData, &width, &height, texturePath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR env map: {} ({})", texturePath, err ? err : "unknown");
            if (err) FreeEXRErrorMessage(err);
            return;
        }
    }
    else
    {
        // HDR / LDR via stbi
        int channels = 0;
        pixelData = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (!pixelData)
        {
            STRELKA_ERROR("Failed to load env map: {}", texturePath);
            return;
        }
    }

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, width, height);

    // Create CUDA array and texture object for the env map (float4)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t envArray = nullptr;
    CUDA_CHECK(cudaMallocArray(&envArray, &channelDesc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(envArray, 0, 0, pixelData,
                                   width * sizeof(float4),
                                   width * sizeof(float4), height,
                                   cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = envArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t envTexObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&envTexObj, &resDesc, &texDesc, nullptr));

    // Track for cleanup
    mTextureArrays.push_back(envArray);
    mTextureObjects.push_back(envTexObj);

    // Upload raw pixel data to device for CDF construction
    const size_t rawDataSize = (size_t)width * height * 4 * sizeof(float);
    mEnvRawDataBuffer.reset(new OptixBuffer(rawDataSize));
    CUDA_CHECK(cudaMemcpy((void*)mEnvRawDataBuffer->getPtr(), pixelData, rawDataSize, cudaMemcpyHostToDevice));

    // Free host pixel data
    if (ext == ".exr" || ext == ".EXR")
        free(pixelData);
    else
        stbi_image_free(pixelData);

    // Allocate CDF buffers
    mEnvCdfXBuffer.reset(new OptixBuffer((size_t)width * height * sizeof(float)));
    mEnvCdfYBuffer.reset(new OptixBuffer((size_t)height * sizeof(float)));

    // Build CDF on GPU
    float totalPower = 0.0f;
    buildEnvMapCdf(
        (const float*)mEnvRawDataBuffer->getPtr(),
        width, height,
        (float*)mEnvCdfXBuffer->getPtr(),
        (float*)mEnvCdfYBuffer->getPtr(),
        &totalPower);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free raw data buffer (no longer needed after CDF build)
    mEnvRawDataBuffer.reset();

    // Store env map params
    mState.params.envMapTexture = envTexObj;
    mState.params.envCdfX = (float*)mEnvCdfXBuffer->getPtr();
    mState.params.envCdfY = (float*)mEnvCdfYBuffer->getPtr();
    mState.params.envMapWidth = width;
    mState.params.envMapHeight = height;
    mState.params.envMapTotalPower = totalPower;
    mState.params.hasEnvMap = true;
    mEnvMapLoaded = true;

    // Auto-calibrate env map to renderer's internal radiance scale.
    // Uncalibrated HDRIs have pixel values ~0.1-100 while the renderer's
    // light system uses intensities ~1000-10000. Scale the env map so its
    // average weighted luminance maps to a reference that produces correct
    // exposure with the photographic camera model.
    const float avgWeightedLum = totalPower / (float)(width * height);
    const float kCalibrationTarget = 1000.0f;
    mEnvMapAutoScale = (avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;

    STRELKA_INFO("Env map CDF built, total power: {}, avgLum: {:.4f}, autoScale: {:.1f}",
                 totalPower, avgWeightedLum, mEnvMapAutoScale);
}

void OptiXRender::destroyTextures()
{
    for (auto obj : mTextureObjects)
        if (obj) cudaDestroyTextureObject(obj);
    mTextureObjects.clear();

    for (auto arr : mTextureArrays)
        if (arr) cudaFreeArray(arr);
    mTextureArrays.clear();
}

bool OptiXRender::createOptixMaterials()
{
    const auto& matDescs = mScene->getMaterials();
    if (matDescs.empty())
    {
        STRELKA_WARNING("No materials in scene");
        return true;
    }

    const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
    const fs::path resourcePath(resourcePathStr);

    // Host-side storage for all texture objects (flat array: mat[0].tex[0..5], mat[1].tex[0..5], ...)
    std::vector<cudaTextureObject_t> allTexObjects(matDescs.size() * MAX_MATERIAL_TEXTURES, 0);

    // Cache: file path -> texture object (avoid loading the same file twice)
    std::unordered_map<std::string, cudaTextureObject_t> texCache;

    auto loadOrCacheTex = [&](const std::string& relPath) -> cudaTextureObject_t {
        if (relPath.empty())
            return 0;
        fs::path fullPath = resourcePath / relPath;
        std::string key = fullPath.string();
        auto it = texCache.find(key);
        if (it != texCache.end())
            return it->second;
        if (!fs::exists(fullPath))
        {
            STRELKA_WARNING("Texture not found: {}", key);
            texCache[key] = 0;
            return 0;
        }
        ::Texture tex = loadTextureFromFile(key);
        cudaTextureObject_t obj = tex.filtered_object;
        texCache[key] = obj;
        return obj;
    };

    mMaterials.resize(matDescs.size());

    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        const auto& desc = matDescs[i];
        cudaTextureObject_t* texSlots = &allTexObjects[i * MAX_MATERIAL_TEXTURES];

        // Copy material params (we'll update texture indices)
        MaterialParams params = desc.params;

        // Load textures from file paths, assign slots
        texSlots[0] = loadOrCacheTex(desc.baseColorTexPath);
        params.base_color_tex = texSlots[0] ? 0 : -1;

        texSlots[1] = loadOrCacheTex(desc.metallicRoughnessTexPath);
        params.metallic_roughness_tex = texSlots[1] ? 1 : -1;

        texSlots[2] = loadOrCacheTex(desc.normalTexPath);
        params.normal_tex = texSlots[2] ? 2 : -1;

        texSlots[3] = loadOrCacheTex(desc.emissionTexPath);
        params.emission_tex = texSlots[3] ? 3 : -1;

        texSlots[4] = loadOrCacheTex(desc.occlusionTexPath);
        params.occlusion_tex = texSlots[4] ? 4 : -1;

        // Slot 5 reserved for transmission texture (not yet populated by gltf loader)
        params.transmission_tex = -1;

        mMaterials[i].params = params;
    }

    // Upload all texture objects to GPU in one contiguous buffer
    const size_t totalTexSize = allTexObjects.size() * sizeof(cudaTextureObject_t);
    mTexturesDataBuffer.reset(new OptixBuffer(totalTexSize));
    CUDA_CHECK(cudaMemcpy(
        (void*)mTexturesDataBuffer->getPtr(), allTexObjects.data(), totalTexSize, cudaMemcpyHostToDevice));

    // Upload MaterialParams to a device buffer (indexed by materialId)
    std::vector<MaterialParams> allParams(matDescs.size());
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        allParams[i] = mMaterials[i].params;
    }
    const size_t paramsSize = allParams.size() * sizeof(MaterialParams);
    mMaterialParamsBuffer.reset(new OptixBuffer(paramsSize));
    CUDA_CHECK(cudaMemcpy(
        (void*)mMaterialParamsBuffer->getPtr(), allParams.data(), paramsSize, cudaMemcpyHostToDevice));
    mMaterialCount = matDescs.size();

    // Set device pointers in Params (will be uploaded each frame)
    mState.params.materials = (MaterialParams*)mMaterialParamsBuffer->getPtr();
    mState.params.materialTextures = (cudaTextureObject_t*)mTexturesDataBuffer->getPtr();

    STRELKA_INFO("Loaded {} materials ({} unique textures cached)", matDescs.size(), texCache.size());
    return true;
}
