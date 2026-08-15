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

#include <cuda_profiler_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <filesystem>
#include <array>
#include <string>
#include <fstream>
#include <memory>
#include <cstdlib>

#include <log.h>
#include <paths.h>

#include <postprocessing/Tonemappers.h>
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

namespace
{

double nowMilliseconds()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// What this process has on the device, as the driver accounts for it.
///
/// The counterpart of Metal's `MTLDevice::currentAllocatedSize`, and CUDA has no
/// equivalent in the runtime API: `cudaMemGetInfo` reports the whole board,
/// which on a machine that is running anything else is a number about that
/// machine rather than about this renderer. NVML does answer per process, so it
/// is loaded by name at runtime -- an optional diagnostic must not become a link
/// dependency, and a box without the management library still has to render.
///
/// Zero means "cannot say", which the Memory panel already understands: it
/// stops drawing the unaccounted slice rather than inventing one.
size_t deviceAllocatedBytes()
{
    // The v2 process record, which is what nvmlDeviceGetComputeRunningProcesses_v3
    // fills. Declared here rather than by including nvml.h so this stays a
    // runtime lookup with no build-time dependency at all.
    struct NvmlProcessInfoV2
    {
        unsigned int pid;
        unsigned long long usedGpuMemory;
        unsigned int gpuInstanceId;
        unsigned int computeInstanceId;
    };

    static void* handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle)
    {
        return 0;
    }

    using InitFn = int (*)();
    using HandleFn = int (*)(unsigned int, void**);
    using ProcFn = int (*)(void*, unsigned int*, NvmlProcessInfoV2*);

    static auto nvmlInit = reinterpret_cast<InitFn>(dlsym(handle, "nvmlInit_v2"));
    static auto nvmlGetHandle = reinterpret_cast<HandleFn>(dlsym(handle, "nvmlDeviceGetHandleByIndex_v2"));
    static auto nvmlGetProcs = reinterpret_cast<ProcFn>(dlsym(handle, "nvmlDeviceGetComputeRunningProcesses_v3"));
    if (!nvmlInit || !nvmlGetHandle || !nvmlGetProcs)
    {
        return 0;
    }

    static const bool initialised = (nvmlInit() == 0);
    if (!initialised)
    {
        return 0;
    }

    int cudaDevice = 0;
    if (cudaGetDevice(&cudaDevice) != cudaSuccess)
    {
        return 0;
    }
    void* device = nullptr;
    if (nvmlGetHandle(static_cast<unsigned int>(cudaDevice), &device) != 0)
    {
        return 0;
    }

    NvmlProcessInfoV2 procs[64] = {};
    unsigned int count = 64;
    if (nvmlGetProcs(device, &count, procs) != 0)
    {
        return 0;
    }
    const unsigned int self = static_cast<unsigned int>(getpid());
    for (unsigned int i = 0; i < count && i < 64; ++i)
    {
        if (procs[i].pid == self)
        {
            return static_cast<size_t>(procs[i].usedGpuMemory);
        }
    }
    return 0;
}

/// What the OS charges this process. VmRSS rather than VmSize: the mapped size
/// includes the whole device address space the driver reserves, which is
/// hundreds of gigabytes and says nothing about memory anybody is using.
size_t processFootprintBytes()
{
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key)
    {
        if (key == "VmRSS:")
        {
            size_t kb = 0;
            if (status >> kb)
            {
                return kb * 1024;
            }
            return 0;
        }
        status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

} // namespace

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

    if (mFrameStartEvent)
        cudaEventDestroy(mFrameStartEvent);
    if (mFrameStopEvent)
        cudaEventDestroy(mFrameStopEvent);

    // The editor's two output slots. Owned here because triggerRenderIfIdle
    // created them; the caller only ever borrows the ready one.
    delete mAsyncOutputBuffers[0];
    delete mAsyncOutputBuffers[1];

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

// Returns what the structure occupies afterwards -- the compacted size when
// compaction happened, the original otherwise. The caller records it, because a
// bare CUdeviceptr cannot be asked how big it is and the memory report refuses
// to estimate.
size_t OptiXRender::compactAccel(CUdeviceptr& buffer,
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
        return outputSizeInBytes;
    }

    // Allocate compacted buffer
    CUdeviceptr compactedBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedBuffer), compactedSize));

    // Compact acceleration structure into new buffer
    OPTIX_CHECK(optixAccelCompact(mState.context, 0, handle, compactedBuffer, compactedSize, &handle));

    // Free original buffer and update pointer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
    buffer = compactedBuffer;

    return compactedSize;
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

    rcurve->gas_bytes =
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

    size_t gasBytes = gas_buffer_sizes.outputSizeInBytes;
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

        gasBytes = compactAccel(d_gas_output_buffer, gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);
    }

    auto rmesh = std::make_unique<Mesh>();
    rmesh->d_gas_output_buffer = d_gas_output_buffer;
    rmesh->gas_handle = gas_handle;
    rmesh->gas_bytes = gasBytes;
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
    mSbtBytes = raygen_record_size + miss_record_size + hit_group_size;
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
    // One mark for the whole set. A per-mesh breadcrumb would say which mesh's
    // dispatch died, but the kernel is the same one for all of them and the
    // memset would cost more than the dispatch it follows.
    markStageSubmitted(optix::GpuStage::Skinning, 0);
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
    mSkinningPtrs.bytes = jointMatSize * sizeof(sutil::Matrix4x4);
    CUDA_CHECK(cudaMalloc(&mSkinningPtrs.d_jointMats, mSkinningPtrs.bytes));
}

void OptiXRender::render(Buffer* output)
{
    // Last frame's pair, if it has landed. Read here rather than at the end of
    // the frame that recorded it: at that point the work has only been enqueued,
    // so a read would either block -- turning an asynchronous submission into a
    // synchronous one -- or find nothing and leave the number at zero forever,
    // which is what it did.
    collectFrameTiming(false);

    // Before the build stages, not after them. They enqueue acceleration builds
    // and the environment CDF, and clearing the marks in between would wipe the
    // record of exactly the submissions a loading scene is most likely to fault
    // in.
    beginFrameBreadcrumbs();

    if (mScenePrep.isBuilding())
    {
        const bool complete = stepSceneBuild(output);
        // Each slice that moved the scene forward is one more thing worth
        // showing; the clock decides how many of them are worth a frame.
        mPublishClock.noteArrivals();

        metal::StreamReadiness readiness;
        readiness.hasOutputTargets = mState.params.accum != nullptr;
        readiness.hasEnvironment = mEnvMapLoaded || !mScene->getEnvLight().has_value();
        readiness.hasTopLevel = mState.ias_handle != 0;
        readiness.buildComplete = complete;

        const double nowMs = nowMilliseconds();
        const double intervalMs = getSettings()->getAs<float>("render/stream/publishIntervalMs");
        if (!canTracePartial(readiness) || !mPublishClock.shouldPublish(nowMs, intervalMs, complete))
        {
            // Nothing to trace yet, or nothing new since the last frame. The busy
            // flag has to come off here: triggerRenderIfIdle sets it before every
            // call, and a build stage that returns without submitting anything
            // leaves nothing behind to clear it, so the build would stall one
            // stage in.
            mRenderBusy.store(false, std::memory_order_release);
            return;
        }
        mPublishClock.notePublished(nowMs);
        // Time to first pixel is the number this whole path exists to move, so it
        // is reported rather than inferred from watching a window.
        if (!mReportedFirstPartialFrame)
        {
            mReportedFirstPartialFrame = true;
            STRELKA_INFO("First frame shown {:.0f} ms into the scene build (stage {})", nowMs - mBuildStartMs,
                         optix::buildStageName(mScenePrep.stage()));
        }
        // The scene under the accumulated image just changed, so what has been
        // accumulated is of a different scene.
        getSharedContext().mSubframeIndex = 0;
    }

    // Edits and animation are only meaningful once the scene is on the device.
    //
    // Both paths below rebuild the top level, and the top level resolves every
    // instance to the bottom level it names -- which, mid-build, is a bottom
    // level that does not exist yet. The load-time `createLight` alone leaves the
    // Lights bit set, so the very first partial frame of every scene went
    // straight into resolveInstanceGeometry with an empty mesh list and took the
    // editor down with it. The build's own Tail stage consumes these bits, so
    // nothing is dropped by waiting.
    const bool sceneOnDevice = !mScenePrep.isBuilding();

    const ChangeBits changes = sceneOnDevice ? mScene->peekChanges() : ChangeBits::None;
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
    if (mEnableMotionBlur && sceneOnDevice)
        mPrevInstances.swap(mScene->getInstances());
    for (size_t i = 0; sceneOnDevice && i < animations.size(); ++i)
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

    const uint32_t width = output->width();
    const uint32_t height = output->height();

    updatePathtracerParams(width, height);

    const uint32_t selectedCameraIdx = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCameraIdx < mScene->getCameraCount() ? selectedCameraIdx : 0);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    View currView = {};

    currView.mCamMatrices = camera.matrices;

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
        getSharedContext().mSubframeIndex = 0;
    }
    // Latched here, where it is read, rather than at the end of the function. A
    // frame that returns early -- and there is now more than one way to, all of
    // them failures -- would otherwise leave the previous pose at whatever it
    // was before, so the next frame compares against a stale camera, calls it a
    // cut and resets the accumulator. The visible symptom was a render pinned at
    // one sample: every attempt reset, incremented, and failed again.
    mPrevView = currView;

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

    params.image = (float4*)((OptixBuffer*)output)->getNativePtr();
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

    if (params.debug == 1)
    {
        samplesThisLaunch = 1;
        enableAccumulation = false;
    }

    params.samples_per_launch = samplesThisLaunch;
    params.enableAccumulation = enableAccumulation;
    params.maxSampleCount = totalSpp;

    if (mFrameStartEvent)
    {
        // The legacy default stream, which cudaStreamCreate's blocking streams
        // synchronise against -- so a pair recorded here brackets the launch on
        // mState.stream as well as the post kernels on this one, and the number
        // is the whole frame rather than the part of it that happens to share a
        // stream with the events.
        latchCudaError(cudaEventRecord(mFrameStartEvent, 0), "record the frame start event");
    }

    if (latchCudaError(cudaMemcpy(reinterpret_cast<void*>(mState.mParamsBuffer->getPtr()), &params, sizeof(params),
                                  cudaMemcpyHostToDevice),
                       "upload the launch parameters"))
    {
        return;
    }
    markStageSubmitted(optix::GpuStage::ParamsUpload, 0);

    if (samplesThisLaunch != 0)
    {
        // Launch OptiX path tracer.
        //
        // Checked rather than OPTIX_CHECK'd: an abort here takes the editor down
        // with the scene still on the device and leaves the harness nothing to
        // report. A latch lets StrelkaCLI say the image is not valid and exit
        // non-zero, which is the logic it has always had and never saw an error
        // to trigger.
        const OptixResult launchResult =
            optixLaunch(mState.pipeline, mState.stream, mState.mParamsBuffer->getPtr(), sizeof(Params), &mState.sbt,
                        width, height,
                        /*depth=*/1);
        if (launchResult != OPTIX_SUCCESS)
        {
            mDeviceError = true;
            if (!mDeviceErrorReported)
            {
                mDeviceErrorReported = true;
                STRELKA_ERROR("optixLaunch failed: [{}] {}", optixGetErrorName(launchResult),
                              optixGetErrorString(launchResult));
                reportGpuStageFailure();
            }
        }
        else
        {
            markStageSubmitted(optix::GpuStage::PathTrace, mState.stream);
        }

        // Advanced whether or not the launch took, which is deliberate and is
        // what Metal does -- it counts at encode time and asks about validity
        // separately. The counter says how many samples have been *submitted*;
        // deviceError() says whether they are worth anything. A counter that
        // stalled on failure would leave every caller that loops until it
        // reaches its target -- StrelkaCLI, and the editor's audit modes --
        // spinning forever on a GPU that will never produce another sample, and
        // never reaching the check they already have for exactly this.
        getSharedContext().mSubframeIndex =
            enableAccumulation ? getSharedContext().mSubframeIndex + samplesThisLaunch : 0;
        if (mDeviceError)
        {
            return;
        }
    }
    else
    {
        // Copy accumulated buffer to output image
        const size_t imageSize = mState.params.image_width * mState.params.image_height * sizeof(float4);
        const void* srcBuffer = nullptr;

        // Select source buffer based on debug mode
        switch (params.debug)
        {
        case 0:
            srcBuffer = params.accum;
            break;
        case 2:
            srcBuffer = params.diffuse;
            break;
        case 3:
            srcBuffer = params.specular;
            break;
        }

        if (srcBuffer)
        {
            CUDA_CHECK(cudaMemcpy(params.image, srcBuffer, imageSize, cudaMemcpyDeviceToDevice));
        }
    }

    // Apply tonemapping except for debug mode 1
    if (params.debug != 1)
    {
        float maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");
        exposureValue *= maxEDR;
        tonemap(tonemapperType, exposureValue, gamma, params.image, width, height);
        markStageSubmitted(optix::GpuStage::Tonemap, 0);
    }

    if (mFrameStopEvent && !latchCudaError(cudaEventRecord(mFrameStopEvent, 0), "record the frame stop event"))
    {
        mFrameTimingPending = true;
    }

    getSharedContext().mFrameNumber++;

    mState.prevParams = mState.params;
}

// ---------------------------------------------------------------------------
// The sliced scene build
// ---------------------------------------------------------------------------

optix::SceneBuildHooks OptiXRender::makeSceneBuildHooks()
{
    optix::SceneBuildHooks hooks;
    hooks.nowMs = []() { return nowMilliseconds(); };
    hooks.onStageTimed = [](optix::BuildStage stage, double elapsedMs) {
        STRELKA_DEBUG("Scene build stage '{}' took {:.0f} ms", optix::buildStageName(stage), elapsedMs);
    };
    hooks.onBuffersEnter = [this]() {
        mBuildStartMs = nowMilliseconds();
        mReportedFirstPartialFrame = false;
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Geometry);
        }
    };
    hooks.buildBuffers = [this]() { buildSceneBuffers(); };
    hooks.onEnvironmentEnter = [this]() {
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Environment);
        }
    };
    hooks.buildEnvironment = [this](Buffer* output) { buildSceneEnvironment(output); };
    hooks.publishMaterialParams = [this]() { publishMaterialParams(); };
    hooks.onStructuresEnter = [this]() {
        if (mLoadProgress && mBlasMeshCursor == 0 && mBlasCurveCursor == 0)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Structures,
                                      static_cast<uint32_t>(mScene->getMeshes().size() + mScene->getCurves().size()));
        }
    };
    hooks.stepStructures = [this](double budgetMs) { return stepStructures(budgetMs); };
    hooks.onMaterialTexturesEnter = [this]() {
        if (mLoadProgress && mMaterialTextureCursor == 0)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Textures,
                                      static_cast<uint32_t>(mScene->getMaterials().size()));
        }
    };
    hooks.stepMaterialTextures = [this](double budgetMs) { return stepMaterialTextures(budgetMs); };
    hooks.onTailEnter = [this]() {
        if (mLoadProgress)
        {
            mLoadProgress->beginStage(LoadProgress::Stage::Done);
        }
    };
    hooks.buildTail = [this](Buffer* output) { buildSceneTail(output); };
    return hooks;
}

bool OptiXRender::stepSceneBuild(Buffer* output)
{
    optix::SceneBuildHooks hooks = makeSceneBuildHooks();
    return mScenePrep.step(hooks, output);
}

void OptiXRender::finishSceneBuild(Buffer* output)
{
    optix::SceneBuildHooks hooks = makeSceneBuildHooks();
    mScenePrep.finish(hooks, output);
}

void OptiXRender::buildSceneBuffers()
{
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
}

// The stage that makes a scene visible before it is loaded.
//
// Nothing here depends on geometry or on materials, and together these are
// already a complete picture: somewhere to accumulate, the sky, the lights, a
// shader binding table, and a top level to trace against. The top level is built
// empty on purpose -- every ray then misses and reaches the environment, so the
// first frame is the scene's own lighting with none of its objects in it yet,
// and the objects appear in that rather than replacing a black screen.
void OptiXRender::buildSceneEnvironment(Buffer* output)
{
    updatePathtracerParams(output->width(), output->height());

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

    createLightBuffer();
    // Built here rather than with the structures because it does not depend on
    // them: the records are packed per instance in the same order the top level
    // will use, so the table the empty top level never reaches is already the one
    // the real top level wants.
    createSbt();
    buildEmptyTopLevel();
}

void OptiXRender::buildEmptyTopLevel()
{
    uploadInstancesToDevice({});

    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = 0;

    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    // Straight into the member the real top level will overwrite. It is a few
    // hundred bytes, and createTopLevelAccelerationStructure reallocates when it
    // needs more, so nothing is leaked by handing it a buffer sized for nothing.
    mTlasBuffer.reset(new OptixBuffer(std::max<size_t>(iasBufferSizes.outputSizeInBytes, 1)));
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < iasBufferSizes.tempSizeInBytes)
    {
        mTempAccelBuffer.reset(new OptixBuffer(std::max<size_t>(iasBufferSizes.tempSizeInBytes, 1)));
    }

    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &iasOptions, &iasInput, 1, mTempAccelBuffer->getPtr(),
                                iasBufferSizes.tempSizeInBytes, mTlasBuffer->getPtr(),
                                iasBufferSizes.outputSizeInBytes, &mState.ias_handle, nullptr, 0));
    markStageSubmitted(optix::GpuStage::AccelBuild, mState.stream);
}

// One slice of the long pole. Returns true when every structure is on the device
// and the real top level has replaced the empty one.
bool OptiXRender::stepStructures(double budgetMs)
{
    const auto& meshes = mScene->getMeshes();
    const auto& curves = mScene->getCurves();

    if (mBlasMeshCursor == 0 && mBlasCurveCursor == 0)
    {
        mOptixMeshes.clear();
        mOptixCurves.clear();
        mOptixMeshes.reserve(meshes.size());
        mOptixCurves.reserve(curves.size());
    }

    const auto started = std::chrono::steady_clock::now();
    auto overBudget = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() >=
               budgetMs;
    };

    while (mBlasMeshCursor < meshes.size())
    {
        mOptixMeshes.emplace_back(createMesh(meshes[mBlasMeshCursor]));
        ++mBlasMeshCursor;
        if (mLoadProgress)
        {
            mLoadProgress->step();
        }
        if (overBudget())
        {
            return false;
        }
    }
    while (mBlasCurveCursor < curves.size())
    {
        mOptixCurves.emplace_back(createCurve(curves[mBlasCurveCursor]));
        ++mBlasCurveCursor;
        if (mLoadProgress)
        {
            mLoadProgress->step();
        }
        if (overBudget())
        {
            return false;
        }
    }

    if (!mTopLevelBuilt)
    {
        // Not sliced. A top level is one build over the instance list, and there
        // is no partial one that is safe to trace: an instance whose bottom level
        // is not built yet is a null traversable, not a missing object.
        createTopLevelAccelerationStructure();
        markStageSubmitted(optix::GpuStage::AccelBuild, mState.stream);
        mTopLevelBuilt = true;
    }
    return true;
}

void OptiXRender::buildSceneTail(Buffer* output)
{
    (void)output;
    // Fresh scene: drop the edit bits the loader left behind, so the frame after
    // this one does not rebuild the top level it has just finished building.
    mScene->consumeChanges();

    // What the scene actually cost, once, from the sizes the objects report
    // rather than from a tally kept at the allocation sites.
    MemoryReport report;
    if (memoryReport(report))
    {
        std::sort(report.gpu.begin(), report.gpu.end(),
                  [](const MemoryReport::Entry& a, const MemoryReport::Entry& b) { return a.bytes > b.bytes; });
        std::string top;
        for (size_t i = 0; i < std::min<size_t>(4, report.gpu.size()); ++i)
        {
            top += fmt::format("{}{} {:.2f} GB", i ? ", " : "", report.gpu[i].name,
                               static_cast<double>(report.gpu[i].bytes) / 1073741824.0);
        }
        STRELKA_INFO("Memory: device {:.2f} GB, process {:.2f} GB; largest: {}",
                     static_cast<double>(report.deviceAllocated) / 1073741824.0,
                     static_cast<double>(report.processFootprint) / 1073741824.0, top);
    }
}

// ---------------------------------------------------------------------------
// Memory report
// ---------------------------------------------------------------------------

// Deliberately not a running tally kept at the allocation sites: those drift the
// moment someone adds a cudaMalloc and forgets the counter, and the first symptom
// is a total that no longer matches the device's. Every figure below is either
// the size an OptixBuffer was created with, a size CUDA is asked for by handle,
// or a size recorded at the one place a bare CUdeviceptr is produced -- and
// whatever is missed shows up as the unaccounted remainder between the sum and
// the two totals rather than vanishing.
bool OptiXRender::memoryReport(MemoryReport& report) const
{
    report.gpu.clear();
    report.cpu.clear();

    auto add = [&report](const char* name, size_t bytes) {
        if (bytes > 0)
        {
            report.gpu.push_back({ name, bytes });
        }
    };
    auto bufBytes = [](const std::unique_ptr<OptixBuffer>& b) { return b ? b->size() : 0; };

    add("Vertices", bufBytes(mVertexBuffer));
    add("Vertices (previous)", bufBytes(mPrevVertexBuffer));
    add("Indices", bufBytes(mIndexBuffer));
    add("Curves", bufBytes(mPointsBuffer) + bufBytes(mWidthsBuffer) + bufBytes(mSegmentIndicesBuffer));

    {
        // cudaArrayGetInfo asks the array itself for its extent and format, so
        // this is the texture's own answer rather than a replay of the arguments
        // it was created with.
        size_t bytes = 0;
        for (cudaArray_t array : mTextureArrays)
        {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            unsigned int flags = 0;
            if (array && cudaArrayGetInfo(&desc, &extent, &flags, array) == cudaSuccess)
            {
                const size_t texels = std::max<size_t>(extent.width, 1) * std::max<size_t>(extent.height, 1) *
                                      std::max<size_t>(extent.depth, 1);
                bytes += texels * ((desc.x + desc.y + desc.z + desc.w) / 8);
            }
        }
        add("Textures", bytes);
    }
    add("Texture table", bufBytes(mTexturesDataBuffer));
    add("Environment", bufBytes(mEnvCdfXBuffer) + bufBytes(mEnvCdfYBuffer) + bufBytes(mEnvRawDataBuffer));

    {
        size_t bytes = 0;
        for (const std::unique_ptr<Mesh>& mesh : mOptixMeshes)
        {
            bytes += mesh ? mesh->gas_bytes : 0;
        }
        for (const std::unique_ptr<Curve>& curve : mOptixCurves)
        {
            bytes += curve ? curve->gas_bytes : 0;
        }
        add("BLAS", bytes);
    }
    add("TLAS", bufBytes(mTlasBuffer));
    // Kept for the lifetime of the renderer so a refit needs no allocation.
    add("Accel scratch", bufBytes(mTempAccelBuffer) + bufBytes(mCompactedSizeBuffer));
    add("Instance descriptors", mState.d_instances_size);
    {
        size_t bytes = 0;
        for (const std::shared_ptr<OptixBuffer>& b : mMotionTransformBuffers)
        {
            bytes += b ? b->size() : 0;
        }
        add("Motion transforms", bytes);
    }
    add("Shader binding table", mSbtBytes);
    add("Materials", bufBytes(mMaterialParamsBuffer));
    add("Lights", bufBytes(mLightBuffer));
    add("Skinning", bufBytes(mVertexSkinDataBuffer) + mSkinningPtrs.bytes);

    {
        // Per pixel rather than per scene, which is why a render at a larger
        // resolution costs more before a ray is cast.
        const size_t pixels = static_cast<size_t>(mState.params.image_width) * mState.params.image_height;
        size_t bytes = 0;
        if (mState.params.accum)
            bytes += pixels * sizeof(float4);
        if (mState.params.diffuse)
            bytes += pixels * sizeof(float4);
        if (mState.params.specular)
            bytes += pixels * sizeof(float4);
        if (mState.params.diffuseCounter)
            bytes += pixels * sizeof(uint16_t);
        if (mState.params.specularCounter)
            bytes += pixels * sizeof(uint16_t);
        add("Accumulation & AOVs", bytes);
    }
    {
        size_t bytes = bufBytes(mState.mParamsBuffer) + bufBytes(mStageMarkBuffer);
        for (const Buffer* b : mAsyncOutputBuffers)
        {
            bytes += b ? static_cast<size_t>(b->width()) * b->height() * Buffer::getElementSize(b->getFormat()) : 0;
        }
        add("Uniforms & output", bytes);
    }

    // The host arrays the editor keeps so Scene::pick() can walk them.
    if (mScene)
    {
        const size_t hostBytes = mScene->getVertices().size() * sizeof(oka::Scene::Vertex) +
                                 mScene->getIndices().size() * sizeof(uint32_t);
        if (hostBytes > 0)
        {
            report.cpu.push_back({ "Host geometry (picking)", hostBytes });
        }
    }

    report.deviceAllocated = deviceAllocatedBytes();
    report.processFootprint = processFootprintBytes();
    return true;
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
    createTimingEvents();
    createModule();
    createProgramGroups();
    createPipeline();

    // Arm the deferred scene build. The first render() call picks it up a stage
    // at a time; a synchronous caller drives it to the end through renderSync.
    //
    // Here rather than on the first render() so isBuildingScene() is true the
    // moment init() returns. The editor waits on it before it measures auto
    // exposure and before it calls the scene ready, and a flag that only became
    // true after the first frame would let both of those happen against a scene
    // with nothing in it.
    mScenePrep.begin();
}

// ---------------------------------------------------------------------------
// Frame timing
// ---------------------------------------------------------------------------

void OptiXRender::createTimingEvents()
{
    // Default flags, not cudaEventDisableTiming -- the timing is the point.
    if (latchCudaError(cudaEventCreate(&mFrameStartEvent), "create the frame start event") ||
        latchCudaError(cudaEventCreate(&mFrameStopEvent), "create the frame stop event"))
    {
        mFrameStartEvent = nullptr;
        mFrameStopEvent = nullptr;
    }

    // One byte per stage. Allocated once and reset per frame, so the fault path
    // costs a single small memset in the steady state and nothing at all when
    // nothing fails.
    mStageMarkBuffer.reset(new OptixBuffer(optix::kGpuStageCount));
}

void OptiXRender::collectFrameTiming(bool wait)
{
    if (!mFrameTimingPending || !mFrameStartEvent || !mFrameStopEvent)
    {
        return;
    }
    if (!wait)
    {
        const cudaError_t query = cudaEventQuery(mFrameStopEvent);
        if (query == cudaErrorNotReady)
        {
            // Still running. The previous frame's number stands, which is what
            // the title bar and the progress line want -- a zero here would read
            // as "instant" rather than "not known yet".
            return;
        }
        if (query != cudaSuccess)
        {
            latchCudaError(query, "query the frame stop event");
            mFrameTimingPending = false;
            return;
        }
    }
    else if (latchCudaError(cudaEventSynchronize(mFrameStopEvent), "wait for the frame stop event"))
    {
        mFrameTimingPending = false;
        return;
    }

    float elapsedMs = 0.0f;
    if (!latchCudaError(cudaEventElapsedTime(&elapsedMs, mFrameStartEvent, mFrameStopEvent), "read the frame time"))
    {
        mLastRenderTimeMs.store(static_cast<double>(elapsedMs), std::memory_order_relaxed);
    }
    mFrameTimingPending = false;
}

// ---------------------------------------------------------------------------
// Device errors and the breadcrumb that says which stage was running
// ---------------------------------------------------------------------------

bool OptiXRender::latchCudaError(cudaError_t err, const char* what)
{
    if (err == cudaSuccess)
    {
        return false;
    }
    mDeviceError = true;
    if (!mDeviceErrorReported)
    {
        mDeviceErrorReported = true;
        STRELKA_ERROR("CUDA failed to {}: {} ({})", what, cudaGetErrorString(err), cudaGetErrorName(err));
        reportGpuStageFailure();
    }
    return true;
}

void OptiXRender::beginFrameBreadcrumbs()
{
    if (!mStageMarkBuffer || mDeviceError)
    {
        return;
    }
    std::fill(std::begin(mStageSubmitted), std::end(mStageSubmitted), static_cast<uint8_t>(0));
    // On the legacy stream, so it is ordered before everything this frame
    // enqueues on either stream.
    cudaMemsetAsync(reinterpret_cast<void*>(mStageMarkBuffer->getPtr()), 0, optix::kGpuStageCount, 0);
}

void OptiXRender::markStageSubmitted(optix::GpuStage stage, CUstream stream)
{
    const size_t index = static_cast<size_t>(stage);
    if (index >= optix::kGpuStageCount)
    {
        return;
    }
    mStageSubmitted[index] = 1;
    if (!mStageMarkBuffer)
    {
        return;
    }
    // Enqueued behind the work it follows, on the same stream, so the byte lands
    // only if that work ran to completion. A fault takes the stream down with
    // the memset still in it, which is exactly what makes the absent mark
    // meaningful.
    cudaMemsetAsync(reinterpret_cast<uint8_t*>(mStageMarkBuffer->getPtr()) + index, 1, 1, stream);
}

void OptiXRender::reportGpuStageFailure()
{
    if (!mStageMarkBuffer)
    {
        return;
    }
    uint8_t completed[optix::kGpuStageCount] = {};
    // Deliberately unchecked: the device has already failed, and a second error
    // out of this copy would say nothing the first one did not.
    if (cudaMemcpy(completed, reinterpret_cast<void*>(mStageMarkBuffer->getPtr()), optix::kGpuStageCount,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        STRELKA_ERROR("GPU fault: the stage breadcrumbs could not be read back either");
        return;
    }

    const optix::GpuStageFailure failure =
        optix::inferGpuStageFailure(completed, mStageSubmitted, optix::kGpuStageCount);
    if (failure.suspectedStage >= 0)
    {
        STRELKA_ERROR("GPU fault while running '{}' (last completed: {})",
                      optix::gpuStageName(static_cast<optix::GpuStage>(failure.suspectedStage)),
                      failure.lastCompletedStage >= 0 ?
                          optix::gpuStageName(static_cast<optix::GpuStage>(failure.lastCompletedStage)) :
                          "nothing");
    }
    else if (failure.allSubmittedCompleted)
    {
        // Everything this frame submitted also finished, so the error came from
        // outside it. Said out loud because the alternative is blaming whichever
        // stage happened to be last, which is how an earlier asynchronous fault
        // gets attributed to the tonemap.
        STRELKA_ERROR("GPU fault: every stage this frame submitted completed; the error is from outside this frame");
    }
    if (mScenePrep.isBuilding())
    {
        STRELKA_ERROR("The scene build was in its '{}' stage", optix::buildStageName(mScenePrep.stage()));
    }
}

void OptiXRender::syncFrameAndLatchErrors()
{
    if (mDeviceError)
    {
        return;
    }
    // Both, and in this order. The launch runs on mState.stream; the post
    // kernels run on the legacy stream, which the launch's blocking stream
    // orders against but which has its own errors to report.
    if (mState.stream && latchCudaError(cudaStreamSynchronize(mState.stream), "finish the render stream"))
    {
        return;
    }
    if (latchCudaError(cudaDeviceSynchronize(), "finish the frame"))
    {
        return;
    }
    // A kernel launch that failed to *start* reports here rather than at the
    // synchronise, and nothing else in this backend would ever look.
    latchCudaError(cudaGetLastError(), "launch this frame's kernels");
}

void OptiXRender::renderSync(Buffer* output)
{
    // A synchronous caller wants the frame, not a responsive window, so the
    // build runs to completion here rather than one stage per call.
    finishSceneBuild(output);
    render(output);
    syncFrameAndLatchErrors();
    // Blocking, because a caller that has just waited for the frame is entitled
    // to the frame's time rather than the previous one's.
    collectFrameTiming(true);
}

// ---------------------------------------------------------------------------
// Non-blocking editor loop
// ---------------------------------------------------------------------------

// Submits a frame and waits for it, which reads as a contradiction next to
// Metal's version and is not one. Metal hands the frame to a completion handler
// and returns; OptiX has no such callback, and the alternatives are polling an
// event from the UI thread -- a second loop that has to be kept in step with
// this one -- or lying about when the frame landed. Everything measuring a frame
// reads isRenderBusy() as "the frame is there", and returning before it is turns
// every one of those measurements into a race. The wait is one launch, which at
// the interactive samples-per-launch this path uses is milliseconds.
void OptiXRender::triggerRenderIfIdle()
{
    if (mRenderBusy.load(std::memory_order_acquire) || deviceError())
    {
        return;
    }

    const uint32_t w = getSettings()->getAs<uint32_t>("render/width");
    const uint32_t h = getSettings()->getAs<uint32_t>("render/height");
    if (w == 0 || h == 0)
    {
        return;
    }

    // The buffer that is not the one being displayed.
    const int ready = mReadyIndex.load(std::memory_order_acquire);
    mWriteIndex = (ready >= 0) ? (1 - ready) : 0;

    if (!mAsyncOutputBuffers[mWriteIndex])
    {
        BufferDesc desc{};
        desc.format = BufferFormat::FLOAT4;
        desc.width = w;
        desc.height = h;
        mAsyncOutputBuffers[mWriteIndex] = createBuffer(desc);
    }
    else if (mAsyncOutputBuffers[mWriteIndex]->width() != w || mAsyncOutputBuffers[mWriteIndex]->height() != h)
    {
        mAsyncOutputBuffers[mWriteIndex]->resize(w, h);
    }

    mRenderBusy.store(true, std::memory_order_release);
    render(mAsyncOutputBuffers[mWriteIndex]);
    // A build stage that published nothing has already cleared the flag and
    // produced no image; publishing the slot then would show the caller a buffer
    // this frame never wrote.
    if (mRenderBusy.load(std::memory_order_acquire))
    {
        syncFrameAndLatchErrors();
        collectFrameTiming(true);
        // A failed frame is not published. Its buffer holds whatever was in it
        // before, and showing that is how a GPU fault comes out looking like a
        // lighting bug; the last good frame stays on screen and the editor's
        // alert says why it stopped moving.
        if (!mDeviceError)
        {
            mReadyIndex.store(mWriteIndex, std::memory_order_release);
        }
        mRenderBusy.store(false, std::memory_order_release);
    }
}

Buffer* OptiXRender::getReadyBuffer()
{
    const int ready = mReadyIndex.load(std::memory_order_acquire);
    return ready >= 0 ? mAsyncOutputBuffers[ready] : nullptr;
}

// ---------------------------------------------------------------------------
// GPU capture
// ---------------------------------------------------------------------------

// Nsight rather than Xcode, and a range rather than a file.
//
// The CUDA analogue of Metal's .gputrace is the profiler's capture range:
// launch under `nsys profile --capture-range=cudaProfilerApi` (or
// `ncu --profile-from-start off`) and these two calls bracket exactly the frame
// that would otherwise have been averaged together with the acceleration
// structure build. The path is where the *profiler* was told to write, not
// something this can choose, so it is reported rather than used -- saying so is
// what stops the next reader assuming a file appeared and going looking for it.
void OptiXRender::beginGpuCapture(const std::string& path)
{
    if (mCaptureActive)
    {
        return;
    }
    const cudaError_t started = cudaProfilerStart();
    if (started != cudaSuccess)
    {
        STRELKA_ERROR("GPU capture failed to start: {}. Run under `nsys profile "
                      "--capture-range=cudaProfilerApi` or `ncu --profile-from-start off`.",
                      cudaGetErrorString(started));
        return;
    }
    mCaptureActive = true;
    STRELKA_INFO("GPU capture range open; the profiler writes it, not Strelka (asked for '{}')", path);
}

void OptiXRender::endGpuCapture()
{
    if (!mCaptureActive)
    {
        return;
    }
    mCaptureActive = false;
    // Everything in the range has to have finished before the range closes, or
    // the profiler attributes this frame's launch to whatever comes next.
    cudaDeviceSynchronize();
    cudaProfilerStop();
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
    markStageSubmitted(optix::GpuStage::EnvCdf, 0);
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

// The material table, without a single texture in it.
//
// Split from the maps on purpose, and the split is what makes a scene appear
// before it has finished loading: this is a few kilobytes of struct copies and
// everything the acceleration structures need to know about materials comes out
// of it, while decoding the maps is minutes of PNG on a large scene. Geometry
// therefore reaches the screen in flat material colours and the maps fill into
// the live table behind it, rather than the whole scene waiting on the last JPEG.
void OptiXRender::publishMaterialParams()
{
    const auto& matDescs = mScene->getMaterials();
    if (matDescs.empty())
    {
        STRELKA_WARNING("No materials in scene");
        return;
    }

    mMaterials.resize(matDescs.size());
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        MaterialParams params = matDescs[i].params;
        // Every slot empty until its stage runs. A -1 is what the shader reads as
        // "no map", so a material published now shades with its factors alone
        // rather than sampling a texture object that is still zero.
        params.base_color_tex = -1;
        params.metallic_roughness_tex = -1;
        params.normal_tex = -1;
        params.emission_tex = -1;
        params.occlusion_tex = -1;
        params.transmission_tex = -1;
        mMaterials[i].params = params;
    }

    mHostMaterialTextures.assign(matDescs.size() * MAX_MATERIAL_TEXTURES, 0);
    const size_t totalTexSize = mHostMaterialTextures.size() * sizeof(cudaTextureObject_t);
    mTexturesDataBuffer.reset(new OptixBuffer(totalTexSize));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(mTexturesDataBuffer->getPtr()), 0, totalTexSize));

    std::vector<MaterialParams> allParams(matDescs.size());
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        allParams[i] = mMaterials[i].params;
    }
    const size_t paramsSize = allParams.size() * sizeof(MaterialParams);
    mMaterialParamsBuffer.reset(new OptixBuffer(paramsSize));
    CUDA_CHECK(
        cudaMemcpy((void*)mMaterialParamsBuffer->getPtr(), allParams.data(), paramsSize, cudaMemcpyHostToDevice));
    mMaterialCount = matDescs.size();

    mState.params.materials = (MaterialParams*)mMaterialParamsBuffer->getPtr();
    mState.params.materialTextures = (cudaTextureObject_t*)mTexturesDataBuffer->getPtr();

    mMaterialTextureCursor = 0;
    mTextureCache.clear();
}

// One slice of texture decoding. Returns true when every material has its maps.
//
// Each material is patched into the live device table as it finishes -- its six
// texture-object slots and its own MaterialParams entry, both at their offsets --
// so a frame traced between two slices is correct for the materials that have
// arrived and correct-without-maps for the ones that have not. Nothing is ever
// half-written: the parameters naming a slot go up after the slot holds the
// object.
bool OptiXRender::stepMaterialTextures(double budgetMs)
{
    const auto& matDescs = mScene->getMaterials();
    if (matDescs.empty() || !mMaterialParamsBuffer || !mTexturesDataBuffer)
    {
        return true;
    }

    const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
    const fs::path resourcePath(resourcePathStr);

    auto loadOrCacheTex = [&](const std::string& relPath) -> cudaTextureObject_t {
        if (relPath.empty())
            return 0;
        const fs::path fullPath = resourcePath / relPath;
        const std::string key = fullPath.string();
        auto it = mTextureCache.find(key);
        if (it != mTextureCache.end())
            return it->second;
        if (!fs::exists(fullPath))
        {
            STRELKA_WARNING("Texture not found: {}", key);
            mTextureCache[key] = 0;
            return 0;
        }
        const ::Texture tex = loadTextureFromFile(key);
        mTextureCache[key] = tex.filtered_object;
        return tex.filtered_object;
    };

    const auto started = std::chrono::steady_clock::now();
    while (mMaterialTextureCursor < matDescs.size())
    {
        const size_t i = mMaterialTextureCursor;
        const auto& desc = matDescs[i];
        cudaTextureObject_t* texSlots = &mHostMaterialTextures[i * MAX_MATERIAL_TEXTURES];
        MaterialParams& params = mMaterials[i].params;

        texSlots[0] = loadOrCacheTex(desc.baseColorTexPath);
        texSlots[1] = loadOrCacheTex(desc.metallicRoughnessTexPath);
        texSlots[2] = loadOrCacheTex(desc.normalTexPath);
        texSlots[3] = loadOrCacheTex(desc.emissionTexPath);
        texSlots[4] = loadOrCacheTex(desc.occlusionTexPath);
        // Slot 5 reserved for transmission texture (not yet populated by gltf loader)
        texSlots[5] = 0;

        params.base_color_tex = texSlots[0] ? 0 : -1;
        params.metallic_roughness_tex = texSlots[1] ? 1 : -1;
        params.normal_tex = texSlots[2] ? 2 : -1;
        params.emission_tex = texSlots[3] ? 3 : -1;
        params.occlusion_tex = texSlots[4] ? 4 : -1;
        params.transmission_tex = -1;

        // Objects first, then the parameters that name them.
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<cudaTextureObject_t*>(mTexturesDataBuffer->getPtr()) +
                                  i * MAX_MATERIAL_TEXTURES,
                              texSlots, MAX_MATERIAL_TEXTURES * sizeof(cudaTextureObject_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<MaterialParams*>(mMaterialParamsBuffer->getPtr()) + i, &params,
                              sizeof(MaterialParams), cudaMemcpyHostToDevice));

        ++mMaterialTextureCursor;
        if (mLoadProgress)
        {
            mLoadProgress->step();
            if (mLoadProgress->isCancelled())
            {
                break;
            }
        }
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() >= budgetMs)
        {
            return mMaterialTextureCursor >= matDescs.size();
        }
    }

    STRELKA_INFO("Loaded {} materials ({} unique textures cached)", matDescs.size(), mTextureCache.size());
    return true;
}
