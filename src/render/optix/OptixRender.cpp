#include "OptixRender.h"

#include "OptixBuffer.h"
#include "device_ptr.h"

#include <env.h>

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
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <vector_types.h>
#include <vector_functions.h>

#include <sutil/vec_math_adv.h>
#include <sutil/Matrix.h>

#include "texture_support_cuda.h"
#include "accel_build_policy.h"
#include "ies_pack.h"
#include <host/emissive_mesh_distribution.h>

// ies_pack.h mirrors the two IES header structs so that it -- and its tests --
// need no CUDA. This is where the mirrors are held to the originals in
// lights.h, which is what the shading path actually reads the buffer through.
static_assert(sizeof(oka::optix_ies::IesBufferHeader) == sizeof(IesGpuBufferHeader));
static_assert(sizeof(oka::optix_ies::IesProfileHeader) == sizeof(IesGpuProfileHeader));
static_assert(offsetof(oka::optix_ies::IesBufferHeader, floatOffset) ==
              offsetof(IesGpuBufferHeader, floatOffset));
static_assert(offsetof(oka::optix_ies::IesProfileHeader, nHorizontal) ==
              offsetof(IesGpuProfileHeader, nHorizontal));
static_assert(offsetof(oka::optix_ies::IesProfileHeader, anglesOffset) ==
              offsetof(IesGpuProfileHeader, anglesOffset));
static_assert(offsetof(oka::optix_ies::IesProfileHeader, candelaOffset) ==
              offsetof(IesGpuProfileHeader, candelaOffset));
static_assert(offsetof(oka::optix_ies::IesProfileHeader, maxCandela) ==
              offsetof(IesGpuProfileHeader, maxCandela));

// accel_build_policy.h mirrors OptixBuildFlags so that it -- and its tests --
// need no OptiX SDK. This is where the mirror is held to the original.
static_assert((uint32_t)oka::optix_accel::kFlagNone == (uint32_t)OPTIX_BUILD_FLAG_NONE);
static_assert((uint32_t)oka::optix_accel::kFlagAllowUpdate == (uint32_t)OPTIX_BUILD_FLAG_ALLOW_UPDATE);
static_assert((uint32_t)oka::optix_accel::kFlagAllowCompaction == (uint32_t)OPTIX_BUILD_FLAG_ALLOW_COMPACTION);
static_assert((uint32_t)oka::optix_accel::kFlagPreferFastTrace == (uint32_t)OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
static_assert((uint32_t)oka::optix_accel::kFlagPreferFastBuild == (uint32_t)OPTIX_BUILD_FLAG_PREFER_FAST_BUILD);
static_assert((uint32_t)oka::optix_accel::kFlagAllowRandomVertexAccess ==
              (uint32_t)OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS);

#include "opacity_micromap_policy.h"
#include <optix_micromap.h>
// The uv the micromap classifies has to be the uv the shader tests, and both
// come out of this: 14 bits a component over [-10, 10], unpacked by the same
// expression on the host and on the device.
#include <strelka/scene/vertex_packing.h>

// The same treatment for the micromap mirrors. A state written with the wrong
// value would not fail to build; it would quietly mark cut-away geometry opaque.
static_assert((uint32_t)oka::optix_omm::kStateTransparent == (uint32_t)OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT);
static_assert((uint32_t)oka::optix_omm::kStateOpaque == (uint32_t)OPTIX_OPACITY_MICROMAP_STATE_OPAQUE);
static_assert((uint32_t)oka::optix_omm::kStateUnknownTransparent ==
              (uint32_t)OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT);
static_assert((uint32_t)oka::optix_omm::kStateUnknownOpaque == (uint32_t)OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE);
static_assert((int32_t)oka::optix_omm::kIndexFullyTransparent ==
              (int32_t)OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_TRANSPARENT);
static_assert((int32_t)oka::optix_omm::kIndexFullyOpaque ==
              (int32_t)OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE);
static_assert((int32_t)oka::optix_omm::kIndexFullyUnknownTransparent ==
              (int32_t)OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_TRANSPARENT);
static_assert((int32_t)oka::optix_omm::kIndexFullyUnknownOpaque ==
              (int32_t)OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_UNKNOWN_OPAQUE);
static_assert(oka::optix_omm::kMaxSubdivisionLevel == OPTIX_OPACITY_MICROMAP_MAX_SUBDIVISION_LEVEL);
static_assert((uint32_t)oka::optix_omm::kAlphaOpaque == (uint32_t)ALPHA_MODE_OPAQUE);
static_assert((uint32_t)oka::optix_omm::kAlphaMask == (uint32_t)ALPHA_MODE_MASK);
static_assert((uint32_t)oka::optix_omm::kAlphaBlend == (uint32_t)ALPHA_MODE_BLEND);

#include "curve_layout.h"

#include <cuda_profiler_api.h>

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <chrono>
#include <cstring>
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
#include <postprocessing/DenoiseGuides.h>
#include <sharc_resolve.h>
#include <skinning/skinning.h>

// Backend-neutral: no Metal headers, and the same table both backends sample
// from. See the note in loadEnvMap().
#include <host/ibl_alias_table.h>

namespace
{

void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
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

inline void optixCheck(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        const char* errorName = optixGetErrorName(res);
        const char* errorString = optixGetErrorString(res);
        STRELKA_ERROR("OptiX call {0} failed: {1}:{2} with [{3}] - [{4}]", call, file, line, errorName, errorString);
        std::abort();
    }
}

inline void optixCheckLog(OptixResult res,
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

} // namespace

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

/// One image decoded to four float channels, and the deleter it needs.
///
/// EXR comes back from tinyexr's malloc and everything else from stb, so the
/// free is not the same call, which is why this carries its own release rather
/// than handing back a bare pointer and a flag for the caller to remember.
struct Rgba32fImage
{
    float* pixels = nullptr;
    int width = 0;
    int height = 0;
    bool fromExr = false;

    bool valid() const
    {
        return pixels != nullptr && width > 0 && height > 0;
    }

    void release()
    {
        if (!pixels)
        {
            return;
        }
        if (fromExr)
        {
            // tinyexr returns a malloc'd buffer, so free is the only correct
            // deleter for it -- there is no RAII form of someone else's malloc.
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            free(pixels);
        }
        else
        {
            stbi_image_free(pixels);
        }
        pixels = nullptr;
    }
};

/// Decode an EXR through tinyexr and anything else through stb, always to RGBA
/// float. `what` names the image in the error message, which is the only reason
/// the environment, its backdrop and a projector's slide cannot share one line.
///
/// stbi_loadf undoes a gamma of 2.2 on a display-encoded file, which is what
/// makes an ordinary PNG usable as a light source rather than as a set of code
/// values.
Rgba32fImage decodeRgba32f(const std::string& path, const char* what)
{
    Rgba32fImage img;
    const std::string ext = fs::path(path).extension().string();
    img.fromExr = (ext == ".exr" || ext == ".EXR");
    if (img.fromExr)
    {
        const char* err = nullptr;
        if (LoadEXR(&img.pixels, &img.width, &img.height, path.c_str(), &err) != TINYEXR_SUCCESS)
        {
            STRELKA_ERROR("Failed to load EXR {}: {} ({})", what, path, err ? err : "unknown");
            if (err)
            {
                FreeEXRErrorMessage(err);
            }
            img.pixels = nullptr;
        }
    }
    else
    {
        int channels = 0;
        img.pixels = stbi_loadf(path.c_str(), &img.width, &img.height, &channels, 4);
        if (!img.pixels)
        {
            STRELKA_ERROR("Failed to load {}: {}", what, path);
        }
    }
    return img;
}

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

    // dlsym hands back void*, and POSIX guarantees only that the round trip
    // through it works; there is no other spelling for looking a function up by
    // name. The three go together because they are one API being bound.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    static auto nvmlInit = reinterpret_cast<InitFn>(dlsym(handle, "nvmlInit_v2"));
    static auto nvmlGetHandle = reinterpret_cast<HandleFn>(dlsym(handle, "nvmlDeviceGetHandleByIndex_v2"));
    static auto nvmlGetProcs = reinterpret_cast<ProcFn>(dlsym(handle, "nvmlDeviceGetComputeRunningProcesses_v3"));
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
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

using RayGenSbtRecord = SbtRecord<RayGenData>;
using MissSbtRecord = SbtRecord<MissData>;
using HitGroupSbtRecord = SbtRecord<HitGroupData>;

namespace
{

bool readSourceFile(std::string& str, const fs::path& filename)
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

} // namespace

OptiXRender::OptiXRender() = default;

OptiXRender::~OptiXRender()
{
    // Destroy texture objects and arrays
    destroyTextures();

    // Free SBT records
    if (mState.sbt.raygenRecord)
        cudaFree(optix::devicePtr<void>(mState.sbt.raygenRecord));
    if (mState.sbt.missRecordBase)
        cudaFree(optix::devicePtr<void>(mState.sbt.missRecordBase));
    if (mState.sbt.hitgroupRecordBase)
        cudaFree(optix::devicePtr<void>(mState.sbt.hitgroupRecordBase));

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
    if (mState.radiance_linear_curve_hit_group)
        optixProgramGroupDestroy(mState.radiance_linear_curve_hit_group);
    for (auto& pg : mState.radiance_hit_groups)
        if (pg) optixProgramGroupDestroy(pg);
    if (mState.occlusion_hit_group)
        optixProgramGroupDestroy(mState.occlusion_hit_group);
    if (mState.occlusion_linear_curve_hit_group)
        optixProgramGroupDestroy(mState.occlusion_linear_curve_hit_group);
    if (mState.light_hit_group)
        optixProgramGroupDestroy(mState.light_hit_group);

    // Destroy modules
    if (mState.ptx_module)
        optixModuleDestroy(mState.ptx_module);
    if (mState.closest_hit_module)
        optixModuleDestroy(mState.closest_hit_module);
    if (mState.m_catromCurveModule)
        optixModuleDestroy(mState.m_catromCurveModule);
    if (mState.m_linearCurveModule)
        optixModuleDestroy(mState.m_linearCurveModule);

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
        cudaFree(optix::devicePtr<void>(mState.d_instances));

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
    OptixDeviceContextOptions options = {};
    CUcontext cuCtx = nullptr;
    unsigned int reorderFlags = 0;

    // Initialize CUDA
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaGetDevice(&mCudaDeviceOrdinal));
    CUDA_CHECK(cudaStreamCreate(&mState.stream));

    OPTIX_CHECK(optixInit());
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
    // Zero means take the current context.
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &mState.context));

    // Ask whether optixReorder() does anything here before paying for its
    // coherence key. The call is documented as a no-op on hardware without the
    // sorting unit, so this is not a correctness gate -- it is what keeps two
    // dependent loads per bounce off a machine that cannot spend them.
    OPTIX_CHECK(optixDeviceContextGetProperty(mState.context,
                                              OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING,
                                              &reorderFlags, sizeof(reorderFlags)));
    mShaderReorderSupported = (reorderFlags & OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING_FLAG_STANDARD) != 0;
    STRELKA_INFO("Shader execution reordering: {}", mShaderReorderSupported ? "supported" : "not available");

    mState.mParamsBuffer = std::make_unique<OptixBuffer>(sizeof(Params));
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
    size_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize, optix::devicePtr<void>(result), sizeof(size_t), cudaMemcpyDeviceToHost));

    // Only compact if it saves space
    if (compactedSize >= outputSizeInBytes)
    {
        return outputSizeInBytes;
    }

    // Allocate compacted buffer
    CUdeviceptr compactedBuffer = 0;
    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(compactedBuffer), compactedSize));

    // Compact acceleration structure into new buffer
    OPTIX_CHECK(optixAccelCompact(mState.context, nullptr, handle, compactedBuffer, compactedSize, &handle));

    // Free original buffer and update pointer
    CUDA_CHECK(cudaFree(optix::devicePtr<void>(buffer)));
    buffer = compactedBuffer;

    return compactedSize;
}

std::unique_ptr<OptiXRender::Curve> OptiXRender::createCurve(const oka::Curve& curve)
{
    auto rcurve = std::make_unique<Curve>();
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = oka::optix_accel::buildFlags(oka::optix_accel::Geometry::Curve);
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t pointsCount = mScene->getCurvesPoint().size(); // total points count in points buffer
    // The sidecar basis selects linear or cubic geometry.
    const bool isLinear = (curve.mType == oka::Curve::Type::eLinear);

    rcurve->isLinear = isLinear;
    rcurve->segmentsPerStrand = curve.mSegmentsPerStrand;

    const std::vector<uint32_t>& vertexCounts = mScene->getCurvesVertexCounts();
    // The sidecar reader computes this too. Checking rather than trusting it
    // costs one pass over the strand counts at load, and the failure it catches
    // -- a strand coordinate that walks off the end of a strand -- is otherwise
    // a subtle shading gradient rather than anything that looks like a bug.
    const uint32_t recomputed = oka::curve_layout::segmentsPerStrand(
        vertexCounts, curve.mVertexCountsStart, curve.mVertexCountsCount, isLinear);
    if (recomputed != curve.mSegmentsPerStrand)
    {
        STRELKA_WARNING("Curve set reports {} segments per strand but its counts imply {}; using the counts",
                        curve.mSegmentsPerStrand, recomputed);
        rcurve->segmentsPerStrand = recomputed;
    }

    const std::vector<int> segmentIndices = oka::curve_layout::segmentIndices(
        vertexCounts, curve.mVertexCountsStart, curve.mVertexCountsCount, curve.mPointsStart, isLinear);

    if (segmentIndices.empty())
    {
        STRELKA_WARNING("Curve set has no segment long enough to build");
        return rcurve;
    }

    const size_t segmentIndicesSize = sizeof(int) * segmentIndices.size();
    // Reuse existing buffer if large enough, otherwise allocate new one
    if (!mSegmentIndicesBuffer || mSegmentIndicesBuffer->size() < segmentIndicesSize)
    {
        mSegmentIndicesBuffer = std::make_unique<OptixBuffer>(segmentIndicesSize);
    }
    CUDA_CHECK(cudaMemcpy(optix::devicePtr<void>(mSegmentIndicesBuffer->getPtr()), segmentIndices.data(),
                          segmentIndicesSize, cudaMemcpyHostToDevice));

    OptixBuildInput curve_input = {};
    curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
    curve_input.curveArray.curveType =
        isLinear ? OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR : OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    // Spherical caps on linear strands, matching Metal (CurveEndCapsSphere) and
    // Cycles' thick round curves: without them a strand is an open tube and a
    // ray down its axis goes straight through the tip.
    curve_input.curveArray.endcapFlags = isLinear ? OPTIX_CURVE_ENDCAP_ON : OPTIX_CURVE_ENDCAP_DEFAULT;

    curve_input.curveArray.numPrimitives = segmentIndices.size();
    CUdeviceptr vertexBuffers[] = { mPointsBuffer->getPtr() };
    curve_input.curveArray.vertexBuffers = vertexBuffers;
    curve_input.curveArray.numVertices = pointsCount;
    curve_input.curveArray.vertexStrideInBytes = sizeof(glm::float3);
    CUdeviceptr widthBuffers[] = { mWidthsBuffer->getPtr() };
    curve_input.curveArray.widthBuffers = widthBuffers;
    curve_input.curveArray.widthStrideInBytes = sizeof(float);
    curve_input.curveArray.normalBuffers = nullptr;
    curve_input.curveArray.normalStrideInBytes = 0;
    curve_input.curveArray.indexBuffer = mSegmentIndicesBuffer->getPtr();
    curve_input.curveArray.indexStrideInBytes = sizeof(int);
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curve_input.curveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &curve_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));

    // Reuse temporary build buffers across GAS builds.
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < gas_buffer_sizes.tempSizeInBytes)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(gas_buffer_sizes.tempSizeInBytes);
    }
    if (!mCompactedSizeBuffer || mCompactedSizeBuffer->size() < sizeof(uint64_t))
    {
        mCompactedSizeBuffer = std::make_unique<OptixBuffer>(sizeof(uint64_t));
    }

    CUdeviceptr d_gas_output_buffer = 0;
    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

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

// ---------------------------------------------------------------------------
// Opacity micromaps
// ---------------------------------------------------------------------------

namespace
{

/// Level 4 -- 256 microtriangles, 64 bytes -- is the finest a triangle is given.
/// Past that the array costs more memory than the traversal it saves, and the
/// classifier's texel scan grows with it while the cutout edge it is resolving
/// does not.
constexpr uint32_t kOmmMaxSubdivisionLevel = 4u;

/// What all of one mesh's micromaps may occupy. A cutout card is two triangles
/// and a forest floor is millions; the level follows the budget rather than the
/// other way round.
constexpr size_t kOmmBytesPerMesh = 64ull << 20;

/// The most texels one microtriangle's classification will look at before it
/// gives up and says unknown. A microtriangle whose uv footprint covers a
/// megatexel is not a cutout edge -- it is a texture mapped so coarsely that the
/// micromap could not resolve it anyway -- and scanning it would turn a scene
/// load into a stall.
constexpr int64_t kOmmMaxTexelsPerMicroTriangle = 4096;

/// Padding follows optix_omm::bilinearTexelSpan's host/device boundary invariant.
constexpr float kOmmUvPad = 1e-4f;

struct Uv
{
    float x = 0.0f;
    float y = 0.0f;
};

Uv unpackUvHost(uint32_t packed)
{
    const glm::float2 uv = oka::unpackUV(packed);
    return Uv{ uv.x, uv.y };
}

Uv barycentricUv(const Uv& a, const Uv& b, const Uv& c, float2 bary)
{
    // interpolateAttrib(): a + bary.x * (b - a) + bary.y * (c - a).
    return Uv{ a.x + bary.x * (b.x - a.x) + bary.y * (c.x - a.x),
               a.y + bary.x * (b.y - a.y) + bary.y * (c.y - a.y) };
}

} // namespace

/// The alpha channel of a material's base-colour texture, as uploaded.
///
/// Decoded through `decodeToPayload` with this render's own texture settings, so
/// the extent, the resampling and the block compression are the ones the device
/// texture went through. Anything else would be describing a different texture:
/// a downscale alone moves an eighth of a dotted mask's texels across a cutoff.
///
/// Returns nullptr when the material has no base-colour texture, which the
/// caller reads as a constant alpha of 1.
const OptiXRender::OmmAlphaImage* OptiXRender::ommAlphaImage(int32_t materialId)
{
    namespace tex = oka::optix_tex;
    namespace omm = oka::optix_omm;

    const auto cached = mOmmAlphaCache.find(materialId);
    if (cached != mOmmAlphaCache.end())
    {
        return &cached->second;
    }

    OmmAlphaImage image;
    const auto& descs = mScene->getMaterials();
    if (materialId < 0 || (size_t)materialId >= descs.size() || descs[materialId].baseColorTexPath.empty())
    {
        return nullptr;
    }

    const fs::path fullPath =
        fs::path(getSettings()->getAs<std::string>("resource/searchPath")) / descs[materialId].baseColorTexPath;
    if (!fs::exists(fullPath))
    {
        mOmmAlphaCache.emplace(materialId, image); // unusable, and stays that way
        return &mOmmAlphaCache.at(materialId);
    }

    const tex::Payload payload = tex::decodeToPayload(fullPath.string(), tex::Kind::Color, textureDecodeSettings());
    if (!payload.valid || payload.levels.empty())
    {
        mOmmAlphaCache.emplace(materialId, image);
        return &mOmmAlphaCache.at(materialId);
    }

    const int width = payload.plan.extent.width;
    const int height = payload.plan.extent.height;
    const std::vector<uint8_t>& level0 = payload.levels[0];
    const size_t texels = (size_t)width * (size_t)height;
    if (width <= 0 || height <= 0)
    {
        mOmmAlphaCache.emplace(materialId, image);
        return &mOmmAlphaCache.at(materialId);
    }

    switch (payload.plan.format)
    {
    case tex::Format::RGBA8:
        if (level0.size() >= texels * 4)
        {
            image.alpha.resize(texels);
            for (size_t i = 0; i < texels; ++i)
                image.alpha[i] = (float)level0[i * 4 + 3] / 255.0f;
            image.usable = true;
            image.tolerance = omm::kExactAlphaTolerance;
        }
        break;
    case tex::Format::RGBA16:
        if (level0.size() >= texels * 8)
        {
            // The decoded level is a byte blob; reading its alpha means viewing
            // it as the texel type the format says it holds.
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            const uint16_t* src = reinterpret_cast<const uint16_t*>(level0.data());
            image.alpha.resize(texels);
            for (size_t i = 0; i < texels; ++i)
                image.alpha[i] = (float)src[i * 4 + 3] / 65535.0f;
            image.usable = true;
            image.tolerance = omm::kExactAlphaTolerance;
        }
        break;
    case tex::Format::RGBA32F:
        if (level0.size() >= texels * 16)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            const float* src = reinterpret_cast<const float*>(level0.data());
            image.alpha.resize(texels);
            for (size_t i = 0; i < texels; ++i)
                image.alpha[i] = src[i * 4 + 3];
            image.usable = true;
            image.tolerance = omm::kExactAlphaTolerance;
        }
        break;
    case tex::Format::BC3:
    {
        // Sixteen bytes a block, the first eight being the BC4-coded alpha. The
        // palette is decoded rather than approximated because a cutout mask goes
        // through it and comes back with edge texels the source never had.
        const int blocksX = (width + 3) / 4;
        const int blocksY = (height + 3) / 4;
        if (level0.size() >= (size_t)blocksX * blocksY * 16)
        {
            image.alpha.assign(texels, 0.0f);
            uint8_t decoded[16];
            for (int by = 0; by < blocksY; ++by)
            {
                for (int bx = 0; bx < blocksX; ++bx)
                {
                    const uint8_t* block = level0.data() + ((size_t)by * blocksX + bx) * 16;
                    omm::decodeBc4AlphaBlock(block, decoded);
                    for (int y = 0; y < 4; ++y)
                    {
                        const int ty = by * 4 + y;
                        if (ty >= height)
                            break;
                        for (int x = 0; x < 4; ++x)
                        {
                            const int tx = bx * 4 + x;
                            if (tx >= width)
                                break;
                            image.alpha[(size_t)ty * width + tx] = (float)decoded[y * 4 + x] / 255.0f;
                        }
                    }
                }
            }
            image.usable = true;
            image.tolerance = omm::kBlockCompressedAlphaTolerance;
        }
        break;
    }
    case tex::Format::BC1:
    case tex::Format::BC5:
        // BC1 carries either no alpha or a punch-through bit whose reading
        // depends on the endpoint order, and BC5 has two channels and no alpha
        // at all. Neither can be read back as the number the sampler returns, so
        // neither gets a micromap -- the any-hit path answers instead, which is
        // exactly what happens today.
        break;
    }

    if (image.usable)
    {
        image.width = width;
        image.height = height;
    }
    else
    {
        STRELKA_WARNING("No opacity micromap for material {}: base colour uploads in a format its alpha cannot be "
                        "read back from exactly",
                        materialId);
    }
    mOmmAlphaCache.emplace(materialId, std::move(image));
    return &mOmmAlphaCache.at(materialId);
}

/// Which material each mesh is drawn with.
///
/// A micromap is attached to the geometry, and the material is on the instance,
/// so the two only line up when every instance of a mesh names the same
/// material. Where they do not, the mesh gets no micromap: describing one
/// cutout while a second instance draws a different one is the one way this
/// feature could change an image.
void OptiXRender::resolveMeshMaterials()
{
    const auto& meshes = mScene->getMeshes();
    mMeshMaterialIds.assign(meshes.size(), -1);
    std::vector<bool> conflicted(meshes.size(), false);

    for (const oka::Instance& instance : mScene->getInstances())
    {
        if (instance.type != oka::Instance::Type::eMesh)
        {
            continue;
        }
        if (instance.mMeshId >= mMeshMaterialIds.size())
        {
            continue;
        }
        const int32_t materialId = (instance.mMaterialId == kInvalidIndex) ? 0 : (int32_t)instance.mMaterialId;
        int32_t& slot = mMeshMaterialIds[instance.mMeshId];
        if (slot < 0 && !conflicted[instance.mMeshId])
        {
            slot = materialId;
        }
        else if (slot != materialId)
        {
            slot = -1;
            conflicted[instance.mMeshId] = true;
        }
    }
}

/// Classify one mesh's triangles against its material's alpha test and build the
/// micromap array that says so.
///
/// Every triangle gets an entry in the index buffer. Most of them are one of the
/// four predefined indices -- a whole triangle inside the cutout, or outside it,
/// or unresolvable -- and cost four bytes. Only a triangle the cutout edge
/// actually crosses gets a micromap of its own.
OptiXRender::MeshOpacityMicromap OptiXRender::buildMeshOpacityMicromap(const oka::Mesh& mesh, size_t meshIndex)
{
    namespace omm = oka::optix_omm;
    MeshOpacityMicromap out;

    if (!mOpacityMicromapsEnabled || meshIndex >= mMeshMaterialIds.size())
    {
        return out;
    }
    if (mesh.isSkeletal)
    {
        // A skeletal mesh is refit every frame and rebuilt every so often, and an
        // update reads its build input back -- micromaps included, under rules
        // that need ALLOW_OPACITY_MICROMAP_UPDATE to relax. The uv does not move
        // under a skeleton, so the micromap would still be right; the update path
        // is what is not worth getting wrong for a cutout nobody skins.
        return out;
    }
    const int32_t materialId = mMeshMaterialIds[meshIndex];
    const auto& descs = mScene->getMaterials();
    if (materialId < 0 || (size_t)materialId >= descs.size())
    {
        return out;
    }

    const MaterialParams& material = descs[materialId].params;
    omm::AlphaRule rule;
    rule.alphaMode = (uint32_t)material.alpha_mode;
    rule.baseAlpha = material.base_color_alpha;
    rule.cutoff = material.alpha_cutoff;
    if (rule.alphaMode == omm::kAlphaOpaque)
    {
        // The instance already carries OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT, so
        // there is no shader to skip and nothing for a micromap to buy.
        return out;
    }

    const OmmAlphaImage* image = ommAlphaImage(materialId);
    if (image != nullptr && !image->usable)
    {
        return out;
    }
    rule.hasTexture = image != nullptr;
    const float tolerance = image != nullptr ? image->tolerance : omm::kExactAlphaTolerance;

    const size_t triangleCount = mesh.mCount / 3;
    if (triangleCount == 0)
    {
        return out;
    }

    const std::vector<uint32_t>& indices = mScene->getIndices();
    const std::vector<oka::Scene::Vertex>& vertices = mScene->getVertices();

    /// Bound the base-colour alpha over everything a bilinear fetch inside a uv
    /// box could read, and hand that to the classifier. Returns Mixed rather
    /// than a bound whenever the box is too big to scan, which is the safe
    /// answer to a question this cannot afford to ask.
    auto classifyUvBox = [&](float u0, float u1, float v0, float v1) -> omm::Coverage {
        if (image == nullptr)
        {
            return omm::classifyCoverage(rule, 1.0f, 1.0f, tolerance);
        }
        const omm::TexelSpan xs = omm::bilinearTexelSpan(u0, u1, image->width, kOmmUvPad);
        const omm::TexelSpan ys = omm::bilinearTexelSpan(v0, v1, image->height, kOmmUvPad);
        const int64_t xCount = xs.full ? image->width : xs.count();
        const int64_t yCount = ys.full ? image->height : ys.count();
        if (xCount * yCount > kOmmMaxTexelsPerMicroTriangle)
        {
            return omm::Coverage::Mixed;
        }
        float minAlpha = 1.0f;
        float maxAlpha = 0.0f;
        const int xBegin = xs.full ? 0 : xs.lo;
        const int yBegin = ys.full ? 0 : ys.lo;
        for (int64_t y = 0; y < yCount; ++y)
        {
            const int ty = omm::wrapTexel((int)(yBegin + y), image->height);
            const float* row = image->alpha.data() + (size_t)ty * image->width;
            for (int64_t x = 0; x < xCount; ++x)
            {
                const float a = row[omm::wrapTexel((int)(xBegin + x), image->width)];
                minAlpha = std::min(minAlpha, a);
                maxAlpha = std::max(maxAlpha, a);
            }
        }
        return omm::classifyCoverage(rule, minAlpha, maxAlpha, tolerance);
    };

    const uint32_t level = omm::chooseSubdivisionLevel(triangleCount, kOmmBytesPerMesh, kOmmMaxSubdivisionLevel);
    const bool subdivide = level != omm::kNoSubdivision;
    const uint32_t microCount = subdivide ? omm::microTriangleCount(level) : 0u;
    const size_t microBytes = subdivide ? omm::microMapBytes(level) : 0u;

    std::vector<int32_t> triangleIndices(triangleCount, omm::kIndexFullyUnknownOpaque);
    std::vector<uint8_t> micromapData;
    std::vector<OptixOpacityMicromapDesc> micromapDescs;
    std::vector<uint8_t> scratch(microBytes, 0u);
    omm::BuildSummary summary;
    summary.triangles = triangleCount;
    summary.subdivisionLevel = subdivide ? level : 0u;

    for (size_t tri = 0; tri < triangleCount; ++tri)
    {
        const size_t base = mesh.mIndex + tri * 3;
        if (base + 2 >= indices.size())
        {
            break;
        }
        const size_t v0 = (size_t)mesh.mVbOffset + indices[base + 0];
        const size_t v1 = (size_t)mesh.mVbOffset + indices[base + 1];
        const size_t v2 = (size_t)mesh.mVbOffset + indices[base + 2];
        if (v0 >= vertices.size() || v1 >= vertices.size() || v2 >= vertices.size())
        {
            break;
        }
        const Uv uv0 = unpackUvHost(vertices[v0].uv);
        const Uv uv1 = unpackUvHost(vertices[v1].uv);
        const Uv uv2 = unpackUvHost(vertices[v2].uv);

        // The whole triangle first. A cutout is mostly leaf and mostly gap, and
        // both answer here for four bytes instead of sixty-four.
        const float triU0 = std::min({ uv0.x, uv1.x, uv2.x });
        const float triU1 = std::max({ uv0.x, uv1.x, uv2.x });
        const float triV0 = std::min({ uv0.y, uv1.y, uv2.y });
        const float triV1 = std::max({ uv0.y, uv1.y, uv2.y });
        const omm::Coverage whole = classifyUvBox(triU0, triU1, triV0, triV1);
        if (whole != omm::Coverage::Mixed || !subdivide)
        {
            triangleIndices[tri] = omm::predefinedIndexFor(whole);
            if (whole == omm::Coverage::Opaque)
                ++summary.uniformOpaque;
            else if (whole == omm::Coverage::Transparent)
                ++summary.uniformTransparent;
            else
                ++summary.uniformUnknown;
            continue;
        }

        std::ranges::fill(scratch, (uint8_t)0);
        uint32_t firstState = 0xFFFFFFFFu;
        bool uniform = true;
        for (uint32_t micro = 0; micro < microCount; ++micro)
        {
            float2 b0, b1, b2;
            optixMicromapIndexToBaseBarycentrics(micro, level, b0, b1, b2);
            const Uv m0 = barycentricUv(uv0, uv1, uv2, b0);
            const Uv m1 = barycentricUv(uv0, uv1, uv2, b1);
            const Uv m2 = barycentricUv(uv0, uv1, uv2, b2);
            // uv is affine in the barycentrics, so the corners' box is the
            // microtriangle's box exactly -- no sampling and nothing missed
            // between the corners.
            const omm::Coverage coverage =
                classifyUvBox(std::min({ m0.x, m1.x, m2.x }), std::max({ m0.x, m1.x, m2.x }),
                              std::min({ m0.y, m1.y, m2.y }), std::max({ m0.y, m1.y, m2.y }));
            const uint32_t state = omm::microStateFor(coverage);
            omm::setMicroState(scratch.data(), micro, state);
            ++summary.microTriangles;
            if (coverage != omm::Coverage::Mixed)
            {
                ++summary.microResolved;
            }
            if (firstState == 0xFFFFFFFFu)
                firstState = state;
            else if (state != firstState)
                uniform = false;
        }

        if (uniform)
        {
            // Subdividing found what the whole-triangle box could not prove, or
            // nothing at all. Either way one index says it.
            const omm::Coverage collapsed = firstState == omm::kStateOpaque      ? omm::Coverage::Opaque :
                                            firstState == omm::kStateTransparent ? omm::Coverage::Transparent :
                                                                                   omm::Coverage::Mixed;
            triangleIndices[tri] = omm::predefinedIndexFor(collapsed);
            if (collapsed == omm::Coverage::Opaque)
                ++summary.uniformOpaque;
            else if (collapsed == omm::Coverage::Transparent)
                ++summary.uniformTransparent;
            else
                ++summary.uniformUnknown;
            continue;
        }

        OptixOpacityMicromapDesc desc = {};
        desc.byteOffset = (unsigned int)micromapData.size();
        desc.subdivisionLevel = (unsigned short)level;
        desc.format = (unsigned short)OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
        triangleIndices[tri] = (int32_t)micromapDescs.size();
        micromapDescs.push_back(desc);
        micromapData.insert(micromapData.end(), scratch.begin(), scratch.end());
        ++summary.subdivided;
    }

    if (summary.isPointless())
    {
        STRELKA_DEBUG("Mesh {}: opacity micromap resolves nothing, skipped", meshIndex);
        return out;
    }

    // The array, when any triangle needed one of its own.
    if (!micromapDescs.empty())
    {
        CUdeviceptr d_input = 0;
        CUdeviceptr d_descs = 0;
        CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(d_input), micromapData.size()));
        CUDA_CHECK(cudaMemcpy(optix::devicePtr<void>(d_input), micromapData.data(), micromapData.size(),
                              cudaMemcpyHostToDevice));
        const size_t descBytes = micromapDescs.size() * sizeof(OptixOpacityMicromapDesc);
        CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(d_descs), descBytes));
        CUDA_CHECK(
            cudaMemcpy(optix::devicePtr<void>(d_descs), micromapDescs.data(), descBytes, cudaMemcpyHostToDevice));

        OptixOpacityMicromapHistogramEntry histogram = {};
        histogram.count = (unsigned int)micromapDescs.size();
        histogram.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
        histogram.subdivisionLevel = level;

        OptixOpacityMicromapArrayBuildInput arrayInput = {};
        arrayInput.flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
        arrayInput.inputBuffer = d_input;
        arrayInput.perMicromapDescBuffer = d_descs;
        arrayInput.numMicromapHistogramEntries = 1;
        arrayInput.micromapHistogramEntries = &histogram;

        OptixMicromapBufferSizes sizes = {};
        OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(mState.context, &arrayInput, &sizes));

        CUdeviceptr d_temp = 0;
        CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(out.array), sizes.outputSizeInBytes));
        CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(d_temp), std::max<size_t>(sizes.tempSizeInBytes, 1)));

        OptixMicromapBuffers buffers = {};
        buffers.output = out.array;
        buffers.outputSizeInBytes = sizes.outputSizeInBytes;
        buffers.temp = d_temp;
        buffers.tempSizeInBytes = sizes.tempSizeInBytes;
        OPTIX_CHECK(optixOpacityMicromapArrayBuild(mState.context, mState.stream, &arrayInput, &buffers));
        CUDA_CHECK(cudaStreamSynchronize(mState.stream));

        CUDA_CHECK(cudaFree(optix::devicePtr<void>(d_temp)));
        CUDA_CHECK(cudaFree(optix::devicePtr<void>(d_input)));
        CUDA_CHECK(cudaFree(optix::devicePtr<void>(d_descs)));
        out.arrayBytes = sizes.outputSizeInBytes;

        out.usage.push_back(
            OptixOpacityMicromapUsageCount{ (unsigned int)summary.subdivided, level,
                                            OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE });
    }

    const size_t indexBytes = triangleIndices.size() * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(out.indices), indexBytes));
    CUDA_CHECK(
        cudaMemcpy(optix::devicePtr<void>(out.indices), triangleIndices.data(), indexBytes, cudaMemcpyHostToDevice));
    out.valid = true;
    mOmmTotalBytes += out.arrayBytes;

    STRELKA_DEBUG("Mesh {}: opacity micromap level {} -- {} triangles, {} opaque, {} cut away, {} left to the "
                  "shader, {} subdivided; {}/{} microtriangles resolved ({} KB)",
                  meshIndex, summary.subdivisionLevel, summary.triangles, summary.uniformOpaque,
                  summary.uniformTransparent, summary.uniformUnknown, summary.subdivided, summary.microResolved,
                  summary.microTriangles, out.arrayBytes / 1024);
    return out;
}

void OptiXRender::releaseOpacityMicromapScratch(MeshOpacityMicromap& omm)
{
    // The index buffer is read by the build and never again; the array is read
    // during traversal and belongs to the Mesh.
    if (omm.indices)
    {
        CUDA_CHECK(cudaFree(optix::devicePtr<void>(omm.indices)));
        omm.indices = 0;
    }
}

std::unique_ptr<OptiXRender::Mesh> OptiXRender::createMesh(const oka::Mesh& mesh, size_t meshIndex)
{
    const bool isSkeletal = mesh.isSkeletal;

    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr d_gas_output_buffer = 0;

    // A static mesh no longer asks for ALLOW_UPDATE. Nothing refits one --
    // updateBottomLevelAccelerationStructures() skips every mesh that is not
    // skeletal -- and a refittable structure is built with a topology that
    // survives being moved rather than one built to be traced, which costs both
    // memory and traversal for a capability this class of geometry never uses.
    const oka::optix_accel::Geometry geometryClass =
        isSkeletal ? oka::optix_accel::Geometry::SkinnedMesh : oka::optix_accel::Geometry::StaticMesh;

    OptixAccelBuildOptions accel_options = {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = oka::optix_accel::buildFlags(geometryClass);

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

    // Opacity micromaps, when this mesh is an alpha cutout and the feature is on.
    // Built before the memory usage is asked for, because attaching them changes
    // what the structure costs.
    MeshOpacityMicromap omm = buildMeshOpacityMicromap(mesh, meshIndex);
    if (omm.valid)
    {
        triangle_input.triangleArray.opacityMicromap.indexingMode =
            OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
        triangle_input.triangleArray.opacityMicromap.opacityMicromapArray = omm.array;
        triangle_input.triangleArray.opacityMicromap.indexBuffer = omm.indices;
        // 32-bit indices: the four predefined states are negative, and a mesh
        // may hold more micromaps than a signed 16-bit index can name. Four
        // bytes a triangle next to the sixty-four a micromap costs is not the
        // place to economise.
        triangle_input.triangleArray.opacityMicromap.indexSizeInBytes = 4;
        triangle_input.triangleArray.opacityMicromap.numMicromapUsageCounts = (unsigned int)omm.usage.size();
        triangle_input.triangleArray.opacityMicromap.micromapUsageCounts = omm.usage.empty() ? nullptr : omm.usage.data();
    }

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

    if (!mTempAccelBuffer || mTempAccelBuffer->size() < gas_buffer_sizes.tempSizeInBytes)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(gas_buffer_sizes.tempSizeInBytes);
    }
    if (!mCompactedSizeBuffer || mCompactedSizeBuffer->size() < sizeof(uint64_t))
    {
        mCompactedSizeBuffer = std::make_unique<OptixBuffer>(sizeof(uint64_t));
    }

    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    // What the structure costs before compaction, for the memory report. The
    // compacted size, when there is one, overwrites it below.
    size_t gasBytes = gas_buffer_sizes.outputSizeInBytes;
    if (!oka::optix_accel::shouldCompact(geometryClass))
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

    releaseOpacityMicromapScratch(omm);

    auto rmesh = std::make_unique<Mesh>();
    rmesh->d_gas_output_buffer = d_gas_output_buffer;
    rmesh->gas_handle = gas_handle;
    rmesh->gas_bytes = gasBytes;
    // Taken over even when the build did not use it, so nothing leaks on a path
    // that returned early.
    rmesh->d_omm_array = omm.array;
    rmesh->omm_bytes = omm.arrayBytes;
    return rmesh;
}

void OptiXRender::createBottomLevelAccelerationStructures()
{
    // Clear existing acceleration structures
    mOptixMeshes.clear();
    mOptixCurves.clear();
    beginOpacityMicromaps();

    // Create BLAS for meshes
    const auto& meshes = mScene->getMeshes();
    mOptixMeshes.reserve(mScene->getMeshes().size());
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        mOptixMeshes.emplace_back(createMesh(meshes[i], i));
    }
    endOpacityMicromaps();

    // Create BLAS for curves
    const auto& curves = mScene->getCurves();
    mOptixCurves.reserve(curves.size());
    for (const auto& curve : curves)
    {
        mOptixCurves.emplace_back(createCurve(curve));
    }
}

/// Read the setting once and work out what each mesh is drawn with, before any
/// structure is built.
///
/// Read through contains() because nothing is obliged to set it: a host that
/// never heard of the feature gets it off, which is the default, rather than the
/// error getAs() logs for a key nobody wrote.
void OptiXRender::beginOpacityMicromaps()
{
    const SettingsManager* settings = getSettings();
    mOpacityMicromapsEnabled = settings->contains("render/pt/opacityMicromaps") &&
                               settings->getAs<bool>("render/pt/opacityMicromaps");
    mOmmTotalBytes = 0;
    mOmmAlphaCache.clear();
    mMeshMaterialIds.clear();
    if (mOpacityMicromapsEnabled)
    {
        resolveMeshMaterials();
    }
}

void OptiXRender::endOpacityMicromaps()
{
    // The decoded alpha channels are build-time scratch; a cutout atlas is tens
    // of megabytes and there is nothing to read it again for.
    mOmmAlphaCache.clear();
    if (mOpacityMicromapsEnabled && mOmmTotalBytes > 0)
    {
        STRELKA_INFO("Opacity micromaps: {} KB across {} meshes", mOmmTotalBytes / 1024, mOptixMeshes.size());
    }
}

void OptiXRender::updateMesh(const oka::Mesh& mesh, int optixMeshesId)
{
    OptixTraversableHandle& gas_handle = mOptixMeshes[optixMeshesId]->gas_handle;
    const CUdeviceptr& d_gas_output_buffer = mOptixMeshes[optixMeshesId]->d_gas_output_buffer;

    // The same flags the structure was built with. An update reads its output
    // buffer as the result of a build with exactly these; spelling them a second
    // time here is how they came to disagree.
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = oka::optix_accel::updateFlags(oka::optix_accel::Geometry::SkinnedMesh);
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

    // An update has its own scratch requirement, which is the one to honour
    // here; tempSizeInBytes is what a full build of the same input would need.
    const size_t updateTempSize = gas_buffer_sizes.tempUpdateSizeInBytes;
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < updateTempSize)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(updateTempSize);
    }

    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &triangle_input, 1,
                                mTempAccelBuffer->getPtr(), updateTempSize, d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
}

/// Rebuild one skeletal BLAS from scratch, into the storage it already has.
///
/// A refit keeps the tree the mesh had when it was built and only moves the
/// bounds, so a character that walks far enough from its bind pose ends up
/// traversing a tree that no longer fits it. A rebuild restores the quality,
/// and the input has not changed shape -- same vertex and index counts, same
/// flags -- so it produces a structure of exactly the same size and can go
/// straight back into the same buffer, with no allocation and no free.
///
/// The handle is returned by the build and is compared rather than assumed: a
/// GAS handle that moved invalidates every instance that names it, which is a
/// TLAS rebuild rather than a refit.
bool OptiXRender::rebuildMesh(const oka::Mesh& mesh, int optixMeshesId)
{
    OptixTraversableHandle& gas_handle = mOptixMeshes[optixMeshesId]->gas_handle;
    const CUdeviceptr d_gas_output_buffer = mOptixMeshes[optixMeshesId]->d_gas_output_buffer;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const CUdeviceptr vertexBuffer = mVertexBuffer->getPtr() + mesh.mVbOffset * sizeof(oka::Scene::Vertex);
    const CUdeviceptr indexBuffer = mIndexBuffer->getPtr() + mesh.mIndex * sizeof(uint32_t);

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

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    if (!mTempAccelBuffer || mTempAccelBuffer->size() < gas_buffer_sizes.tempSizeInBytes)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(gas_buffer_sizes.tempSizeInBytes);
    }

    const OptixTraversableHandle before = gas_handle;
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &triangle_input, 1,
                                mTempAccelBuffer->getPtr(), gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &gas_handle, nullptr, 0));
    return gas_handle != before;
}

/// Refit skeletal BLASes and rebuild them by bounded round robin.
///
/// Returns whether any BLAS handle moved, which the caller needs because an
/// instance naming a stale handle cannot be fixed by refitting the TLAS.
bool OptiXRender::updateBottomLevelAccelerationStructures()
{
    const auto& meshes = mScene->getMeshes();
    size_t rebuiltThisFrame = 0;
    bool handlesChanged = false;

    for (size_t index = 0; index < meshes.size(); ++index)
    {
        if (!meshes[index].isSkeletal)
        {
            continue;
        }
        if (rebuiltThisFrame < kMaxBlasRebuildsPerFrame && index >= mNextBlasRebuildIndex)
        {
            handlesChanged |= rebuildMesh(meshes[index], static_cast<int>(index));
            ++rebuiltThisFrame;
            mNextBlasRebuildIndex = index + 1;
        }
        else
        {
            updateMesh(meshes[index], static_cast<int>(index));
        }
    }

    // The pass ran out of meshes before it ran out of budget, so the next one
    // starts over at the front.
    if (rebuiltThisFrame < kMaxBlasRebuildsPerFrame)
    {
        mNextBlasRebuildIndex = 0;
    }
    return handlesChanged;
}

void OptiXRender::resolveInstanceGeometry(OptixInstance& oi, const oka::Instance& instance) const
{
    // Whether this instance can skip the shadow any-hit entirely.
    //
    // Per instance rather than scene-wide, because the scene-wide question
    // ("is there a cutout anywhere") is always yes in a forest and would drag
    // trunks, rocks and ground into the alpha callback with the needles. Curves
    // are always opaque: a strand has no uv to test.
    bool opaque = true;
    // Whether this instance is the boundary of a participating medium rather
    // than a surface. It goes on its own visibility bit, because a shadow ray
    // must not be stopped by a fog gizmo -- RAY_MASK_SHADOW is the geometry bits
    // alone -- and because the transmittance walk needs a traversal that finds
    // the boundaries and nothing else.
    bool mediumBoundary = false;
    if (instance.type == oka::Instance::Type::eMesh)
    {
        const auto& materials = mScene->getMaterials();
        const uint32_t materialId = (instance.mMaterialId == kInvalidIndex) ? 0u : instance.mMaterialId;
        if (materialId < materials.size())
        {
            opaque = materials[materialId].params.alpha_mode == ALPHA_MODE_OPAQUE;
            mediumBoundary = (materials[materialId].params.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u;
        }
    }
    oi.flags = opaque ? OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT : OPTIX_INSTANCE_FLAG_NONE;

    switch (instance.type)
    {
    case oka::Instance::Type::eMesh:
        oi.traversableHandle = mOptixMeshes[instance.mMeshId]->gas_handle;
        oi.visibilityMask = mediumBoundary ? GEOMETRY_MASK_MEDIUM : GEOMETRY_MASK_TRIANGLE;
        break;
    case oka::Instance::Type::eCurve:
        oi.traversableHandle = mOptixCurves[instance.mCurveId]->gas_handle;
        oi.visibilityMask = GEOMETRY_MASK_CURVE;
        break;
    case oka::Instance::Type::eLight:
    {
        oi.traversableHandle = mOptixMeshes[instance.mMeshId]->gas_handle;
        // A light's proxy mesh is how the light is picked in the editor and how a
        // BSDF ray finds an emitter for MIS. It is not always something to look at.
        //
        // Punctual and disabled lights have no visible proxy; camera-hidden area
        // lights remain visible to bounce rays for MIS.
        const auto& descs = mScene->getLightsDesc();
        const bool known = instance.mLightId < descs.size();
        const bool enabled = known ? descs[instance.mLightId].enabled : true;
        const bool visibleToCamera = known ? descs[instance.mLightId].visibleToCamera : true;
        const uint32_t lightType = known ? descs[instance.mLightId].type : (uint32_t)LIGHT_TYPE_RECT;
        // Smooth discs and ellipsoids are intersected analytically by the ray
        // programs. Suppress the coarse editor proxy so it cannot become a
        // nearer, incompatible surface.
        const bool analyticArea = lightUsesAnalyticAreaIntersection((int)lightType);
        const bool infinite = lightType == LIGHT_TYPE_DISTANT || lightType == LIGHT_TYPE_DOME;
        if (!enabled || lightTypeIsPunctual((int)lightType) || analyticArea || infinite)
        {
            oi.visibilityMask = 0;
        }
        else
        {
            oi.visibilityMask = visibleToCamera ? GEOMETRY_MASK_LIGHT : GEOMETRY_MASK_LIGHT_HIDDEN;
        }
        break;
    }
    default:
        STRELKA_ERROR("Unknown instance type");
        std::abort();
        break;
    }
}

bool OptiXRender::sceneHasBoundedMedium() const
{
    if (!mScene)
    {
        return false;
    }
    for (const auto& material : mScene->getMaterials())
    {
        if ((material.params.medium_flags & MEDIUM_FLAG_BOUNDARY) != 0u)
        {
            return true;
        }
    }
    return false;
}

void OptiXRender::uploadInstancesToDevice(const std::vector<OptixInstance>& optixInstances)
{
    const size_t instancesSize = sizeof(OptixInstance) * optixInstances.size();
    if (instancesSize != mState.d_instances_size)
    {
        if (mState.d_instances)
        {
            CUDA_CHECK(cudaFree(optix::devicePtr<void>(mState.d_instances)));
        }
        CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(mState.d_instances), instancesSize));
        mState.d_instances_size = instancesSize;
    }
    CUDA_CHECK(cudaMemcpy(
        optix::devicePtr<void>(mState.d_instances), optixInstances.data(), instancesSize, cudaMemcpyHostToDevice));
}

void OptiXRender::createTopLevelAccelerationStructure()
{
    // A full rebuild can change the instance count, and the SBT is indexed by
    // instance -- `sbtOffset = index * RAY_TYPE_COUNT` -- so every record after
    // the change points at the wrong geometry, and any beyond the end of the
    // table is read out of bounds. The SBT was built once at frame 0 and never
    // again, so this was live for every scene that adds or removes an instance.
    // A refit cannot change the count, which is why updateTopLevelAcceleration-
    // Structure() does not set this.
    mSbtDirty = true;

    // Pool fixed-size motion transforms to avoid per-frame allocation churn.
    size_t motionTransformCursor = 0;

    const std::vector<oka::Instance>& instances = mScene->getInstances();

    // Build OptixInstance array
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(instances.size());

    for (size_t instID = 0; instID < instances.size(); ++instID)
    {
        const auto& instance = instances[instID];
        OptixInstance oi = {};
        resolveInstanceGeometry(oi, instance);
        // Stable scene-instance identity for emissive-mesh hit-side PDF lookup.
        // sbtOffset is an address, not an identity, and may be rebuilt.
        oi.instanceId = static_cast<unsigned int>(instID);

        // If instance is animated, create linear matrix motion transform; else set transform directly
        if (mEnableMotionBlur && instance.isAnimated)
        {
            OptixMatrixMotionTransform matrixMotionTransform = {};
            OptixTraversableHandle matrixMotionTransformHandle = 0;

            matrixMotionTransform.child = oi.traversableHandle;
            matrixMotionTransform.motionOptions.numKeys = NUM_MOTION_KEYS;
            matrixMotionTransform.motionOptions.flags = OPTIX_MOTION_FLAG_NONE;
            matrixMotionTransform.motionOptions.timeBegin = 0.0f;
            matrixMotionTransform.motionOptions.timeEnd = 1.0f;

            memcpy(matrixMotionTransform.transform[0],
                   glm::value_ptr(glm::float3x4(glm::rowMajor4(mPrevInstances[instID].transform))), sizeof(float) * 12);
            memcpy(matrixMotionTransform.transform[1],
                   glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);

            if (motionTransformCursor >= mMotionTransformBuffers.size())
            {
                mMotionTransformBuffers.push_back(
                    std::make_shared<OptixBuffer>(sizeof(OptixMatrixMotionTransform)));
            }
            const std::shared_ptr<OptixBuffer>& motionTransformBuffer =
                mMotionTransformBuffers[motionTransformCursor++];
            CUDA_CHECK(cudaMemcpy(motionTransformBuffer->getNativePtr(), &matrixMotionTransform,
                                  sizeof(OptixMatrixMotionTransform), cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixConvertPointerToTraversableHandle(mState.context, motionTransformBuffer->getPtr(),
                                                               OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                                                               &matrixMotionTransformHandle));

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
    createEmissiveMeshLights();
    mTlasInstanceCount = optixInstances.size();

    // Setup IAS build input
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = static_cast<int>(optixInstances.size());

    // With motion blur on, the instance structure is rebuilt every frame rather
    // than refit -- the motion transforms it points at are rewritten -- so it
    // takes the static flags and is compacted. Without it, the structure is refit
    // as transforms move, and a refittable structure must not be compacted: an
    // update reads its output buffer as the build's own output, at the build's
    // own size, and optixAccelCompact replaces both. That is what this code used
    // to do, and it then refit the compacted copy in place.
    const oka::optix_accel::Geometry tlasClass =
        mEnableMotionBlur ? oka::optix_accel::Geometry::StaticTlas : oka::optix_accel::Geometry::RefittableTlas;

    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = oka::optix_accel::buildFlags(tlasClass);
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    // Allocate buffers
    size_t outputBufferSize = iasBufferSizes.outputSizeInBytes;
    if (!mTlasBuffer || mTlasBuffer->size() < outputBufferSize)
    {
        mTlasBuffer = std::make_unique<OptixBuffer>(outputBufferSize);
    }

    if (!mTempAccelBuffer || mTempAccelBuffer->size() < iasBufferSizes.tempSizeInBytes)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(iasBufferSizes.tempSizeInBytes);
    }
    if (!mCompactedSizeBuffer || mCompactedSizeBuffer->size() < sizeof(uint64_t))
    {
        mCompactedSizeBuffer = std::make_unique<OptixBuffer>(sizeof(uint64_t));
    }

    const bool compactTlas = oka::optix_accel::shouldCompact(tlasClass);

    // Setup compaction property
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = mCompactedSizeBuffer->getPtr();

    // Build IAS
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &iasOptions, &iasInput,
                                1, // num build inputs
                                mTempAccelBuffer->getPtr(), iasBufferSizes.tempSizeInBytes, mTlasBuffer->getPtr(),
                                outputBufferSize, &mState.ias_handle, compactTlas ? &property : nullptr,
                                compactTlas ? 1 : 0 // num emitted properties
                                ));

    if (!compactTlas)
    {
        // The refittable case. mTlasBuffer holds the build's own output at the
        // build's own size, which is what updateTopLevelAccelerationStructure()
        // needs to find there.
        mTlasOutputSize = outputBufferSize;
        return;
    }

    // Compact acceleration structure
    size_t compactedSize = 0;
    CUDA_CHECK(cudaMemcpy(&compactedSize, optix::devicePtr<void>(property.result), sizeof(size_t), cudaMemcpyDeviceToHost));

    // Only compact if it saves space
    if (compactedSize < outputBufferSize)
    {
        // Create new buffer for compacted data
        std::unique_ptr<OptixBuffer> compactedBuffer(new OptixBuffer(compactedSize));

        // Compact acceleration structure into new buffer
        OPTIX_CHECK(optixAccelCompact(mState.context, nullptr, mState.ias_handle,
                                     compactedBuffer->getPtr(), compactedSize, &mState.ias_handle));

        mTlasBuffer = std::move(compactedBuffer);
        outputBufferSize = compactedSize;
    }
    mTlasOutputSize = outputBufferSize;
}

void oka::OptiXRender::updateTopLevelAccelerationStructure()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();

    // Build OptixInstance array (no motion blur for refit)
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(instances.size());

    for (size_t instID = 0; instID < instances.size(); ++instID)
    {
        const oka::Instance& instance = instances[instID];
        OptixInstance oi = {};
        resolveInstanceGeometry(oi, instance);
        oi.instanceId = static_cast<unsigned int>(instID);
        memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);
        oi.sbtOffset = static_cast<unsigned int>(optixInstances.size() * RAY_TYPE_COUNT);
        optixInstances.push_back(oi);
    }

    uploadInstancesToDevice(optixInstances);
    createEmissiveMeshLights();

    // Setup IAS build (refit) input
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = static_cast<int>(optixInstances.size());

    // The flags the refittable structure was built with -- not a second guess at
    // them. They used to read PREFER_FAST_BUILD here against a build that said
    // PREFER_FAST_TRACE | ALLOW_COMPACTION, and an update is documented against
    // the build's own flags.
    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = oka::optix_accel::updateFlags(oka::optix_accel::Geometry::RefittableTlas);
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    const size_t updateTempSize = iasBufferSizes.tempUpdateSizeInBytes;
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < updateTempSize)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(updateTempSize);
    }

    // Build (refit) IAS. The output size is the size the build wrote, which is
    // not mTlasBuffer->size(): that buffer is reused across scenes and only ever
    // grows.
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &iasOptions, &iasInput,
                                1, // num build inputs
                                mTempAccelBuffer->getPtr(), updateTempSize, mTlasBuffer->getPtr(),
                                mTlasOutputSize, &mState.ias_handle, nullptr,
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

    // Leave registers unlimited by default; the environment knob is diagnostic.
    moduleOptions.maxRegisterCount = static_cast<int>(envUint("STRELKA_OPTIX_MAX_REGISTERS", 0));

    // Compile the launch parameters that select whole features in as constants,
    // so the branches they gate -- and everything under those branches -- are
    // dead code the compiler removes rather than instructions the kernel carries
    // past. See PipelineSpec for what is bound and, more importantly, what is
    // deliberately not.
    //
    // The values have to outlive optixModuleCreate, which is why they are named
    // locals rather than temporaries in the initialiser list.
    const PipelineSpec& spec = mPipelineSpec;
    const OptixModuleCompileBoundValueEntry boundValues[] = {
#define STRELKA_BOUND_VALUE(field)                                                                                     \
    OptixModuleCompileBoundValueEntry                                                                                  \
    {                                                                                                                  \
        offsetof(Params, field), sizeof(Params::field), &spec.field, "params." #field                                  \
    }
        STRELKA_BOUND_VALUE(sharcCapacity),      STRELKA_BOUND_VALUE(sharcResponsive),
        STRELKA_BOUND_VALUE(debug),
        STRELKA_BOUND_VALUE(estimatorMode),      STRELKA_BOUND_VALUE(volumeModel),
        STRELKA_BOUND_VALUE(misHeuristic),       STRELKA_BOUND_VALUE(subsurfaceIterations),
        STRELKA_BOUND_VALUE(risCandidates),      STRELKA_BOUND_VALUE(denoiseDepthMode),
        STRELKA_BOUND_VALUE(hasBoundedMedium),   STRELKA_BOUND_VALUE(hasFog),
        STRELKA_BOUND_VALUE(enableMotionBlur),   STRELKA_BOUND_VALUE(writeAov),
        STRELKA_BOUND_VALUE(writeSplitAov),      STRELKA_BOUND_VALUE(guidePrimaryHit),
        STRELKA_BOUND_VALUE(hasEnvMap),          STRELKA_BOUND_VALUE(hasEnvBackground),
        STRELKA_BOUND_VALUE(enableShaderReorder),
#undef STRELKA_BOUND_VALUE
    };
    moduleOptions.boundValues = boundValues;
    moduleOptions.numBoundValues = static_cast<unsigned int>(std::size(boundValues));

    // Setup pipeline compilation options
    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.usesMotionBlur = mEnableMotionBlur;
    pipelineOptions.traversableGraphFlags = mEnableMotionBlur ?
                                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY :
                                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = STRELKA_PAYLOAD_COUNT;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.exceptionFlags =
        mEnableValidation ?
            (OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW) :
            OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
                                             OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE |
                                             OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
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

    // Store options for later use. The bound-value table is a local, so the copy
    // that outlives this function must not keep pointing at it -- nothing reads
    // the stored options to compile with, and a dangling pointer that is only
    // dereferenced by a future caller is the kind that stays hidden.
    mState.pipeline_compile_options = pipelineOptions;
    mState.module_compile_options = moduleOptions;
    mState.module_compile_options.boundValues = nullptr;
    mState.module_compile_options.numBoundValues = 0;

    mPipelineSpecValid = true;

    // Create curve modules. One intersector per basis: the basis is compiled
    // into the built-in intersection program, so a scene that mixes linear and
    // cubic strands needs both, selected per instance in the SBT.
    OptixBuiltinISOptions builtinOptions = {};
    builtinOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    builtinOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;

    OPTIX_CHECK(optixBuiltinISModuleGet(
        mState.context, &moduleOptions, &pipelineOptions, &builtinOptions, &mState.m_catromCurveModule));

    builtinOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
    // The intersector has to know the strands were built with end caps, or it
    // will not test them and every strand loses its tip.
    builtinOptions.curveEndcapFlags = OPTIX_CURVE_ENDCAP_ON;
    OPTIX_CHECK(optixBuiltinISModuleGet(
        mState.context, &moduleOptions, &pipelineOptions, &builtinOptions, &mState.m_linearCurveModule));
}

void OptiXRender::createProgramGroups()
{
    const OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

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
    // The closest-hit module, not the raygen one: the miss program has to make a
    // next-event estimate when the atmosphere scatters the ray that was on its
    // way to the sky, and connectToLight is defined there.
    miss_prog_group_desc.miss.module = mState.closest_hit_module;
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
    OptixProgramGroup radiance_hit_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &radiance_hit_group));
    mState.radiance_default_hit_group = radiance_hit_group;

    // Same closest hit, the linear intersector.
    hit_prog_group_desc.hitgroup.moduleIS = mState.m_linearCurveModule;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log,
                                            &mState.radiance_linear_curve_hit_group));

    OptixProgramGroupDesc light_hit_prog_group_desc = {};
    light_hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    light_hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    light_hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__light";
    sizeof_log = sizeof(log);
    OptixProgramGroup light_hit_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &light_hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &light_hit_group));
    mState.light_hit_group = light_hit_group;

    memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    // The alpha test for shadow rays. It lives beside the material lookup in the
    // closest-hit module rather than with the occlusion closest hit, which is
    // the whole point of a hit group being able to mix modules. Instances whose
    // material is opaque carry OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT and never
    // reach it, so a forest floor is not dragged into the callback with the
    // leaves.
    hit_prog_group_desc.hitgroup.moduleAH = mState.closest_hit_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__occlusion";

    hit_prog_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr; // automatically supplied for built-in module

    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log, &mState.occlusion_hit_group));

    hit_prog_group_desc.hitgroup.moduleIS = mState.m_linearCurveModule;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(mState.context, &hit_prog_group_desc,
                                            1, // num program groups
                                            &program_group_options, log, &sizeof_log,
                                            &mState.occlusion_linear_curve_hit_group));
}

void OptiXRender::createPipeline()
{
    OptixPipeline pipeline = nullptr;
    const uint32_t max_trace_depth = 2;
    std::vector<OptixProgramGroup> program_groups = {};

    program_groups.push_back(mState.raygen_prog_group);
    program_groups.push_back(mState.radiance_miss_group);
    program_groups.push_back(mState.radiance_default_hit_group);
    program_groups.push_back(mState.radiance_linear_curve_hit_group);
    program_groups.push_back(mState.occlusion_miss_group);
    program_groups.push_back(mState.occlusion_hit_group);
    program_groups.push_back(mState.occlusion_linear_curve_hit_group);
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

    uint32_t direct_callable_stack_size_from_traversal = 0;
    uint32_t direct_callable_stack_size_from_state = 0;
    uint32_t continuation_stack_size = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDepth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    const int maxTraversableDepth = mEnableMotionBlur ? 3 : 2;
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          maxTraversableDepth));
    // Live PerRayData held across optixInvoke contributes to the per-thread
    // continuation stack, so log both sizes together.
    STRELKA_DEBUG("Pipeline stack: continuation={} B, dc_from_traversal={} B, dc_from_state={} B, PerRayData={} B",
                  continuation_stack_size, direct_callable_stack_size_from_traversal,
                  direct_callable_stack_size_from_state, sizeof(PerRayData));
    mState.pipeline = pipeline;
}

OptiXRender::PipelineSpec OptiXRender::specFor(const Params& params) const
{
    PipelineSpec spec;
    spec.sharcCapacity = params.sharcCapacity;
    spec.sharcResponsive = params.sharcResponsive;
    spec.debug = params.debug;
    spec.estimatorMode = params.estimatorMode;
    spec.volumeModel = params.volumeModel;
    spec.misHeuristic = params.misHeuristic;
    spec.subsurfaceIterations = params.subsurfaceIterations;
    spec.risCandidates = params.risCandidates;
    spec.denoiseDepthMode = params.denoiseDepthMode;
    spec.hasBoundedMedium = params.hasBoundedMedium;
    spec.hasFog = params.hasFog;
    spec.enableMotionBlur = params.enableMotionBlur;
    spec.writeAov = params.writeAov;
    spec.writeSplitAov = params.writeSplitAov;
    spec.guidePrimaryHit = params.guidePrimaryHit;
    spec.hasEnvMap = params.hasEnvMap;
    spec.hasEnvBackground = params.hasEnvBackground;
    spec.enableShaderReorder = params.enableShaderReorder;
    return spec;
}

void OptiXRender::destroyPipeline()
{
    // Pipeline first, then the groups it links, then the modules they name.
    // Handles are cleared as they go: a rebuild that throws half way must not
    // leave the launch path holding a destroyed pipeline.
    if (mState.pipeline)
    {
        OPTIX_CHECK(optixPipelineDestroy(mState.pipeline));
        mState.pipeline = nullptr;
    }
    for (OptixProgramGroup* group : { &mState.raygen_prog_group, &mState.radiance_miss_group,
                                      &mState.radiance_default_hit_group, &mState.radiance_linear_curve_hit_group,
                                      &mState.occlusion_miss_group, &mState.occlusion_hit_group,
                                      &mState.occlusion_linear_curve_hit_group, &mState.light_hit_group })
    {
        if (*group)
        {
            OPTIX_CHECK(optixProgramGroupDestroy(*group));
            *group = nullptr;
        }
    }
    for (OptixModule* module : { &mState.ptx_module, &mState.closest_hit_module, &mState.m_catromCurveModule,
                                 &mState.m_linearCurveModule })
    {
        if (*module)
        {
            OPTIX_CHECK(optixModuleDestroy(*module));
            *module = nullptr;
        }
    }
}

void OptiXRender::ensurePipelineSpecialization(const Params& params)
{
    const PipelineSpec wanted = specFor(params);
    if (mPipelineSpecValid && wanted == mPipelineSpec)
    {
        return;
    }

    // A recompile is seconds of driver work, and it invalidates every handle the
    // in-flight launch is using. Nothing may still be running against the
    // pipeline about to be destroyed.
    if (mState.stream)
    {
        latchCudaError(cudaStreamSynchronize(mState.stream), "drain the stream before respecialising the pipeline");
    }
    latchCudaError(cudaDeviceSynchronize(), "drain the device before respecialising the pipeline");

    const bool firstBuild = !mPipelineSpecValid;
    mPipelineSpec = wanted;

    const auto begin = std::chrono::steady_clock::now();
    destroyPipeline();
    createModule();
    createProgramGroups();
    createPipeline();
    // The SBT holds program group headers, and every one of them just moved.
    createSbt();
    const auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count();

    // Logged rather than silent: this is the one thing in the frame that can
    // cost a second, and a scene or a setting that makes it happen every frame
    // would otherwise read as "the renderer became slow".
    STRELKA_INFO("ACTION pipeline_specialize reason={} took_ms={} sharc={} medium={} fog={} motion={} aov={} debug={}",
                 firstBuild ? "first_build" : "spec_changed", ms, wanted.sharcCapacity != 0u, wanted.hasBoundedMedium,
                 wanted.hasFog, wanted.enableMotionBlur, wanted.writeAov, wanted.debug);
}

void OptiXRender::createSbt()
{
    // Free previous SBT records if they exist
    if (mState.sbt.raygenRecord)
        CUDA_CHECK(cudaFree(optix::devicePtr<void>(mState.sbt.raygenRecord)));
    if (mState.sbt.missRecordBase)
        CUDA_CHECK(cudaFree(optix::devicePtr<void>(mState.sbt.missRecordBase)));
    if (mState.sbt.hitgroupRecordBase)
        CUDA_CHECK(cudaFree(optix::devicePtr<void>(mState.sbt.hitgroupRecordBase)));
    mState.sbt = {};
    mSbtDirty = false;

    // Create raygen record
    CUdeviceptr raygen_record = 0;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(raygen_record), raygen_record_size));

    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(optix::devicePtr<void>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    // Create miss records
    CUdeviceptr miss_record = 0;
    const size_t miss_record_size = sizeof(MissSbtRecord) * RAY_TYPE_COUNT;
    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(miss_record), miss_record_size));

    std::vector<MissSbtRecord> miss_records(RAY_TYPE_COUNT);

    // Radiance miss record
    MissSbtRecord& radiance_miss = miss_records[RAY_TYPE_RADIANCE];
    radiance_miss.data.bg_color = mMissColor;
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_miss_group, &radiance_miss));

    // Occlusion miss record
    MissSbtRecord& occlusion_miss = miss_records[RAY_TYPE_OCCLUSION];
    // Named rather than brace-initialised: MissData wraps a float3, so `{0,0,0}`
    // needs a nested brace and only ever compiled by accident.
    occlusion_miss.data.bg_color = make_float3(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_miss_group, &occlusion_miss));

    CUDA_CHECK(cudaMemcpy(
        optix::devicePtr<void>(miss_record), miss_records.data(), miss_record_size, cudaMemcpyHostToDevice));

    // Create hit group records
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    const uint32_t hit_group_count = std::max(1u, static_cast<uint32_t>(instances.size())) * RAY_TYPE_COUNT;
    const size_t hit_group_size = sizeof(HitGroupSbtRecord) * hit_group_count;

    std::vector<HitGroupSbtRecord> hit_groups(hit_group_count);
    CUdeviceptr hit_group_record = 0;
    CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(hit_group_record), hit_group_size));

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
            const uint32_t material_idx = (instance.mMaterialId == kInvalidIndex) ? 0u : instance.mMaterialId;
            // Curves not yet built use the cubic group temporarily and dirty the
            // table so render() rebuilds it before tracing them.
            const bool curveReady = instance.type == oka::Instance::Type::eCurve &&
                                    instance.mCurveId < mOptixCurves.size() &&
                                    mOptixCurves[instance.mCurveId] != nullptr;
            if (instance.type == oka::Instance::Type::eCurve && !curveReady)
            {
                mSbtDirty = true;
            }
            const bool linearCurve = curveReady && mOptixCurves[instance.mCurveId]->isLinear;

            // Radiance hit group
            HitGroupSbtRecord& radiance_hit = hit_groups[i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE];

            if (instance.type == oka::Instance::Type::eLight)
            {
                radiance_hit.data.lightId = static_cast<int32_t>(instance.mLightId);
                OPTIX_CHECK(optixSbtRecordPackHeader(mState.light_hit_group, &radiance_hit));
            }
            else
            {
                OPTIX_CHECK(optixSbtRecordPackHeader(
                    linearCurve ? mState.radiance_linear_curve_hit_group : mState.radiance_default_hit_group,
                    &radiance_hit));
                radiance_hit.data.lightId = -1;
            }

            // Material is looked up from device buffer by materialId
            radiance_hit.data.materialId = static_cast<int32_t>(material_idx);

            // Set mesh data if applicable
            if (instance.type == oka::Instance::Type::eMesh)
            {
                const oka::Mesh& mesh = meshes[instance.mMeshId];
                radiance_hit.data.indexCount = static_cast<int32_t>(mesh.mCount);
                radiance_hit.data.indexOffset = static_cast<int32_t>(mesh.mIndex);
                radiance_hit.data.vertexOffset = static_cast<int32_t>(mesh.mVbOffset);
            }
            else if (instance.type == oka::Instance::Type::eCurve)
            {
                // Same staging caveat as the intersector choice above: zero until
                // the curve set exists, and the table is rebuilt when it does.
                radiance_hit.data.curveSegmentsPerStrand =
                    curveReady ? mOptixCurves[instance.mCurveId]->segmentsPerStrand : 0u;
            }

            // Occlusion hit group. It carries the same payload as the radiance
            // one, which it did not use to: the shadow any-hit needs the
            // material to know whether the surface is a cutout, and the index
            // and vertex offsets to find the uv it must test. Left at zero, a
            // cutout shadow ray read triangle 0 of mesh 0 for every hit.
            HitGroupSbtRecord& occlusion_hit = hit_groups[i * RAY_TYPE_COUNT + RAY_TYPE_OCCLUSION];
            OPTIX_CHECK(optixSbtRecordPackHeader(
                linearCurve ? mState.occlusion_linear_curve_hit_group : mState.occlusion_hit_group,
                &occlusion_hit));
            occlusion_hit.data = radiance_hit.data;
            occlusion_hit.data.lightId = -1;
        }
    }

    CUDA_CHECK(cudaMemcpy(
        optix::devicePtr<void>(hit_group_record), hit_groups.data(), hit_group_size, cudaMemcpyHostToDevice));

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

/// Size the radiance cache, decide whether it needs clearing, and hand the
/// device the four numbers that drive it.
///
/// Off is `sharcCapacity == 0`, and the device code reads that before it reads
/// anything else in the group, so every other field is free to be stale when the
/// feature is off. That is what makes the default a byte-for-byte no-op instead
/// of a second code path that happens to agree.
void OptiXRender::updateSharcParams(const oka::Camera& camera, uint32_t width, uint32_t height)
{
    const SettingsManager& settings = *getSettings();
    Params& params = mState.params;

    const bool want = settings.contains("render/pt/sharc") && settings.getAs<bool>("render/pt/sharc");
    uint32_t capacity = 0;
    if (want)
    {
        capacity = settings.contains("render/pt/sharcCapacity") ?
                       settings.getAs<uint32_t>("render/pt/sharcCapacity") :
                       (1u << 22);
        capacity = std::max(oka::sharc::kMinCapacity, capacity);
        // The probe run masks rather than divides, so the table has to be a
        // power of two. Rounded down: a capacity somebody typed is a memory
        // budget, and rounding up could double it.
        while ((capacity & (capacity - 1u)) != 0u)
        {
            capacity &= capacity - 1u;
        }
    }

    if (capacity != mSharcCapacity)
    {
        mSharcBuffer.reset();
        mSharcCapacity = 0;
        if (capacity != 0)
        {
            mSharcBuffer = std::make_unique<OptixBuffer>((size_t)capacity * sizeof(SharcEntry));
            mSharcCapacity = capacity;
            mSharcClearPending = true;
            STRELKA_INFO("Radiance cache: {} entries ({:.1f} MB)", capacity,
                         (double)capacity * sizeof(SharcEntry) / 1e6);
        }
    }

    // One record per pixel for the visit each path is allowed. Sized here rather
    // than carried in the payload: in PerRayData it was 28 bytes of continuation
    // stack on every path of every scene, cache or no cache. See SharcPathState.
    const size_t pathStateCount = (size_t)params.image_width * params.image_height;
    if (mSharcCapacity != 0 && (!mSharcPathBuffer || mSharcPathStateCount != pathStateCount))
    {
        mSharcPathBuffer = std::make_unique<OptixBuffer>(pathStateCount * sizeof(SharcPathState));
        mSharcPathStateCount = pathStateCount;
    }
    else if (mSharcCapacity == 0 && mSharcPathBuffer)
    {
        mSharcPathBuffer.reset();
        mSharcPathStateCount = 0;
    }

    params.sharcCapacity = mSharcCapacity;
    params.sharcEntries = mSharcCapacity ? (SharcEntry*)mSharcBuffer->getNativePtr() : nullptr;
    params.sharcPath = mSharcPathBuffer ? (SharcPathState*)mSharcPathBuffer->getNativePtr() : nullptr;

    // SHARC grid debug needs the scene-derived base size before cache allocation.
    const float aspect = height > 0 ? (float)width / (float)height : 1.0f;
    const float tanHalfFov = std::tan(glm::radians(camera.fovForAspect(aspect)) * 0.5f);
    const float pixelAngle = height > 0 ? 2.0f * tanHalfFov / (float)height : 1.0f;
    const float voxelPixels = settings.contains("render/pt/sharcVoxelPixels") ?
                                  settings.getAs<float>("render/pt/sharcVoxelPixels") :
                                  4.0f;
    params.sharcBaseSize = pixelAngle * std::max(1.0f, voxelPixels);

    if (mSharcCapacity == 0)
    {
        // Clear responsive pointers before returning to avoid stale device addresses.
        params.sharcResponsive = 0u;
        params.sharcResponsiveLights = nullptr;
        return;
    }

    params.sharcMinSamples =
        settings.contains("render/pt/sharcMinSamples") ? settings.getAs<uint32_t>("render/pt/sharcMinSamples") : 8u;
    params.sharcDepth = settings.contains("render/pt/sharcDepth") ? settings.getAs<uint32_t>("render/pt/sharcDepth") : 1u;
    params.sharcReadMaxSubframe = settings.contains("render/pt/sharcReadFrames") ?
                                      settings.getAs<uint32_t>("render/pt/sharcReadFrames") :
                                      128u;

    // The temporal window, and how long an unvisited entry survives. Both are
    // frames, both go to the resolve pass, and both are clamped there to the
    // SDK's bounds -- a stale threshold below kStaleFrameNumMin in particular
    // costs more in re-insertion than the table it frees is worth.
    mSharcAccumFrames = settings.contains("render/pt/sharcAccumFrames") ?
                            settings.getAs<uint32_t>("render/pt/sharcAccumFrames") :
                            32u;
    mSharcStaleFrames = settings.contains("render/pt/sharcStaleFrames") ?
                            settings.getAs<uint32_t>("render/pt/sharcStaleFrames") :
                            64u;
    mSharcResponsiveFrames = settings.contains("render/pt/sharcResponsiveFrames") ?
                                 settings.getAs<uint32_t>("render/pt/sharcResponsiveFrames") :
                                 4u;

    // Responsive lighting is a property of the scene, not a setting: it is on
    // exactly when some light asked for it. The setting below can only turn it
    // off, which is what makes it usable as an A/B against a scene that has one.
    const bool responsiveAllowed = !settings.contains("render/pt/sharcResponsiveLighting") ||
                                   settings.getAs<bool>("render/pt/sharcResponsiveLighting");
    const uint32_t responsive = (responsiveAllowed && mSharcResponsiveLightCount > 0) ? 1u : 0u;
    if (responsive != params.sharcResponsive)
    {
        // The split changes what an entry holds, so the entries that were filled
        // under the other setting are answers to a different question.
        mSharcClearPending = true;
    }
    params.sharcResponsive = responsive;
    // getNativePtr() rather than getPtr(): the latter hands back a CUdeviceptr,
    // and casting an integer to a pointer is both a tidy error here and a real
    // pessimisation. The device address is already a pointer on this side.
    params.sharcResponsiveLights =
        responsive ? static_cast<const uint32_t*>(mSharcResponsiveLightBuffer->getNativePtr()) : nullptr;

    // Kept for the resolve pass, which runs after the launch and has no camera.
    //
    // Derived exactly as the device derives it -- the translation of the inverse
    // view, which is what params.viewToWorld[3,7,11] holds a few lines below --
    // rather than from Camera::position. The two agree, but only one of them is
    // guaranteed to: the voxel level a shading point lands in and the level
    // reprojection looks for have to be computed from the same eye, and a
    // discrepancy there would show up as reprojection quietly finding nothing.
    {
        const glm::mat4 viewToWorld = glm::inverse(camera.matrices.view);
        mSharcCameraPosition[0] = viewToWorld[3][0];
        mSharcCameraPosition[1] = viewToWorld[3][1];
        mSharcCameraPosition[2] = viewToWorld[3][2];
    }

    // An explicit reset, from the editor's button or a config. Consumed here so
    // that holding the setting true does not clear the table every frame.
    if (settings.contains("render/pt/sharcReset") && settings.getAs<bool>("render/pt/sharcReset"))
    {
        mSharcClearPending = true;
        getSettings()->setAs<bool>("render/pt/sharcReset", false);
    }

    // The scene itself changing is a different matter from the camera moving:
    // the voxels still exist, but what they saw does not. resetTemporalHistory()
    // is the renderer's own declaration that nothing from before applies.
    if (mResetTemporalHistory)
    {
        mSharcClearPending = true;
    }

    if (mSharcClearPending)
    {
        CUDA_CHECK(cudaMemsetAsync(optix::devicePtr<void>(mSharcBuffer->getPtr()), 0,
                                   (size_t)mSharcCapacity * sizeof(SharcEntry), mState.stream));
        mSharcClearPending = false;
    }
}

/// Fold one frame of deposits into the cache. See sharc_resolve.h.
///
/// Ordering is the whole of the correctness here, and it is why this is a
/// separate pass rather than something the shading path does inline:
///
///     launch            paths atomicAdd into `accum`
///     -- stream order --
///     resolve           merges `accum` into `resolved`, ages, evicts, zeroes
///     -- stream order --
///     next launch       paths read `resolved`
///
/// Everything runs on mState.stream, so each stage sees the previous one's
/// writes without an explicit barrier -- the CUDA stream is the barrier. Moving
/// either of these off that stream reintroduces the race silently: the symptom
/// is a cache that reads a frame's partial sums, which looks like noise rather
/// than like a synchronisation bug.
void OptiXRender::resolveSharc()
{
    if (mSharcCapacity == 0 || !mSharcBuffer)
    {
        return;
    }

    SharcEntry* entries = static_cast<SharcEntry*>(mSharcBuffer->getNativePtr());

    SharcResolveParams resolveParams;
    resolveParams.capacity = mSharcCapacity;
    resolveParams.accumFrameNumMax = mSharcAccumFrames;
    resolveParams.staleFrameNumMax = mSharcStaleFrames;
    resolveParams.responsiveFrameNumMax = mSharcResponsiveFrames;
    for (int i = 0; i < 3; ++i)
    {
        resolveParams.cameraPosition[i] = mSharcCameraPosition[i];
        resolveParams.cameraPositionPrev[i] = mSharcPrevCameraPosition[i];
    }
    // Nothing to reproject on the first frame, and nothing to reproject from a
    // camera that has not moved -- in which case the probe would find the entry
    // itself half the time and a neighbour it has no business blending the rest.
    const float dx = mSharcCameraPosition[0] - mSharcPrevCameraPosition[0];
    const float dy = mSharcCameraPosition[1] - mSharcPrevCameraPosition[1];
    const float dz = mSharcCameraPosition[2] - mSharcPrevCameraPosition[2];
    resolveParams.reproject = mSharcHasPrevCameraPosition && (dx * dx + dy * dy + dz * dz) > 1e-12f;

    sharcResolve(entries, resolveParams, mState.stream);

    for (int i = 0; i < 3; ++i)
    {
        mSharcPrevCameraPosition[i] = mSharcCameraPosition[i];
    }
    mSharcHasPrevCameraPosition = true;

    // Occupancy, only while something is asking for it. The counter is read back
    // a frame late on purpose: the alternative is synchronising the render
    // stream for a number that goes into a debug panel.
    const SettingsManager& settings = *getSettings();
    const bool wantOccupancy =
        settings.contains("render/pt/sharcReportOccupancy") && settings.getAs<bool>("render/pt/sharcReportOccupancy");
    if (!wantOccupancy)
    {
        mSharcOccupancyCounter.reset();
        mSharcOccupancyEntries = 0;
        return;
    }
    if (!mSharcOccupancyCounter)
    {
        mSharcOccupancyCounter = std::make_unique<OptixBuffer>(sizeof(uint32_t));
    }
    uint32_t* counter = static_cast<uint32_t*>(mSharcOccupancyCounter->getNativePtr());
    // Last frame's count, before this frame overwrites it.
    CUDA_CHECK(cudaMemcpyAsync(&mSharcOccupancyEntries, counter, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                               mState.stream));
    sharcCountOccupancy(entries, mSharcCapacity, counter, mState.stream);
}

void OptiXRender::updatePathtracerParams(const uint32_t width, const uint32_t height)
{
    // The split buffers exist only while somebody is asking for them, so a
    // flipped Params::writeSplitAov is a reallocation the same way a resolution
    // change is.
    const bool splitAllocated = mState.params.diffuse != nullptr;
    bool needRealloc = splitAllocated != mState.params.writeSplitAov;
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
        mDisplayReadbackBuffer.reset();
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
        mState.params.diffuse = nullptr;
        mState.params.diffuseCounter = nullptr;
        mState.params.specular = nullptr;
        mState.params.specularCounter = nullptr;
        const size_t frameSize = static_cast<size_t>(mState.params.image_width) * mState.params.image_height;
        CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(mState.params.accum), frameSize * sizeof(float4)));

        if (mState.params.writeSplitAov)
        {
            CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(mState.params.diffuse), frameSize * sizeof(float4)));
            CUDA_CHECK(cudaMemset(mState.params.diffuse, 0, frameSize * sizeof(float4)));
            CUDA_CHECK(
                cudaMalloc(optix::deviceAllocTarget(mState.params.diffuseCounter), frameSize * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemset(mState.params.diffuseCounter, 0, frameSize * sizeof(uint16_t)));

            CUDA_CHECK(cudaMalloc(optix::deviceAllocTarget(mState.params.specular), frameSize * sizeof(float4)));
            CUDA_CHECK(cudaMemset(mState.params.specular, 0, frameSize * sizeof(float4)));
            CUDA_CHECK(
                cudaMalloc(optix::deviceAllocTarget(mState.params.specularCounter), frameSize * sizeof(uint16_t)));
            CUDA_CHECK(cudaMemset(mState.params.specularCounter, 0, frameSize * sizeof(uint16_t)));
        }
    }
}

// The plan's production size must match the struct written by the device.
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

bool OptiXRender::readDisplayTextureWithMaxOutput(std::vector<float>& out,
                                                  uint32_t& width,
                                                  uint32_t& height,
                                                  float maxOutput)
{
    size_t imageSize = 0;
    float4 *scratch = nullptr;
    float3 exposure;
    ToneMapperType tonemapperType = ToneMapperType::eNone;

    if (mDisplayImage == nullptr || mDisplayWidth == 0 || mDisplayHeight == 0)
    {
        return false;
    }
    width = mDisplayWidth;
    height = mDisplayHeight;
    out.resize(static_cast<size_t>(width) * height * 4);
    imageSize = out.size() * sizeof(float);
    if (!mDisplayReadbackBuffer)
    {
        mDisplayReadbackBuffer = std::make_unique<OptixBuffer>(imageSize);
    }
    else if (mDisplayReadbackBuffer->size() != imageSize)
    {
        mDisplayReadbackBuffer->realloc(imageSize);
    }
    scratch = static_cast<float4*>(mDisplayReadbackBuffer->getNativePtr());
    CUDA_CHECK(cudaMemcpy(scratch, mDisplayImage, imageSize, cudaMemcpyDeviceToDevice));
    if (mDisplayPresentation.content == PresentationContent::SceneLinear)
    {
        exposure = make_float3(mDisplayPresentation.exposure[0],
                               mDisplayPresentation.exposure[1],
                               mDisplayPresentation.exposure[2]);
        tonemapperType = static_cast<ToneMapperType>(mDisplayPresentation.tonemapper);
        tonemap(tonemapperType,
                exposure,
                maxOutput,
                0.0f,
                scratch,
                width,
                height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), scratch, imageSize, cudaMemcpyDeviceToHost));
    return true;
}

bool OptiXRender::readDisplayTexture(std::vector<float>& out, uint32_t& width, uint32_t& height)
{
    return readDisplayTextureWithMaxOutput(out, width, height, mDisplayPresentation.maxOutput);
}

bool OptiXRender::readDisplayTextureSdr(std::vector<float>& out, uint32_t& width, uint32_t& height)
{
    return readDisplayTextureWithMaxOutput(out, width, height, 1.0f);
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
        CUDA_CHECK(cudaMemcpy(out.data(), optix::devicePtr<const void>(mDenoiser.output()),
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

    const size_t jointMatSize = jointMat.size();
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

                cuApplySkinning(256, static_cast<int>(mesh.mVbOffset), static_cast<int>(mesh.mSbOffset),
                                optix::devicePtr<void>(mVertexBuffer->getPtr()),
                                optix::devicePtr<void>(mVertexSkinDataBuffer->getPtr()), mSkinningPtrs.d_jointMats,
                                jointMatOffset, mesh.mVertexCount);
            }
        }
    }
    // One mark for the whole set. A per-mesh breadcrumb would say which mesh's
    // dispatch died, but the kernel is the same one for all of them and the
    // memset would cost more than the dispatch it follows.
    markStageSubmitted(optix::GpuStage::Skinning, nullptr);
}

// Bounding-box diagonal of the skinned vertices, read back from the GPU.
//
// It exists because a character collapsing to a point is invisible to every
// whole-frame metric: a skeleton that never got its joint matrices renders as a
// speck at the origin, which is small next to its surroundings, so coverage and
// mean brightness barely move. The CI smoke test greps the line the CLI prints
// from this, so it has to fail loudly rather than return something plausible.
//
// -1 when nothing in the scene is skinned, which is the base class's "cannot
// answer"; -2 when a skinned position came back non-finite, because a NaN in
// the vertex buffer is a different fault from a collapse and reporting it as a
// zero extent would send the reader after the wrong one.
float OptiXRender::skinnedGeometryExtent()
{
    size_t first = SIZE_MAX, last = 0;
    for (const auto& m : mScene->getMeshes())
    {
        if (!m.isSkeletal)
            continue;
        first = std::min(first, (size_t)m.mVbOffset);
        last = std::max(last, (size_t)m.mVbOffset + (size_t)m.mVertexCount);
    }
    if (first == SIZE_MAX || !mVertexBuffer)
    {
        return -1.0f;
    }

    // The skinning kernel writes into whichever buffer is current, and with
    // motion blur on applySkinning() swaps the two every frame -- so reading
    // mVertexBuffer unconditionally is right, and reading the other one would
    // report the previous pose half the time.
    const size_t count = last - first;
    const size_t bytes = count * sizeof(oka::Scene::Vertex);
    if (count == 0 || mVertexBuffer->size() < (first + count) * sizeof(oka::Scene::Vertex))
    {
        return -1.0f;
    }

    std::vector<oka::Scene::Vertex> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(),
                          optix::devicePtr<const void>(mVertexBuffer->getPtr() +
                                                        first * sizeof(oka::Scene::Vertex)),
                          bytes, cudaMemcpyDeviceToHost));

    glm::float3 lo(1e30f), hi(-1e30f);
    for (const auto& v : host)
    {
        if (!std::isfinite(v.pos.x) || !std::isfinite(v.pos.y) || !std::isfinite(v.pos.z))
        {
            return -2.0f; // non-finite: a different fault from a collapse
        }
        lo = glm::min(lo, v.pos);
        hi = glm::max(hi, v.pos);
    }
    return glm::length(hi - lo);
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
            mJointMatOffsets.push_back(static_cast<int>(currJointMats.size()));
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

        // The environment -- the lighting map and the separate backdrop both --
        // is loaded by the build's own Environment stage, in
        // buildSceneEnvironment(), rather than here: it is one of the stages the
        // slice above is stepping through. What is left to do here is say how
        // much of the scene has arrived.
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
    const bool geometryChanged = any(changes & ChangeBits::Geometry);
    if (geometryChanged)
    {
        createVertexBuffer();
        createIndexBuffer();
        createBottomLevelAccelerationStructures();
        createTopLevelAccelerationStructure();
        createSbt();
    }
    if (any(changes & ChangeBits::Lights))
    {
        createLightBuffer();
        if (!geometryChanged)
        {
            createTopLevelAccelerationStructure();
        }
    }
    if (!geometryChanged && any(changes & ChangeBits::Transforms))
    {
        // Instance transforms changed outside the animation path
        createTopLevelAccelerationStructure();
    }
    if (any(changes & ChangeBits::Materials))
    {
        // The bit was consumed and nothing acted on it, so every material edit
        // made after load -- everything the editor's material panel does -- was
        // dropped on the floor: the device-side MaterialParams array is only
        // ever written by the build's material stages, which ran once while the
        // scene was streaming in.
        //
        // Both halves of that pair have to run: publishMaterialParams() releases
        // the old texture set and rewrites the parameter table with every slot
        // empty, and the texture stage is what fills the slots back in. An edit
        // is not a load -- it happens between two frames rather than across a
        // build -- so the slices are run to completion here instead of being
        // spread over frames. Instance flags are derived from alpha_mode, so the
        // TLAS has to follow.
        //
        // The progress sink is put aside for the duration: it belongs to a load,
        // and its cancellation flag stays set after an abandoned one, which would
        // otherwise stop this loop after a single material and leave the rest of
        // the scene's maps unloaded for the life of the session.
        LoadProgress* const progress = mLoadProgress;
        mLoadProgress = nullptr;
        publishMaterialParams();
        while (!stepMaterialTextures(std::numeric_limits<double>::max()))
        {
        }
        mLoadProgress = progress;
        createTopLevelAccelerationStructure();
    }
    if (any(changes & ChangeBits::Env))
    {
        updateSceneEnvironment();
        updateEmitterSelectionProbabilities();
        getSharedContext().mSubframeIndex = 0;
        mSharcClearPending = true;
    }
    if (any(changes &
            (ChangeBits::Lights | ChangeBits::Transforms | ChangeBits::Materials | ChangeBits::Geometry |
             ChangeBits::Env)))
    {
        mScene->consumeChanges();
    }

    const SettingsManager& settings = *getSettings();
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
        const float currAnimTime = settings.getAs<float>(scrollName);

        const float EPSILON = 1e-6f; // 0.000001

        if (std::abs(animations[i].current - currAnimTime) > EPSILON)
        {
            animStateChanged = true;
            animations[i].current = currAnimTime;
            accelStructureDirty |= mScene->applyAnimation(i);
        }
    }

    // Acceleration structure refit / rebuild.
    //
    // What decides between them is now a property of the structure rather than
    // a frame counter: a BLAS is refit unless it is this frame's turn in the
    // round-robin, and the TLAS is refit unless something a refit cannot express
    // has changed -- an instance count, or a child handle that moved under it.
    //
    // Motion blur is the exception on both. A motion GAS is built from two
    // vertex buffers and OPTIX_BUILD_OPERATION_UPDATE would have to be given the
    // same motion options and both buffers again; a motion IAS is built without
    // ALLOW_UPDATE at all. Both are rebuilt outright, which is what they already
    // did and why this branch is kept rather than merged.
    if (animStateChanged)
    {
        bool tlasNeedsRebuild = mEnableMotionBlur;
        if (accelStructureDirty)
        {
            applySkinning();
            if (mEnableMotionBlur)
            {
                createBottomLevelAccelerationStructures();
            }
            else
            {
                tlasNeedsRebuild |= updateBottomLevelAccelerationStructures();
            }
        }

        // An instance the TLAS does not have yet cannot be refit into it.
        tlasNeedsRebuild |= mScene->getInstances().size() != mTlasInstanceCount;

        if (tlasNeedsRebuild)
        {
            createTopLevelAccelerationStructure();
        }
        else
        {
            updateTopLevelAccelerationStructure();
        }
    }

    if (mSbtDirty)
    {
        createSbt();
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

    // Resolved before the buffers are sized, because it decides whether two of
    // them exist at all. Turning it on mid-render restarts accumulation: the
    // split's history is memset by the allocation below, and only subframe 0
    // resets it in the raygen, so the frames before the switch would be missing
    // from it while the beauty image counted them.
    const bool writeSplitAov =
        settings.contains("render/pt/splitAov") && settings.getAs<bool>("render/pt/splitAov");
    settingsChanged |= writeSplitAov != mState.params.writeSplitAov;
    mState.params.writeSplitAov = writeSplitAov;

    updatePathtracerParams(width, height);
    updateGuideBuffers(plan);

    const uint32_t selectedCameraIdx = settings.getAs<uint32_t>("render/selectedCamera");
    oka::Camera& camera = mScene->getCamera(selectedCameraIdx < mScene->getCameraCount() ? selectedCameraIdx : 0);
    camera.updateAspectRatio(static_cast<float>(outputWidth) / static_cast<float>(outputHeight));
    camera.updateViewMatrix();

    View currView = {};

    currView.mCamMatrices = camera.matrices;

    // Capture the previous rendered camera before the latch below mutates it;
    // reprojection uses this pose to build prevWorldToClip.
    const View prevView = mPrevView;

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
    params.scene.vb = optix::devicePtr<Vertex>(mVertexBuffer->getPtr());

    if (mEnableMotionBlur)
        params.scene.vb_prev = optix::devicePtr<Vertex>(mPrevVertexBuffer->getPtr());
    params.enableMotionBlur = mEnableMotionBlur;
    settingsChanged |= (params.isMotionBlurVisible != settings.getAs<bool>("render/isMotionBlurVisible"));
    params.isMotionBlurVisible = settings.getAs<bool>("render/isMotionBlurVisible");

    if (settingsChanged || animStateChanged)
    {
        getSharedContext().mSubframeIndex = 0;
    }

    params.scene.ib = optix::devicePtr<uint32_t>(mIndexBuffer->getPtr());
    params.scene.lights = optix::devicePtr<UniformLight>(mLightBuffer->getPtr());
    params.scene.numLights = mScene->getLights().size();
    updateEmitterSelectionProbabilities();
    // createLightBuffer() always leaves this populated -- with a zero-count
    // header when the scene has no profile -- so the shading path never sees
    // null here. Guarded anyway: a launch before the first scene build would.
    params.scene.iesProfiles =
        mIesBuffer ? optix::devicePtr<const IesGpuBufferHeader>(mIesBuffer->getPtr()) : nullptr;

    // When the 2x model is upscaling, the caller's buffer is twice the size the
    // path tracer runs at, so the tracer writes into its own buffer and the
    // denoiser is what fills the caller's.
    params.image = plan.upscale ? (float4*)mRenderImageBuffer->getNativePtr() :
                                  (float4*)((OptixBuffer*)output)->getNativePtr();
    params.samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    params.handle = mState.ias_handle;
    // Clamped to what PerRayData::depth counts to. That field is nine bits of a
    // packed word rather than a whole one (see the note on PerRayData), and the
    // closest hit stops a path by setting it *to* max_depth, which the raygen
    // then increments once more -- so the ceiling is 255, not 511. The clamp is
    // here so the narrowing is a bound the renderer states rather than a wrap a
    // caller discovers; no scene asks for a path 255 bounces long, and the
    // ladder's deepest row is 16.
    params.max_depth = std::min(settings.getAs<uint32_t>("render/pt/depth"), 255u);

    params.rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    params.enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    params.debug = settings.getAs<uint32_t>("render/pt/debug");
    params.shadowRayTmin = settings.getAs<float>("render/pt/dev/shadowRayTmin");
    params.materialRayTmin = settings.getAs<float>("render/pt/dev/materialRayTmin");
    params.misHeuristic = settings.getAs<uint32_t>("render/pt/misHeuristic");
    params.volumeModel = settings.getAs<uint32_t>("render/material/volumeModel");
    // A kill switch on top of the capability: reordering cannot change an
    // image, so the only reason to turn it off on hardware that has it is to
    // measure what it is worth.
    params.enableShaderReorder =
        mShaderReorderSupported &&
        (!settings.contains("render/pt/shaderReorder") || settings.getAs<bool>("render/pt/shaderReorder"));

    // Estimator controls. Each one resets accumulation when it moves, because the
    // frames before and after are estimates of different things (estimatorMode,
    // clampIndirect) or drawn from different densities (risCandidates), and
    // averaging them together hides the very difference they exist to show.
    const uint32_t risCandidates = std::max(settings.getAs<uint32_t>("render/pt/risCandidates"), 1u);
    const uint32_t estimatorMode = settings.getAs<uint32_t>("render/validate/estimatorMode");
    const float clampIndirect = settings.getAs<float>("render/pt/clampIndirect");
    if (params.risCandidates != risCandidates || params.estimatorMode != estimatorMode ||
        params.clampIndirect != clampIndirect)
    {
        getSharedContext().mSubframeIndex = 0;
    }
    params.risCandidates = risCandidates;
    params.estimatorMode = estimatorMode;
    params.clampIndirect = clampIndirect;

    // How long one random walk may be, and whether any walk can happen at all.
    // The step ceiling is the same setting that sizes the Metal wavefront's
    // extra dispatch iterations, so a walk gets the same budget on both
    // backends; `hasBoundedMedium` gates the second traversal a shadow ray takes
    // to accumulate optical depth, which is pure cost in a scene with no gizmo.
    params.subsurfaceIterations =
        settings.contains("render/pt/subsurfaceIterations")
            ? std::min(settings.getAs<uint32_t>("render/pt/subsurfaceIterations"), 256u)
            : 64u;
    params.hasBoundedMedium = sceneHasBoundedMedium();

    // The atmosphere, from the same place and with the same on/off test Metal
    // uses (MetalFrameUniforms.mm), so a scene with an `atmosphere` sidecar
    // block hazes identically on the two backends. Density is the switch: a
    // block that is present and zero is a scene that turned the haze off, and
    // paying a free-flight draw per segment for it is not free.
    {
        const auto& atmosphere = mScene->getAtmosphere();
        const bool on = atmosphere.has_value() && atmosphere->density > 0.0f;
        params.hasFog = on;
        params.fogSigmaT = on ? atmosphere->density : 0.0f;
        params.fogAnisotropy = on ? atmosphere->anisotropy : 0.0f;
        params.fogHeight = on ? atmosphere->height : 0.0f;
        params.fogAlbedo = on ? make_float3(atmosphere->color.x, atmosphere->color.y, atmosphere->color.z)
                              : make_float3(0.0f);
    }
    // Dropped once the numbers have been reported, so the steady state pays
    // neither the memset nor the three atomics' guard.
    params.iorStats = (mIorStatsBuffer && !mReportedIorStats)
                          ? optix::devicePtr<uint32_t>(mIorStatsBuffer->getPtr())
                          : nullptr;
    updateSharcParams(camera, params.image_width, params.image_height);

    memcpy(params.viewToWorld, glm::value_ptr(glm::transpose(glm::inverse(camera.matrices.view))),
           sizeof(params.viewToWorld));
    memcpy(params.clipToView, glm::value_ptr(glm::transpose(camera.matrices.invPerspective)), sizeof(params.clipToView));

    // Projection. The half-extents are adapted to the render aspect the same way
    // the perspective fov is (Camera::magForAspect), so a camera authored square
    // and rendered wide keeps its framing instead of stretching -- otherwise a
    // whole-frame comparison measures the reframe rather than the feature.
    params.projectionType = (uint32_t)camera.projection;
    {
        float halfWidth = camera.xmag;
        float halfHeight = camera.ymag;
        const float aspect =
            (params.image_height > 0) ? (float)params.image_width / (float)params.image_height : 1.0f;
        camera.magForAspect(aspect, halfWidth, halfHeight);
        params.orthoHalfWidth = halfWidth;
        params.orthoHalfHeight = halfHeight;
    }

    // World to clip, this frame's and the previous frame's. sutil::Matrix4x4 is
    // row major and glm is column major, hence the transpose -- the same one the
    // two matrices above take.
    const glm::mat4 worldToClip = camera.matrices.perspective * camera.matrices.view;
    const glm::mat4 prevWorldToClip = prevView.mCamMatrices.perspective * prevView.mCamMatrices.view;
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
    const float filmIso = settings.getAs<float>("render/post/tonemapper/filmIso");
    // The candela per meter square factor
    const float cm2_factor = settings.getAs<float>("render/post/tonemapper/cm2_factor");
    // The fractional aperture number; e.g., 11 means aperture "f/11." It adjusts the size of the opening of the "camera
    // iris" and is expressed as a ratio. The higher this value, the lower the exposure.
    const float fStop = settings.getAs<float>("render/post/tonemapper/fStop");
    // Controls the duration, in fractions of a second, that the "shutter" is open; e.g., the value 100 means that the
    // "shutter" is open for 1/100th of a second. The higher this value, the greater the exposure
    const float shutterSpeed = settings.getAs<float>("render/post/tonemapper/shutterSpeed");
    // Specifies the main color temperature of the light sources; the color that will be mapped to "white" on output,
    // e.g., an incoming color of this hue/saturation will be mapped to grayscale, but its intensity will remain
    // unchanged. This is similar to white balance controls on digital cameras.
    const float3 whitePoint{ 1.0f, 1.0f, 1.0f };
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
    // Clamped to zero rather than left negative: an AOV debug view forces a
    // launch even once mSubframeIndex has reached totalSpp (see below), which
    // pushes mSubframeIndex one step past totalSpp every frame it stays
    // selected. Left signed, that makes leftSpp negative and samplesThisLaunch
    // -- stored into a uint32_t -- wraps to ~4.29 billion, which becomes the
    // raygen's per-pixel sample-loop trip count and hangs the launch (and the
    // editor, which synchronizes on it every frame).
    // The subtraction is done in int64_t rather than left to wrap: totalSpp is
    // uint32_t and mSubframeIndex is size_t, so the natural spelling underflows
    // to ~1.8e19 and reaches int32_t only through an implementation-defined
    // conversion that happens to give back the negative number this wants.
    const int64_t remaining =
        static_cast<int64_t>(totalSpp) - static_cast<int64_t>(getSharedContext().mSubframeIndex);
    const int32_t leftSpp = static_cast<int32_t>(std::max<int64_t>(0, remaining));
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

    // Last, because it reads the finished parameters: everything the modules are
    // compiled against has to be settled before they are compiled against it.
    ensurePipelineSpecialization(params);

    if (mFrameStartEvent)
    {
        // The legacy default stream, which cudaStreamCreate's blocking streams
        // synchronise against -- so a pair recorded here brackets the launch on
        // mState.stream as well as the post kernels on this one, and the number
        // is the whole frame rather than the part of it that happens to share a
        // stream with the events.
        latchCudaError(cudaEventRecord(mFrameStartEvent, nullptr), "record the frame start event");
    }

    if (latchCudaError(cudaMemcpy(optix::devicePtr<void>(mState.mParamsBuffer->getPtr()), &params, sizeof(params),
                                  cudaMemcpyHostToDevice),
                       "upload the launch parameters"))
    {
        return;
    }
    markStageSubmitted(optix::GpuStage::ParamsUpload, nullptr);

    if (samplesThisLaunch != 0)
    {
        // Per launch, so the report reads "per sample" the way Metal's does.
        if (params.iorStats != nullptr)
        {
            cudaMemsetAsync(params.iorStats, 0, IOR_STAT_COUNT * sizeof(uint32_t), mState.stream);
        }

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
            // Update -> resolve -> query, in stream order. The launch above is
            // the update and the next one is the query; this is what makes the
            // deposits it just made readable, and ages out what the camera has
            // left behind. See resolveSharc().
            resolveSharc();
        }

        // Update subframe index for accumulation.
        //
        // A non-accumulating launch is finished the moment it returns -- there is
        // nothing further to converge -- so it reports the full budget rather than
        // zero. Reporting zero is what hung StrelkaCLI forever on the single-hit
        // debug views (`render.debug` 1 and 2 disable accumulation): the headless
        // loop is `while (mSubframeIndex < spp)`, so an index that resets every
        // frame never lets it exit. The interactive path is unaffected, because
        // samplesThisLaunch is computed from samplesPerLaunch and ignores the
        // remaining budget entirely when accumulation is off.
        //
        // Advanced whether or not the launch took, which is deliberate and is
        // what Metal does -- it counts at encode time and asks about validity
        // separately. The counter says how many samples have been *submitted*;
        // deviceError() says whether they are worth anything. A counter that
        // stalled on failure would leave every caller that loops until it
        // reaches its target -- StrelkaCLI, and the editor's audit modes --
        // spinning forever on a GPU that will never produce another sample, and
        // never reaching the check they already have for exactly this.
        getSharedContext().mSubframeIndex =
            enableAccumulation ? getSharedContext().mSubframeIndex + samplesThisLaunch : totalSpp;
        if (mDeviceError)
        {
            return;
        }
    }
    else
    {
        // Nothing was traced this frame -- the render has converged, or the scene
        // is empty. Re-present the accumulated image so the caller still gets a
        // picture rather than whatever was in its buffer.
        if (params.debug == 0)
        {
            const size_t imageSize =
                static_cast<size_t>(mState.params.image_width) * mState.params.image_height * sizeof(float4);
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
    //
    // That includes the single-hit views (eNormal, eMotionBlur), not only the
    // AOV ones: `< DEBUG_MODE_FIRST_AOV` let debug values 1 and 2 through, so
    // the temporal denoiser ran its reprojection on a buffer holding an
    // encoded normal instead of radiance, blending it against history from
    // whatever debug view (or none) the previous frame happened to be in.
    // Wrong-domain history at a disocclusion or an invalid motion vector reads
    // as a network-shaped blotch with no relation to the scene under it --
    // fixed to the screen rather than to any surface, because the denoiser
    // runs in screen space.
    const bool runDenoiser = plan.enabled() && params.debug == (uint32_t)DebugMode::eNone;
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
                copyDenoisedToImage(optix::devicePtr<const float4>(mDenoiser.output()), displayImage, outputWidth, outputHeight);
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
    // the right scene at the right size, which a black frame is not. Every
    // debug view falls in here now that runDenoiser is eNone-only above --
    // single-hit views (1, 2) need the same point-sample fallback the AOV
    // views (>= 3) always did, or a half-size render stays half the display.
    if (plan.upscale && (mDenoiserFallback || params.debug != (uint32_t)DebugMode::eNone))
    {
        upscalePointSample(params.image, width, height, displayImage, outputWidth, outputHeight);
    }
    mResetTemporalHistory = false;

    // Publication stays scene-linear. Presentation state travels with the exact
    // slot it describes so a display can apply exposure, the curve and gamma
    // once, after acquiring that slot. Debug views already contain display
    // values and bypass the complete presentation transform.
    mPendingPresentation.exposure[0] = exposureValue.x;
    mPendingPresentation.exposure[1] = exposureValue.y;
    mPendingPresentation.exposure[2] = exposureValue.z;
    mPendingPresentation.maxOutput =
        std::max(settings.getAs<float>("render/post/tonemapper/maxEDR"), 1.0f);
    mPendingPresentation.gamma = gamma;
    mPendingPresentation.tonemapper = static_cast<uint32_t>(tonemapperType);
    // One debug view is not in display values: DebugMode::eSharcRadiance shows
    // what the radiance cache holds, in scene units. Looking at it is only
    // useful beside the beauty render at the same exposure, and bypassing the
    // presentation transform would make every voxel above one the same white.
    // Everything else the debug menu offers -- normals, motion, the guides, the
    // cache's grid, occupancy and bounce heatmaps -- is already a colour.
    mPendingPresentation.content = DEBUG_MODE_IS_SCENE_LINEAR(params.debug) ?
                                       PresentationContent::SceneLinear :
                                       PresentationContent::DebugDisplayLinear;

    if (mFrameStopEvent && !latchCudaError(cudaEventRecord(mFrameStopEvent, nullptr), "record the frame stop event"))
    {
        mFrameTimingPending = true;
    }

    mDisplayImage = displayImage;
    mDisplayWidth = outputWidth;
    mDisplayHeight = outputHeight;
    mDisplayPresentation = mPendingPresentation;

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

    updateSceneEnvironment();

    createLightBuffer();
    // Built here rather than with the structures because it does not depend on
    // them: the records are packed per instance in the same order the top level
    // will use, so the table the empty top level never reaches is already the one
    // the real top level wants.
    createSbt();
    buildEmptyTopLevel();
}

void OptiXRender::updateSceneEnvironment()
{
    destroyEnvironmentTextures();

    const auto& envLight = mScene->getEnvLight();
    // A dome with no texture is still a light: a uniform sky of one colour, which
    // is what a V-Ray dome with use_dome_tex off is, and what a furnace test is.
    // Without this a sidecar of the form {"environment": {"color": [1,1,1]}} lit
    // nothing at all -- hasEnvMap needs a texture and the miss colour was a hard
    // zero -- so such a scene rendered black but for its lamps.
    //
    // Carried on the miss colour rather than as a sampled light, which is not a
    // shortcut: next-event estimation exists to importance sample a distribution
    // the BSDF cannot see, and a constant environment has none. For a Lambertian
    // surface the cosine-weighted BSDF sample IS the optimal strategy, so what is
    // left to converge is visibility alone. Same reasoning, and the same place to
    // put it, as MetalFrameUniforms.
    mMissColor = make_float3(0.0f);
    if (envLight.has_value() && envLight->texturePath.empty())
    {
        const glm::float3 c = envLight->color * envLight->intensity;
        mMissColor = make_float3(c.x, c.y, c.z);
    }
    mSbtDirty = true; // the miss record carries it

    if (envLight.has_value() && !envLight->texturePath.empty())
    {
        const std::string resourcePathStr = getSettings()->getAs<std::string>("resource/searchPath");
        const fs::path envTexPath = fs::path(resourcePathStr) / envLight->texturePath;
        loadEnvMap(envTexPath.string());
        mState.params.envMapIntensity = mEnvMapAutoScale * envLight->intensity;
        // Kept in double until the store, which is what the implicit conversion did
        // before the cast was made explicit -- M_PI is a double, so narrowing the
        // factor first would move the result by an ulp.
        mState.params.envMapRotation = static_cast<float>(envLight->rotationY * (M_PI / 180.0));
        mState.params.envMapColorTint = make_float3(envLight->color.x, envLight->color.y, envLight->color.z);
        // A backdrop the camera sees instead of the lighting environment. The
        // intensity is the backdrop's own; the auto-calibration scale above is
        // not applied to it, because it exists to reconcile the *lighting*
        // map's units with the analytic lights and the backdrop lights nothing.
        if (!envLight->backgroundTexturePath.empty())
        {
            loadEnvBackground((fs::path(resourcePathStr) / envLight->backgroundTexturePath).string());
            mState.params.envBackgroundIntensity = envLight->backgroundIntensity;
        }
    }
    else
    {
        mState.params.hasEnvMap = false;
        mState.params.hasEnvBackground = false;
    }
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
    mTlasBuffer = std::make_unique<OptixBuffer>(std::max<size_t>(iasBufferSizes.outputSizeInBytes, 1));
    if (!mTempAccelBuffer || mTempAccelBuffer->size() < iasBufferSizes.tempSizeInBytes)
    {
        mTempAccelBuffer = std::make_unique<OptixBuffer>(std::max<size_t>(iasBufferSizes.tempSizeInBytes, 1));
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
        beginOpacityMicromaps();
    }

    const auto started = std::chrono::steady_clock::now();
    auto overBudget = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() >=
               budgetMs;
    };

    while (mBlasMeshCursor < meshes.size())
    {
        mOptixMeshes.emplace_back(createMesh(meshes[mBlasMeshCursor], mBlasMeshCursor));
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
    endOpacityMicromaps();
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
        std::ranges::sort(report.gpu,
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
    add("Emissive mesh sampling", bufBytes(mEmissiveMeshBuffer) + bufBytes(mEmissiveTriangleBuffer) +
                                      bufBytes(mEmissiveInstanceTransformBuffer) +
                                      bufBytes(mPrevEmissiveInstanceTransformBuffer));

    {
        // cudaArrayGetInfo asks the array itself for its extent and format, so
        // this is the texture's own answer rather than a replay of the arguments
        // it was created with.
        auto arrayBytes = [](cudaArray_t array) -> size_t {
            cudaChannelFormatDesc desc{};
            cudaExtent extent{};
            unsigned int flags = 0;
            if (!array || cudaArrayGetInfo(&desc, &extent, &flags, array) != cudaSuccess)
            {
                return 0;
            }
            const size_t texels = std::max<size_t>(extent.width, 1) * std::max<size_t>(extent.height, 1) *
                                  std::max<size_t>(extent.depth, 1);
            return texels * ((desc.x + desc.y + desc.z + desc.w) / 8);
        };
        // A mipmapped array only answers per level, so its levels are walked
        // until the driver says there are no more.
        auto mipmappedBytes = [&arrayBytes](cudaMipmappedArray_t mipmapped) -> size_t {
            size_t bytes = 0;
            for (uint32_t level = 0; mipmapped; ++level)
            {
                cudaArray_t levelArray = nullptr;
                if (cudaGetMipmappedArrayLevel(&levelArray, mipmapped, level) != cudaSuccess)
                {
                    cudaGetLastError();
                    break;
                }
                bytes += arrayBytes(levelArray);
            }
            return bytes;
        };

        // Both sets: the general one holds the environment, the material one
        // holds everything the maps decoded into, and the report is about the
        // device rather than about which of them owns what.
        size_t bytes = 0;
        for (cudaArray_t array : mTextureArrays)
            bytes += arrayBytes(array);
        for (cudaArray_t array : mMaterialTextureArrays)
            bytes += arrayBytes(array);
        for (cudaMipmappedArray_t mipmapped : mTextureMipmappedArrays)
            bytes += mipmappedBytes(mipmapped);
        for (cudaMipmappedArray_t mipmapped : mMaterialTextureMipmappedArrays)
            bytes += mipmappedBytes(mipmapped);
        add("Textures", bytes);
    }
    add("Texture table", bufBytes(mTexturesDataBuffer));
    // The alias table replaced the 2D CDF and its raw-radiance staging buffer,
    // so it is what the environment now costs beyond its texture arrays.
    add("Environment", bufBytes(mEnvAliasBuffer));

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
    {
        // Reported apart from the BLAS it belongs to: a micromap is the one part
        // of the acceleration structure a setting turns on, so seeing what it
        // costs is what makes the setting a decision rather than a guess.
        size_t bytes = 0;
        for (const std::unique_ptr<Mesh>& mesh : mOptixMeshes)
        {
            bytes += mesh ? mesh->omm_bytes : 0;
        }
        add("Opacity micromaps", bytes);
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
    add("IES profiles", bufBytes(mIesBuffer));
    add("Skinning", bufBytes(mVertexSkinDataBuffer) + mSkinningPtrs.bytes);
    // The hash table and the per-pixel visit records. Both are zero unless the
    // cache is on, and neither was in this report before the second one existed
    // -- a per-pixel allocation that the memory report does not know about is
    // exactly the kind this report is for.
    add("Radiance cache", bufBytes(mSharcBuffer) + bufBytes(mSharcPathBuffer));

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
    mStageMarkBuffer = std::make_unique<OptixBuffer>(optix::kGpuStageCount);

    // Three words, allocated once. The counting on the device is guarded on this
    // pointer being non-null, so a failed allocation costs the report and
    // nothing else.
    mIorStatsBuffer = std::make_unique<OptixBuffer>(IOR_STAT_COUNT * sizeof(uint32_t));
}

void OptiXRender::reportIorStackStats()
{
    if (mReportedIorStats || !mIorStatsBuffer || mDeviceError)
    {
        return;
    }
    uint32_t stats[IOR_STAT_COUNT] = {};
    if (cudaMemcpy(stats, optix::devicePtr<void>(mIorStatsBuffer->getPtr()), sizeof(stats),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaGetLastError();
        return;
    }
    const uint32_t overflow = stats[IOR_STAT_OVERFLOW];
    const uint32_t unmatched = stats[IOR_STAT_UNMATCHED];
    const uint32_t escaped = stats[IOR_STAT_ESCAPED_INSIDE];
    if (overflow == 0 && unmatched == 0 && escaped == 0)
    {
        return;
    }
    // Once. These are a property of the asset, not of the frame, and a warning
    // per frame would bury everything else in an interactive session.
    mReportedIorStats = true;
    STRELKA_WARNING(
        "Nested dielectrics lost paths, per sample: {} push(es) onto a full stack of {}, "
        "{} pop(s) that matched nothing, {} path(s) that reached the environment still "
        "inside a medium. The first wants a deeper stack; the other two are a mesh with a "
        "hole in it, seen from each side -- and the third is the one no exit event can "
        "catch, because the ray left through the hole. Each of them carries the wrong "
        "medium, and therefore the wrong absorption, for the rest of its life.",
        // IOR_STACK_SIZE is an unnamed enum, which fmt 11 refuses to format
        // through an implicit conversion; the cast is what makes it an int.
        overflow, static_cast<int>(IOR_STACK_SIZE), unmatched, escaped);
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
    std::ranges::fill(mStageSubmitted, static_cast<uint8_t>(0));
    // On the legacy stream, so it is ordered before everything this frame
    // enqueues on either stream.
    cudaMemsetAsync(optix::devicePtr<void>(mStageMarkBuffer->getPtr()), 0, optix::kGpuStageCount, nullptr);
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
    cudaMemsetAsync(optix::devicePtr<uint8_t>(mStageMarkBuffer->getPtr()) + index, 1, 1, stream);
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
    if (cudaMemcpy(completed, optix::devicePtr<void>(mStageMarkBuffer->getPtr()), optix::kGpuStageCount,
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

    // The device is idle and the counters are settled, which is the one point in
    // the frame where reading them back costs nothing extra.
    reportIorStackStats();
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
            mFramePresentation[mWriteIndex] = mPendingPresentation;
            mFrameSerials[mWriteIndex] = mNextFrameSerial++;
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

Render::ReadyFrame OptiXRender::getReadyFrame()
{
    const int ready = mReadyIndex.load(std::memory_order_acquire);

    if (ready < 0 || ready > 1)
    {
        return {};
    }
    return { mAsyncOutputBuffers[ready], nullptr, mFrameSerials[ready], mFramePresentation[ready] };
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
    const size_t size = static_cast<size_t>(desc.height) * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);

    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePtr, size));

    return new OptixBuffer(devicePtr, desc.format, desc.width, desc.height);
}

namespace
{

template <typename T>
void createOrUpdateBuffer(std::unique_ptr<OptixBuffer>& buffer, const std::vector<T>& data)
{
    const size_t bufferSize = data.size() * sizeof(T);

    if (buffer == nullptr)
    {
        buffer = std::make_unique<OptixBuffer>(bufferSize);
    }
    if (buffer->size() != bufferSize)
    {
        buffer->realloc(bufferSize);
    }
    if (bufferSize > 0)
    {
        CUDA_CHECK(
            cudaMemcpy(optix::devicePtr<void>(buffer->getPtr()), data.data(), bufferSize, cudaMemcpyHostToDevice));
    }
}

} // namespace

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
    const std::vector<Scene::Light>& lights = mScene->getLights();
    std::vector<double> powers;
    powers.reserve(lights.size());
    for (const Scene::Light& light : lights)
    {
        powers.push_back(metal::analyticLightPower(light));
    }
    const metal::LightSelectionTable selection = metal::buildLightSelectionAlias(powers);
    mAnalyticLightPower = selection.totalPower;

    static_assert(sizeof(Scene::Light) == offsetof(UniformLight, selectionAliasThreshold));
    std::vector<UniformLight> gpuLights(lights.size());
    for (size_t i = 0; i < lights.size(); ++i)
    {
        std::memcpy(&gpuLights[i], &lights[i], sizeof(Scene::Light));
        gpuLights[i].color.w = selection.entries[i].pdf;
        gpuLights[i].selectionAliasThreshold = selection.entries[i].aliasThreshold;
        gpuLights[i].selectionAlias = selection.entries[i].alias;
    }
    createOrUpdateBuffer(mLightBuffer, gpuLights);
    mState.params.scene.lights = optix::devicePtr<UniformLight>(mLightBuffer->getPtr());
    mState.params.scene.numLights = static_cast<uint32_t>(gpuLights.size());
    updateEmitterSelectionProbabilities();
    createIesBuffer();
    createProjectorTextures();
    createSharcResponsiveLightBuffer();
}

void OptiXRender::updateEmitterSelectionProbabilities()
{
    Params& params = mState.params;
    glm::float3 boundsMin(0.0f);
    glm::float3 boundsMax(0.0f);
    const double sceneExtent = mScene != nullptr && mScene->worldBounds(boundsMin, boundsMax) ?
                                   std::max(static_cast<double>(glm::length(boundsMax - boundsMin)), 1e-4) :
                                   1e16;
    const double tintLuminance = 0.2126 * std::max(params.envMapColorTint.x, 0.0f) +
                                 0.7152 * std::max(params.envMapColorTint.y, 0.0f) +
                                 0.0722 * std::max(params.envMapColorTint.z, 0.0f);
    const double envPower =
        metal::environmentLightPower(mEnvMapPower, sceneExtent, params.envMapIntensity, tintLuminance);
    const metal::EmitterSelectionProbabilities selection = metal::emitterSelectionProbabilities(
        params.hasEnvMap, envPower, params.scene.numLights > 0u, mAnalyticLightPower,
        params.scene.numEmissiveMeshes > 0u, mEmissiveMeshPower);
    params.envSelectionPdf = selection.environment;
    params.scene.meshLightSelectionPdf = selection.meshGivenLocal;
}

void OptiXRender::createEmissiveMeshLights()
{
    const auto& instances = mScene->getInstances();
    const auto& meshes = mScene->getMeshes();
    const auto& materials = mScene->getMaterials();

    std::vector<oka::render::EmissiveMeshBuildInput> inputs;
    inputs.reserve(instances.size());
    std::vector<EmissiveInstanceTransform> currentTransforms(instances.size());
    std::vector<EmissiveInstanceTransform> previousTransforms(instances.size());

    for (size_t instanceId = 0; instanceId < instances.size(); ++instanceId)
    {
        const oka::Instance& instance = instances[instanceId];
        const glm::mat4& previous = mEnableMotionBlur && instanceId < mPrevInstances.size() ?
                                        mPrevInstances[instanceId].transform :
                                        instance.transform;
        std::memcpy(currentTransforms[instanceId].matrix,
                    glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12u);
        std::memcpy(previousTransforms[instanceId].matrix, glm::value_ptr(glm::float3x4(glm::rowMajor4(previous))),
                    sizeof(float) * 12u);

        if (instance.type != oka::Instance::Type::eMesh || instance.mMeshId >= meshes.size())
        {
            continue;
        }
        const uint32_t materialId = instance.mMaterialId == kInvalidIndex ? 0u : instance.mMaterialId;
        if (materialId >= materials.size())
        {
            continue;
        }

        const oka::Mesh& mesh = meshes[instance.mMeshId];
        oka::render::EmissiveMeshBuildInput input;
        input.instanceId = static_cast<uint32_t>(instanceId);
        // OptiX emits one GAS geometry per scene mesh, so primitive hits use
        // geometry zero within the instance.
        input.geometryId = 0u;
        input.vertexOffset = mesh.mVbOffset;
        input.indexOffset = mesh.mIndex;
        input.materialId = materialId;
        const bool potentiallyChanging = mesh.isSkeletal || (mEnableMotionBlur && previous != instance.transform);
        input.trianglePowers = oka::render::emissiveTrianglePowers(
            *mScene, mesh, materials[materialId], instance.transform, potentiallyChanging);

        // Traversal linearly interpolates the previous/current transform over
        // shutter time. Keep any triangle that has area at either endpoint in
        // the proposal support; the power is only a variance heuristic.
        if (mEnableMotionBlur)
        {
            const std::vector<double> previousPowers =
                oka::render::emissiveTrianglePowers(*mScene, mesh, materials[materialId], previous);
            for (size_t triangle = 0; triangle < input.trianglePowers.size(); ++triangle)
            {
                input.trianglePowers[triangle] = std::max(input.trianglePowers[triangle], previousPowers[triangle]);
            }
        }
        inputs.push_back(std::move(input));
    }

    const oka::render::EmissiveMeshDistribution distribution = oka::render::buildEmissiveMeshDistribution(inputs);
    mEmissiveMeshPower = distribution.totalPower;

    if (distribution.meshes.empty())
    {
        mEmissiveMeshBuffer.reset();
        mEmissiveTriangleBuffer.reset();
    }
    else
    {
        createOrUpdateBuffer(mEmissiveMeshBuffer, distribution.meshes);
        createOrUpdateBuffer(mEmissiveTriangleBuffer, distribution.triangles);
    }
    if (currentTransforms.empty())
    {
        mEmissiveInstanceTransformBuffer.reset();
        mPrevEmissiveInstanceTransformBuffer.reset();
    }
    else
    {
        createOrUpdateBuffer(mEmissiveInstanceTransformBuffer, currentTransforms);
        createOrUpdateBuffer(mPrevEmissiveInstanceTransformBuffer, previousTransforms);
    }

    SceneData& scene = mState.params.scene;
    scene.numEmissiveMeshes = static_cast<uint32_t>(distribution.meshes.size());
    scene.emissiveMeshes =
        mEmissiveMeshBuffer ? optix::devicePtr<const EmissiveMeshLight>(mEmissiveMeshBuffer->getPtr()) : nullptr;
    scene.emissiveTriangles = mEmissiveTriangleBuffer ?
                                  optix::devicePtr<const EmissiveTriangleLight>(mEmissiveTriangleBuffer->getPtr()) :
                                  nullptr;
    scene.emissiveInstanceTransforms =
        mEmissiveInstanceTransformBuffer ?
            optix::devicePtr<const EmissiveInstanceTransform>(mEmissiveInstanceTransformBuffer->getPtr()) :
            nullptr;
    scene.prevEmissiveInstanceTransforms =
        mPrevEmissiveInstanceTransformBuffer ?
            optix::devicePtr<const EmissiveInstanceTransform>(mPrevEmissiveInstanceTransformBuffer->getPtr()) :
            nullptr;
    updateEmitterSelectionProbabilities();
}

/// Upload the images projector lights throw, in the order the scene registered
/// them, and publish the table each light's points[0].z indexes.
///
/// Rebuilt with the light set rather than with the materials, and tracked in its
/// own resource list, because the two have different lifetimes: editing a
/// material reloads every material texture, and a projector's slide must not go
/// with them while a light in the buffer still names it.
///
/// The images go up as RGBA float rather than through the block-compressed
/// material path. A projector's texels are its emission, magnified across a wall
/// by a factor of ten or more, where the 4x4 block artefacts a BC7 encode leaves
/// behind are not a subtle quality difference -- and a scene has a handful of
/// slides, not the thousands of maps that made compression worth its cost there.
void OptiXRender::createProjectorTextures()
{
    destroyProjectorTextures();

    const std::vector<std::string>& images = mScene->getProjectorImages();
    if (images.empty())
    {
        mState.params.scene.projectorTextures = nullptr;
        return;
    }

    std::vector<cudaTextureObject_t> table(images.size(), 0);
    for (size_t i = 0; i < images.size(); ++i)
    {
        Rgba32fImage image = decodeRgba32f(images[i], "projector image");
        if (!image.valid())
        {
            // Left at zero: the shader then throws a plain white frame, so a
            // missing file is a visible white rectangle rather than a light that
            // silently stopped working.
            continue;
        }

        const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaArray_t array = nullptr;
        CUDA_CHECK(cudaMallocArray(&array, &channelDesc, image.width, image.height));
        CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, image.pixels, image.width * sizeof(float4),
                                       image.width * sizeof(float4), image.height, cudaMemcpyHostToDevice));
        image.release();

        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = array;

        cudaTextureDesc texDesc{};
        // Clamp on both axes: a slide has an edge, and wrapping would tile the
        // wall with copies of the frame the moment a bilinear tap reached a
        // texel past the border.
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t texObj = 0;
        CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

        mProjectorTextureArrays.push_back(array);
        mProjectorTextureObjects.push_back(texObj);
        table[i] = texObj;
        STRELKA_INFO("Loaded projector image: {} ({}x{})", images[i], image.width, image.height);
    }

    const size_t bytes = table.size() * sizeof(cudaTextureObject_t);
    mProjectorTextureBuffer = std::make_unique<OptixBuffer>(bytes);
    CUDA_CHECK(cudaMemcpy(optix::devicePtr<void>(mProjectorTextureBuffer->getPtr()), table.data(), bytes, cudaMemcpyHostToDevice));
    mState.params.scene.projectorTextures =
        optix::devicePtr<const cudaTextureObject_t>(mProjectorTextureBuffer->getPtr());
}

void OptiXRender::destroyProjectorTextures()
{
    for (const cudaTextureObject_t obj : mProjectorTextureObjects)
    {
        if (obj)
            cudaDestroyTextureObject(obj);
    }
    mProjectorTextureObjects.clear();

    for (cudaArray_t arr : mProjectorTextureArrays)
    {
        if (arr)
            cudaFreeArray(arr);
    }
    mProjectorTextureArrays.clear();

    mProjectorTextureBuffer.reset();
    mState.params.scene.projectorTextures = nullptr;
}

/// One bit per light, set where the scene marked the light responsive.
///
/// Built here rather than in updateSharcParams because it is a property of the
/// light set and changes only when that does -- and because a per-frame rebuild
/// would upload a bitset every frame for a feature most scenes never switch on.
///
/// The buffer is dropped entirely when no light is responsive, which is what
/// `params.sharcResponsive` reads, and that value is bound into the pipeline as
/// a constant: a scene without a responsive light compiles out the second probe
/// on every cached read and the second deposit on every path.
void OptiXRender::createSharcResponsiveLightBuffer()
{
    const auto& descs = mScene->getLightsDesc();
    const size_t words = (descs.size() + 31u) / 32u;
    std::vector<uint32_t> bits(words, 0u);
    uint32_t responsiveCount = 0;
    for (size_t i = 0; i < descs.size(); ++i)
    {
        if (descs[i].responsive)
        {
            bits[i >> 5u] |= 1u << (i & 31u);
            ++responsiveCount;
        }
    }

    mSharcResponsiveLightCount = responsiveCount;
    if (responsiveCount == 0)
    {
        mSharcResponsiveLightBuffer.reset();
        return;
    }
    createOrUpdateBuffer(mSharcResponsiveLightBuffer, bits);
    STRELKA_INFO("Radiance cache: {} of {} lights are responsive", responsiveCount, descs.size());
}

void OptiXRender::createIesBuffer()
{
    // Packed here rather than cached against the scene's profile list because
    // the tables are small -- a 181x1 luminaire is under a kilobyte -- and this
    // runs only when the light set changes.
    std::vector<oka::optix_ies::Profile> profiles;
    profiles.reserve(mScene->getIesProfiles().size());
    for (const Scene::IesProfile& p : mScene->getIesProfiles())
    {
        oka::optix_ies::Profile out;
        out.verticalAngles = p.verticalAngles;
        out.horizontalAngles = p.horizontalAngles;
        out.candela = p.candela;
        out.maxCandela = p.maxCandela;
        profiles.push_back(std::move(out));
    }

    // Always upload, even with no profiles: packProfiles() returns a zero-count
    // header, and a real pointer to one is what lets the shading path multiply
    // by sampleIesCandela() unconditionally instead of branching per light.
    createOrUpdateBuffer(mIesBuffer, oka::optix_ies::packProfiles(profiles));
    if (!profiles.empty())
    {
        STRELKA_INFO("Uploaded {} IES profile(s), {} bytes", profiles.size(), mIesBuffer->size());
    }
}

oka::optix_tex::DecodeSettings OptiXRender::textureDecodeSettings() const
{
    // Read through contains(): getAs() on a key nobody set logs an error and
    // asserts, and these four are optional -- a host that never sets them should
    // get the documented default, not a diagnostic per texture.
    const SettingsManager* s = getSettings();
    oka::optix_tex::DecodeSettings settings;
    if (s->contains("render/texture/maxDimension"))
        settings.maxDimension = s->getAs<uint32_t>("render/texture/maxDimension");
    if (s->contains("render/texture/downscale"))
        settings.downscale = std::max(1u, s->getAs<uint32_t>("render/texture/downscale"));
    if (s->contains("render/texture/compress"))
        settings.blockCompress = s->getAs<bool>("render/texture/compress");
    // Mip chains cost a third of the texture memory and buy nothing until a
    // level is selected: `tex2D` from a ray tracing program has no derivatives,
    // so it reads level 0. Off until ray-cone LOD asks for them, which is what
    // `render/texture/mips` is for.
    if (s->contains("render/texture/mips"))
        settings.wantMips = s->getAs<bool>("render/texture/mips");
    return settings;
}

Texture OptiXRender::loadTextureFromFile(const std::string& fileName, oka::optix_tex::Kind kind)
{
    namespace tex = oka::optix_tex;

    const tex::DecodeSettings settings = textureDecodeSettings();
    const std::string cacheDir = getSettings()->contains("render/texture/cachePath") ?
                                     getSettings()->getAs<std::string>("render/texture/cachePath") :
                                     std::string();
    const std::string cacheFile =
        cacheDir.empty() ?
            std::string() :
            (fs::path(cacheDir) /
             tex::cacheKey(fileName, kind, settings.maxDimension, settings.downscale, settings.blockCompress,
                           settings.wantMips))
                .string();

    tex::Payload payload = tex::readCachedPayload(cacheFile);
    if (!payload.valid)
    {
        payload = tex::decodeToPayload(fileName, kind, settings);
        if (!payload.valid)
        {
            STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
            return {};
        }
        writeCachedPayload(payload, cacheFile);
    }
    (payload.fromCache ? mTextureCacheHits : mTextureCacheMisses)++;

    tex::TextureResources res = tex::createTexture(payload);
    if (res.object == 0)
    {
        // A compressed upload that the driver refuses should cost this texture
        // its compression, not the render. Retry once, uncompressed, so that a
        // scene still shades rather than losing a map to a format decision.
        if (isCompressed(payload.plan.format) && settings.blockCompress)
        {
            STRELKA_WARNING("Block-compressed upload failed for {}, retrying uncompressed", fileName.c_str());
            tex::DecodeSettings fallback = settings;
            fallback.blockCompress = false;
            payload = tex::decodeToPayload(fileName, kind, fallback);
            res = tex::createTexture(payload);
        }
        if (res.object == 0)
        {
            STRELKA_ERROR("Unable to upload texture: {}", fileName.c_str());
            return {};
        }
    }

    // Tracked in the *material* set, not the general one: these are released and
    // reloaded whenever publishMaterialParams() runs again, which is what makes an
    // edited material reach the GPU. The general set holds the environment, which
    // a material reload must not free.
    if (res.array)
        mMaterialTextureArrays.push_back(res.array);
    if (res.mipmapped)
        mMaterialTextureMipmappedArrays.push_back(res.mipmapped);
    mMaterialTextureObjects.push_back(res.object);

    return { res.object,
             make_uint3((uint32_t)payload.plan.extent.width, (uint32_t)payload.plan.extent.height, 1),
             payload.plan.levels };
}

void OptiXRender::loadEnvMap(const std::string& texturePath)
{
    Rgba32fImage image = decodeRgba32f(texturePath, "env map");
    if (!image.valid())
    {
        return;
    }
    const int width = image.width;
    const int height = image.height;
    const float* pixelData = image.pixels;

    STRELKA_INFO("Loaded env map: {} ({}x{})", texturePath, width, height);
    metal::sanitizeEnvironmentPixels(image.pixels, width, height);

    // Create CUDA array and texture object for the env map (float4)
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
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

    // The alias table comes from the shared host builder in
    // render/host/ibl_alias_table.h. It is backend-neutral and
    // already has a unit test; reusing it rather than writing a second
    // implementation is what makes the two backends importance-sample the same
    // HDRI from the same distribution, which is the only way their EXRs can be
    // compared texel for texel.
    const auto aliasResult = metal::buildSolidAngleIblAliasTable(pixelData, width, height);
    static_assert(sizeof(EnvAliasEntry) == sizeof(metal::EnvAliasEntry),
                  "device EnvAliasEntry must match the host builder's entry");
    static_assert(alignof(EnvAliasEntry) == alignof(metal::EnvAliasEntry),
                  "device EnvAliasEntry must match the host builder's entry");

    image.release();

    const size_t aliasBytes = aliasResult.alias.size() * sizeof(metal::EnvAliasEntry);
    mEnvAliasBuffer = std::make_unique<OptixBuffer>(aliasBytes);
    CUDA_CHECK(cudaMemcpy(optix::devicePtr<void>(mEnvAliasBuffer->getPtr()), aliasResult.alias.data(), aliasBytes,
                          cudaMemcpyHostToDevice));
    // The stage the breadcrumbs know as EnvCdf is now the alias table: the CDF
    // kernel it was named for is gone, but it is still the same point in the
    // frame -- the environment's sampling distribution reaching the device -- so
    // a fault there is still reported against it.
    markStageSubmitted(optix::GpuStage::EnvCdf, nullptr);

    // Store env map params
    mState.params.envMapTexture = envTexObj;
    mState.params.envAliasTable = optix::devicePtr<const EnvAliasEntry>(mEnvAliasBuffer->getPtr());
    mState.params.envPdfScale = aliasResult.envPdfScale;
    mState.params.envMapWidth = width;
    mState.params.envMapHeight = height;
    mState.params.hasEnvMap = true;
    mEnvMapLoaded = true;
    mEnvMapPower = aliasResult.totalPower;

    // Environment auto-calibration is opt-in because its content-derived scale
    // overrides authored lighting units.
    const float avgWeightedLum = static_cast<float>(aliasResult.averageWeightedLuminance);
    const bool autoCalibrate = getSettings()->getAs<bool>("render/env/autoCalibrate");
    const float kCalibrationTarget = 1000.0f;
    mEnvMapAutoScale = (autoCalibrate && avgWeightedLum > 1e-6f) ? kCalibrationTarget / avgWeightedLum : 1.0f;

    STRELKA_INFO(
        "Env map alias table built: {} texels ({:.1f} MB), total power: {:.1f}, avgLum: {:.4f}, autoScale: {:.1f}",
        aliasResult.alias.size(), aliasBytes / (1024.0 * 1024.0), aliasResult.totalPower, avgWeightedLum,
        mEnvMapAutoScale);
}

void OptiXRender::loadEnvBackground(const std::string& texturePath)
{
    Rgba32fImage image = decodeRgba32f(texturePath, "env background");
    if (!image.valid())
    {
        return;
    }
    const int width = image.width;
    const int height = image.height;
    metal::sanitizeEnvironmentPixels(image.pixels, width, height);

    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t bgArray = nullptr;
    CUDA_CHECK(cudaMallocArray(&bgArray, &channelDesc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(
        bgArray, 0, 0, image.pixels, width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice));

    image.release();

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = bgArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t bgTexObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&bgTexObj, &resDesc, &texDesc, nullptr));

    mTextureArrays.push_back(bgArray);
    mTextureObjects.push_back(bgTexObj);

    mState.params.envBackgroundTexture = bgTexObj;
    mState.params.hasEnvBackground = true;

    STRELKA_INFO("Loaded env background: {} ({}x{})", texturePath, width, height);
}

void OptiXRender::destroyMaterialTextures()
{
    for (auto obj : mMaterialTextureObjects)
        if (obj) cudaDestroyTextureObject(obj);
    mMaterialTextureObjects.clear();

    for (auto arr : mMaterialTextureArrays)
        if (arr) cudaFreeArray(arr);
    mMaterialTextureArrays.clear();

    for (auto arr : mMaterialTextureMipmappedArrays)
        if (arr) cudaFreeMipmappedArray(arr);
    mMaterialTextureMipmappedArrays.clear();
}

void OptiXRender::destroyTextures()
{
    destroyMaterialTextures();
    destroyProjectorTextures();

    destroyEnvironmentTextures();
}

void OptiXRender::destroyEnvironmentTextures()
{
    for (auto obj : mTextureObjects)
        if (obj) cudaDestroyTextureObject(obj);
    mTextureObjects.clear();

    for (auto arr : mTextureArrays)
        if (arr) cudaFreeArray(arr);
    mTextureArrays.clear();

    for (auto arr : mTextureMipmappedArrays)
        if (arr) cudaFreeMipmappedArray(arr);
    mTextureMipmappedArrays.clear();

    mEnvAliasBuffer.reset();
    mState.params.envMapTexture = 0;
    mState.params.envBackgroundTexture = 0;
    mState.params.envAliasTable = nullptr;
    mState.params.envMapWidth = 0u;
    mState.params.envMapHeight = 0u;
    mState.params.envPdfScale = 0.0f;
    mState.params.hasEnvMap = false;
    mState.params.hasEnvBackground = false;
    mState.params.envBackgroundIntensity = 1.0f;
    mEnvMapLoaded = false;
    mEnvMapPower = 0.0;
    mEnvMapAutoScale = 1.0f;
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

    // Every texture the texture stage below loads is loaded again when this runs
    // again, so the previous set is released first. Without this a material edit
    // -- which now really does re-run this -- leaks a full copy of the scene's
    // textures each time.
    destroyMaterialTextures();

    mMaterials.resize(matDescs.size());
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        MaterialParams params = matDescs[i].params;
        if (params.material_type == MATERIAL_TYPE_OPENPBR)
        {
            const OpenPBRParams& openpbr = matDescs[i].openpbr;
            const bool hasEmissionMap =
                !matDescs[i].openpbrTexPaths[OPENPBR_TEX_EMISSION_COLOR].empty();
            params.emission = hasEmissionMap ? make_float3(1.0f) :
                                               make_float3(openpbr.emission_color.r, openpbr.emission_color.g,
                                                           openpbr.emission_color.b);
            params.emission_strength = openpbr.emission_luminance;
        }
        // Every slot empty until its stage runs. A -1 is what the shader reads as
        // "no map", so a material published now shades with its factors alone
        // rather than sampling a texture object that is still zero.
        params.base_color_tex = -1;
        params.metallic_roughness_tex = -1;
        params.normal_tex = -1;
        params.emission_tex = -1;
        params.occlusion_tex = -1;
        // Slot 5 is the transmission texture. Scene::MaterialDescription has no
        // field for it, so the glTF loader has nothing to hand over and the slot
        // stays empty; adding it is a loader change, reported as a hand-off.
        params.transmission_tex = -1;
        mMaterials[i].params = params;
    }

    mHostMaterialTextures.assign(matDescs.size() * MAX_MATERIAL_TEXTURES, 0);
    const size_t totalTexSize = mHostMaterialTextures.size() * sizeof(cudaTextureObject_t);
    mTexturesDataBuffer = std::make_unique<OptixBuffer>(totalTexSize);
    CUDA_CHECK(cudaMemset(optix::devicePtr<void>(mTexturesDataBuffer->getPtr()), 0, totalTexSize));

    std::vector<MaterialParams> allParams(matDescs.size());
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        allParams[i] = mMaterials[i].params;
    }
    const size_t paramsSize = allParams.size() * sizeof(MaterialParams);
    mMaterialParamsBuffer = std::make_unique<OptixBuffer>(paramsSize);
    CUDA_CHECK(
        cudaMemcpy(optix::devicePtr<void>(mMaterialParamsBuffer->getPtr()), allParams.data(), paramsSize,
                   cudaMemcpyHostToDevice));
    mMaterialCount = matDescs.size();

    mState.params.materials = optix::devicePtr<MaterialParams>(mMaterialParamsBuffer->getPtr());
    mState.params.materialTextures = optix::devicePtr<cudaTextureObject_t>(mTexturesDataBuffer->getPtr());

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

    // Cache: file path + how it is read -> texture object. The kind is part of
    // the key: the same file used as a base colour and as a roughness map is two
    // different textures, because only one of them is sRGB encoded.
    auto loadOrCacheTex = [&](const std::string& relPath, oka::optix_tex::Kind kind) -> cudaTextureObject_t {
        if (relPath.empty())
            return 0;
        const fs::path fullPath = resourcePath / relPath;
        const std::string key = fullPath.string() + "|" + std::to_string((int)kind);
        auto it = mTextureCache.find(key);
        if (it != mTextureCache.end())
            return it->second;
        if (!fs::exists(fullPath))
        {
            STRELKA_WARNING("Texture not found: {}", fullPath.string());
            mTextureCache[key] = 0;
            return 0;
        }
        const ::Texture tex = loadTextureFromFile(fullPath.string(), kind);
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

        // The kind per slot is the same split the Metal backend makes: base
        // colour and emission are sRGB encoded, everything else is linear data
        // that a transfer function would corrupt, and a normal map is a
        // direction rather than a colour at all.
        texSlots[0] = loadOrCacheTex(desc.baseColorTexPath, oka::optix_tex::Kind::Color);
        texSlots[1] = loadOrCacheTex(desc.metallicRoughnessTexPath, oka::optix_tex::Kind::NonColor);
        texSlots[2] = loadOrCacheTex(desc.normalTexPath, oka::optix_tex::Kind::Normal);
        const bool openpbrEmission = desc.params.material_type == MATERIAL_TYPE_OPENPBR &&
                                     !desc.openpbrTexPaths[OPENPBR_TEX_EMISSION_COLOR].empty();
        const std::string& emissionPath =
            openpbrEmission ? desc.openpbrTexPaths[OPENPBR_TEX_EMISSION_COLOR] : desc.emissionTexPath;
        oka::optix_tex::Kind emissionKind = oka::optix_tex::Kind::Color;
        if (openpbrEmission &&
            desc.openpbrTexColorSpace[OPENPBR_TEX_EMISSION_COLOR] == oka::TexColorSpace::Linear)
        {
            emissionKind = oka::optix_tex::Kind::NonColor;
        }
        texSlots[3] = loadOrCacheTex(emissionPath, emissionKind);
        texSlots[4] = loadOrCacheTex(desc.occlusionTexPath, oka::optix_tex::Kind::NonColor);
        // Slot 5 is the transmission texture. Scene::MaterialDescription has no
        // field for it, so the glTF loader has nothing to hand over and the slot
        // stays empty; adding it is a loader change, reported as a hand-off.
        texSlots[5] = 0;

        params.base_color_tex = texSlots[0] ? 0 : -1;
        params.metallic_roughness_tex = texSlots[1] ? 1 : -1;
        params.normal_tex = texSlots[2] ? 2 : -1;
        params.emission_tex = texSlots[3] ? 3 : -1;
        params.occlusion_tex = texSlots[4] ? 4 : -1;
        params.transmission_tex = -1;

        // Objects first, then the parameters that name them.
        CUDA_CHECK(cudaMemcpy(optix::devicePtr<cudaTextureObject_t>(mTexturesDataBuffer->getPtr()) +
                                  i * MAX_MATERIAL_TEXTURES,
                              texSlots, MAX_MATERIAL_TEXTURES * sizeof(cudaTextureObject_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(optix::devicePtr<MaterialParams>(mMaterialParamsBuffer->getPtr()) + i, &params,
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

    STRELKA_INFO("Loaded {} materials ({} unique textures, {} cache hits, {} decoded)", matDescs.size(),
                 mTextureCache.size(), mTextureCacheHits, mTextureCacheMisses);
    return true;
}
