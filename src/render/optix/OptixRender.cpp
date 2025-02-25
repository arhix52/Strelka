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

#include <vector_types.h>
#include <vector_functions.h>

#include <sutil/vec_math_adv.h>

#include "texture_support_cuda.h"

#include <filesystem>
#include <array>
#include <string>
#include <fstream>
#include <memory>

#include <log.h>

#include "cuda_checks.h"
#include "postprocessing/Tonemappers.h"

#include "Camera.h"

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
        STRELKA_ERROR("OptiX call {0} failed: {1}:{2}", call, file, line);
        assert(0);
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
        STRELKA_FATAL("OptiX call {0} failed: {1}:{2} : {3}", call, file, line, log);
        assert(0);
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

OptiXRender::OptiXRender()
{
}

OptiXRender::~OptiXRender()
{
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

OptiXRender::Curve* OptiXRender::createCurve(const oka::Curve& curve)
{
    Curve* rcurve = new Curve();
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS |
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const uint32_t pointsCount = mScene->getCurvesPoint().size(); // total points count in points buffer
    const int degree = 3;

    // each oka::Curves could contains many curves
    const uint32_t numCurves = curve.mVertexCountsCount;

    std::vector<int> segmentIndices;
    uint32_t offsetInsideCurveArray = 0;
    for (int curveIndex = 0; curveIndex < numCurves; ++curveIndex)
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
    CUdeviceptr d_segmentIndices = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segmentIndices), segmentIndicesSize));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_segmentIndices), segmentIndices.data(), segmentIndicesSize, cudaMemcpyHostToDevice));
    // Curve build input.
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
    curve_input.curveArray.vertexBuffers = &d_points;
    curve_input.curveArray.numVertices = pointsCount;
    curve_input.curveArray.vertexStrideInBytes = sizeof(glm::float3);
    curve_input.curveArray.widthBuffers = &d_widths;
    curve_input.curveArray.widthStrideInBytes = sizeof(float);
    curve_input.curveArray.normalBuffers = 0;
    curve_input.curveArray.normalStrideInBytes = 0;
    curve_input.curveArray.indexBuffer = d_segmentIndices;
    curve_input.curveArray.indexStrideInBytes = sizeof(int);
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curve_input.curveArray.primitiveIndexOffset = 0;

    // curve_input.curveArray.endcapFlags = OPTIX_CURVE_ENDCAP_ON;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &curve_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rcurve->d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    CUdeviceptr compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>((&compactedSizeBuffer)), sizeof(uint64_t)));

    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &curve_input,
                                1, // num build inputs
                                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, rcurve->d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &rcurve->gas_handle,
                                &property, // emitted property list
                                1)); // num emitted properties

    compactAccel(rcurve->d_gas_output_buffer, rcurve->gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_segmentIndices)));

    return rcurve;
}

OptiXRender::Mesh* OptiXRender::createMesh(const oka::Mesh& mesh)
{
    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;

    // Configure acceleration structure build options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Set up triangle input data
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

    // Allocate required buffers
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    // Set up compaction
    CUdeviceptr compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSizeBuffer), sizeof(uint64_t)));

    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    // Build acceleration structure
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &accel_options, &triangle_input,
                                1, // num build inputs
                                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                &property, // emitted property list
                                1)); // num emitted properties

    // Compact the acceleration structure
    compactAccel(d_gas_output_buffer, gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);

    // Free temporary buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));

    // Create and return mesh object
    Mesh* rmesh = new Mesh();
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
    mOptixMeshes.reserve(meshes.size());
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

void OptiXRender::createTopLevelAccelerationStructure()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();

    // Create OptixInstances from scene instances
    std::vector<OptixInstance> optixInstances;
    optixInstances.reserve(instances.size());

    for (const auto& instance : instances)
    {
        OptixInstance oi = {};

        // Set traversable handle and visibility mask based on instance type
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
            assert(0);
            break;
        }

        // Set transform and SBT offset
        memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(instance.transform))), sizeof(float) * 12);
        oi.sbtOffset = static_cast<unsigned int>(optixInstances.size() * RAY_TYPE_COUNT);

        optixInstances.push_back(oi);
    }

    // Allocate/reallocate device memory for instances if needed
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

    // Copy instances to device
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(mState.d_instances), optixInstances.data(), instancesSize, cudaMemcpyHostToDevice));

    // Setup IAS build input
    OptixBuildInput iasInput = {};
    iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    iasInput.instanceArray.instances = mState.d_instances;
    iasInput.instanceArray.numInstances = static_cast<int>(optixInstances.size());

    // Setup IAS build options
    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    iasOptions.motionOptions.numKeys = 1;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory requirements
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &iasOptions, &iasInput, 1, &iasBufferSizes));

    // Allocate buffers
    CUdeviceptr outputBuffer, tempBuffer, compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputBuffer), iasBufferSizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), iasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSizeBuffer), sizeof(uint64_t)));

    // Setup compaction property
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    // Build IAS
    OPTIX_CHECK(optixAccelBuild(mState.context, mState.stream, &iasOptions, &iasInput,
                                1, // num build inputs
                                tempBuffer, iasBufferSizes.tempSizeInBytes, outputBuffer,
                                iasBufferSizes.outputSizeInBytes, &mState.ias_handle, &property,
                                1 // num emitted properties
                                ));

    // Compact acceleration structure
    compactAccel(outputBuffer, mState.ias_handle, property.result, iasBufferSizes.outputSizeInBytes);

    // Cleanup temporary buffers
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(outputBuffer)));
}

void oka::OptiXRender::updateTopLevelAccelerationStructure()
{
    const std::vector<oka::Instance>& instances = mScene->getInstances();

    std::vector<OptixInstance> optixInstances;
    for (int i = 0; i < instances.size(); ++i)
    {
        OptixInstance oi = {};
        const oka::Instance& curr = instances[i];
        switch (curr.type)
        {
        case oka::Instance::Type::eMesh:
            oi.traversableHandle = mOptixMeshes[curr.mMeshId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_TRIANGLE;
            break;
        case oka::Instance::Type::eCurve:
            oi.traversableHandle = mOptixCurves[curr.mCurveId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_CURVE;
            break;
        case oka::Instance::Type::eLight:
            oi.traversableHandle = mOptixMeshes[curr.mMeshId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_LIGHT;
            break;
        default:
            STRELKA_ERROR("Unknown instance type");
            assert(0);
            break;
        }
        // fill common instance data
        memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(curr.transform))), sizeof(float) * 12);
        oi.sbtOffset = static_cast<unsigned int>(i * RAY_TYPE_COUNT);
        optixInstances.push_back(oi);
    }

    const size_t need_instances_size = sizeof(OptixInstance) * optixInstances.size();
    if (need_instances_size != mState.d_instances_size)
    {
        if (mState.d_instances != 0)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(mState.d_instances)));
        }
        CUDA_CHECK(cudaMalloc((void**)&mState.d_instances, need_instances_size));
        mState.d_instances_size = need_instances_size;
    }
    CUDA_CHECK(cudaMemcpy((void*)mState.d_instances, optixInstances.data(), need_instances_size, cudaMemcpyHostToDevice));

    OptixBuildInput ias_instance_input = {};
    ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_instance_input.instanceArray.instances = mState.d_instances;
    ias_instance_input.instanceArray.numInstances = static_cast<int>(optixInstances.size());
    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(
        optixAccelComputeMemoryUsage(mState.context, &ias_accel_options, &ias_instance_input, 1, &ias_buffer_sizes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_ias_and_compacted_size), ias_buffer_sizes.outputSizeInBytes));

    CUdeviceptr d_ias_temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes));

    CUdeviceptr compactedSizeBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>((&compactedSizeBuffer)), sizeof(uint64_t)));
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = compactedSizeBuffer;

    OPTIX_CHECK(optixAccelBuild(mState.context, 0, &ias_accel_options, &ias_instance_input, 1, d_ias_temp_buffer,
                                ias_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_ias_and_compacted_size,
                                ias_buffer_sizes.outputSizeInBytes, &mState.ias_handle, &property, 1));

    compactAccel(d_buffer_temp_output_ias_and_compacted_size, mState.ias_handle, property.result,
                 ias_buffer_sizes.outputSizeInBytes);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_buffer_temp_output_ias_and_compacted_size)));
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
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = 2;
    pipelineOptions.numAttributeValues = 2;
    pipelineOptions.exceptionFlags =
        mEnableValidation ?
            (OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW) :
            OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;

    // Load and create main module
    const fs::path optixPath = fs::current_path() / "optix/render_generated_OptixRender.cu.optixir";
    std::string optixSource;
    readSourceFile(optixSource, optixPath);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(mState.context, &moduleOptions, &pipelineOptions, optixSource.c_str(),
                                      optixSource.size(), log, &sizeof_log, &mState.ptx_module));

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

OptixProgramGroup OptiXRender::createRadianceClosestHitProgramGroup(PathTracerState& state,
                                                                    const char* module_code,
                                                                    size_t module_size)
{
    // Create material module
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixModule mat_module = nullptr;
    OPTIX_CHECK_LOG(optixModuleCreate(state.context, &state.module_compile_options, &state.pipeline_compile_options,
                                      module_code, module_size, log, &sizeof_log, &mat_module));

    // Configure hit group program
    OptixProgramGroupDesc hit_group_desc = {};
    hit_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_group_desc.hitgroup.moduleCH = mat_module;
    hit_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    hit_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_group_desc.hitgroup.entryFunctionNameIS = nullptr; // Built-in module auto-supplies this

    // Create program group
    OptixProgramGroup hit_group = nullptr;
    OptixProgramGroupOptions options = {};
    OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hit_group_desc, 1, &options, log, &sizeof_log, &hit_group));

    return hit_group;
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
    hit_prog_group_desc.hitgroup.moduleCH = mState.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
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

    for (auto& m : mMaterials)
    {
        program_groups.push_back(m.programGroup);
    }

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
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          2 // maxTraversableDepth
                                          ));
    mState.pipeline = pipeline;
}

void OptiXRender::createSbt()
{
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
    occlusion_miss.data = { 0.0f, 0.0f, 0.0f };
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
            const int material_idx = instance.mMaterialId == -1 ? 0 : instance.mMaterialId;
            const Material& material = getMaterial(material_idx);

            // Radiance hit group
            HitGroupSbtRecord& radiance_hit = hit_groups[i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE];

            if (instance.type == oka::Instance::Type::eLight)
            {
                radiance_hit.data.lightId = instance.mLightId;
                OPTIX_CHECK(optixSbtRecordPackHeader(mState.light_hit_group, &radiance_hit));
            }
            else
            {
                OPTIX_CHECK(optixSbtRecordPackHeader(material.programGroup, &radiance_hit));
            }

            // Set material data
            radiance_hit.data.argData = material.d_argData;
            radiance_hit.data.roData = material.d_roData;
            radiance_hit.data.resHandler = material.d_textureHandler;

            // Set mesh data if applicable
            if (instance.type == oka::Instance::Type::eMesh)
            {
                const oka::Mesh& mesh = meshes[instance.mMeshId];
                radiance_hit.data.indexCount = mesh.mCount;
                radiance_hit.data.indexOffset = mesh.mIndex;
                radiance_hit.data.vertexOffset = mesh.mVbOffset;
                radiance_hit.data.lightId = -1;
            }

            // Set transform matrices
            memcpy(radiance_hit.data.object_to_world, glm::value_ptr(glm::float4x4(glm::rowMajor4(instance.transform))),
                   sizeof(float4) * 4);

            glm::mat4 world_to_object = glm::inverse(instance.transform);
            memcpy(radiance_hit.data.world_to_object, glm::value_ptr(glm::float4x4(glm::rowMajor4(world_to_object))),
                   sizeof(float4) * 4);

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
    }
    mState.params.image_width = width;
    mState.params.image_height = height;
    if (needRealloc)
    {
        getSettings()->setAs<bool>("render/pt/isResized", true);
        if (mState.params.accum)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.accum));
        }
        if (mState.params.diffuse)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.diffuse));
        }
        if (mState.params.specular)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.specular));
        }
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

void OptiXRender::render(Buffer* output)
{
    if (getSharedContext().mFrameNumber == 0)
    {
        createOptixMaterials();
        createPipeline();
        createVertexBuffer();
        createIndexBuffer();
        // upload all curve data
        createPointsBuffer();
        createWidthsBuffer();
        createBottomLevelAccelerationStructures();
        createTopLevelAccelerationStructure();
        createSbt();
        createLightBuffer();
    }

    if (mScene->getDirtyState() == DirtyFlag::eLights)
    {
        createLightBuffer();
        createTopLevelAccelerationStructure();
        // createSbt();

        mScene->clearDirtyState();
    }

    SettingsManager& settings = *getSettings();
    bool settingsChanged = false;

    /*
    // node0 rotation
    if (mScene->getNodes().size() != 0) 
    {
        uint32_t rotationY = settings.getAs<uint32_t>("render/nodes/rotationY");
        if (rotationAngle != rotationY * 0.01f) 
        {
            settingsChanged = true;
            rotationAngle = rotationY * 0.01f;
            glm::quat rotationQuat = glm::angleAxis(rotationAngle, glm::vec3(0.0f, 1.0f, 0.0f));

            mScene->animateNode(0, oka::Scene::AnimationChannel::PathType::ROTATION, rotationQuat);
        }
    }*/

    // Animation changes
    std::vector<oka::Scene::Animation> &animations = mScene->getAnimations();
    bool blasChanged = false;
    for (int i = 0; i < animations.size(); ++i) 
    {
        const std::string scrollNameStr = "render/animation/anim" + std::to_string(i) + "/time";
        const char *scrollName = scrollNameStr.c_str();
        float currAnimTime = settings.getAs<float>(scrollName);

        const float EPSILON = 1e-6f; // 0.000001

        if (std::abs(animations[i].current - currAnimTime) > EPSILON) {
            settingsChanged = true;
            animations[i].current = currAnimTime;
            blasChanged = mScene->applyAnimation(i) ? true : blasChanged;
        }
    }

    // full rebuild or TLAS refit/reduild
    if (settingsChanged) {
        if (blasChanged) {
            mScene->applySkinning();
            createVertexBuffer();
            createBottomLevelAccelerationStructures();
            createTopLevelAccelerationStructure();
        }
        else {
            if(updateCount < 10) {
                updateTopLevelAccelerationStructure();
                updateCount++;
            }
            else {
                createTopLevelAccelerationStructure();
                updateCount = 0;
            }
        }
    }

    const uint32_t width = output->width();
    const uint32_t height = output->height();

    updatePathtracerParams(width, height);

    oka::Camera& camera = mScene->getCamera(0);
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

    static uint32_t rectLightSamplingMethodPrev = 0;
    const uint32_t rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");
    settingsChanged |= (rectLightSamplingMethodPrev != rectLightSamplingMethod);
    rectLightSamplingMethodPrev = rectLightSamplingMethod;

    static bool enableAccumulationPrev = 0;
    bool enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    settingsChanged |= (enableAccumulationPrev != enableAccumulation);
    enableAccumulationPrev = enableAccumulation;

    static uint32_t sspTotalPrev = 0;
    const uint32_t sspTotal = settings.getAs<uint32_t>("render/pt/sppTotal");
    settingsChanged |= (sspTotalPrev > sspTotal); // reset only if new spp less than already accumulated
    sspTotalPrev = sspTotal;

    const float gamma = settings.getAs<float>("render/post/gamma");
    const ToneMapperType tonemapperType = (ToneMapperType)settings.getAs<uint32_t>("render/pt/tonemapperType");

    if (settingsChanged)
    {
        getSharedContext().mSubframeIndex = 0;
    }

    Params& params = mState.params;
    params.scene.vb = (Vertex*)mVertexBuffer->getPtr();
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

    memcpy(params.viewToWorld, glm::value_ptr(glm::transpose(glm::inverse(camera.matrices.view))),
           sizeof(params.viewToWorld));
    memcpy(params.clipToView, glm::value_ptr(glm::transpose(camera.matrices.invPerspective)), sizeof(params.clipToView));
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

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(mState.mParamsBuffer->getPtr()), &params, sizeof(params), cudaMemcpyHostToDevice));

    if (samplesThisLaunch != 0)
    {
        // Launch OptiX path tracer
        OPTIX_CHECK(optixLaunch(mState.pipeline, mState.stream, mState.mParamsBuffer->getPtr(), sizeof(Params),
                                &mState.sbt, width, height,
                                /*depth=*/1));
        CUDA_SYNC_CHECK();

        // Update subframe index for accumulation
        getSharedContext().mSubframeIndex =
            enableAccumulation ? getSharedContext().mSubframeIndex + samplesThisLaunch : 0;
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
        //float maxEDR = settings.getAs<float>("render/post/tonemapper/maxEDR");
        //exposureValue *= maxEDR;
        tonemap(tonemapperType, exposureValue, gamma, params.image, width, height);
    }

    getSharedContext().mFrameNumber++;

    mPrevView = currView;
    mState.prevParams = mState.params;
}

void OptiXRender::init()
{
    // TODO: move USD_DIR to settings 
    const char* envUSDPath = std::getenv("USD_DIR");
    mEnableValidation = getSettings()->getAs<bool>("render/enableValidation");

    fs::path usdMdlLibPath;
    if (envUSDPath)
    {
        usdMdlLibPath = (fs::path(envUSDPath) / fs::path("libraries/mdl/")).make_preferred();
    }
    const fs::path cwdPath = fs::current_path();
    STRELKA_DEBUG("cwdPath: {}", cwdPath.string().c_str());
    const fs::path mtlxPath = (cwdPath / fs::path("data/materials/mtlx")).make_preferred();
    STRELKA_DEBUG("mtlxPath: {}", mtlxPath.string().c_str());
    const fs::path mdlPath = (cwdPath / fs::path("data/materials/mdl")).make_preferred();

    const std::string usdMdlLibPathStr = usdMdlLibPath.string();
    const std::string mtlxPathStr = mtlxPath.string().c_str();
    const std::string mdlPathStr = mdlPath.string().c_str();

    const char* paths[] = { usdMdlLibPathStr.c_str(), mtlxPathStr.c_str(), mdlPathStr.c_str() };
    bool res = mMaterialManager.addMdlSearchPath(paths, sizeof(paths) / sizeof(char*));

    if (!res)
    {
        STRELKA_FATAL("Wrong mdl paths configuration!");
        assert(0);
        return;
    }

    // default material
    {
        oka::Scene::MaterialDescription defaultMaterial{};
        defaultMaterial.file = "default.mdl";
        defaultMaterial.name = "default_material";
        defaultMaterial.type = oka::Scene::MaterialDescription::Type::eMdl;
        mScene->addMaterial(defaultMaterial);
    }

    createContext();
    // createAccelerationStructure();
    createModule();
    createProgramGroups();
    createPipeline();
    // createSbt();
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
void createOrUpdateRawBuffer(CUdeviceptr& buffer, const std::vector<T>& data)
{
    if (data.empty())
    {
        return;
    }

    // Free old buffer if it exists
    if (buffer)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
        buffer = 0;
    }

    const size_t bufferSize = data.size() * sizeof(T);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer), bufferSize));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(buffer), data.data(), bufferSize, cudaMemcpyHostToDevice));
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

    createOrUpdateRawBuffer(d_points, devicePoints);
}

void OptiXRender::createWidthsBuffer()
{
    createOrUpdateRawBuffer(d_widths, mScene->getCurvesWidths());
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

void OptiXRender::createVertexBuffer()
{
    createOrUpdateBuffer(mVertexBuffer, mScene->getVertices());
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
    return Texture(tex_obj, tex_obj_unfilt, make_uint3(texWidth, texHeight, 1));
}

bool OptiXRender::createOptixMaterials()
{
    // Create maps to cache resources
    std::unordered_map<std::string, MaterialManager::Module*> nameToModule;
    std::unordered_map<std::string, MaterialManager::MaterialInstance*> nameToInstance;
    std::unordered_map<std::string, MaterialManager::CompiledMaterial*> nameToCompiled;
    std::vector<MaterialManager::CompiledMaterial*> compiledMaterials;

    // Pre-allocate vectors to avoid reallocations
    const auto& matDescs = mScene->getMaterials();
    compiledMaterials.reserve(matDescs.size());

    // Process each material description
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        const auto& currMatDesc = matDescs[i];

        // Try to reuse already compiled material
        if (currMatDesc.type == oka::Scene::MaterialDescription::Type::eMdl)
        {
            if (auto it = nameToCompiled.find(currMatDesc.name); it != nameToCompiled.end())
            {
                compiledMaterials.emplace_back(it->second);
                continue;
            }

            // Create or get cached MDL module
            MaterialManager::Module* mdlModule = nullptr;
            auto moduleIt = nameToModule.find(currMatDesc.file);
            if (moduleIt != nameToModule.end())
            {
                mdlModule = moduleIt->second;
            }
            else
            {
                mdlModule = mMaterialManager.createModule(currMatDesc.file.c_str());
                if (!mdlModule)
                {
                    STRELKA_ERROR("Failed to load MDL file: {}, falling back to default.mdl", currMatDesc.file);
                    mdlModule = nameToModule["default.mdl"];
                    if (!mdlModule)
                    {
                        STRELKA_FATAL("Default material module not found!");
                        return false;
                    }
                }
                nameToModule[currMatDesc.file] = mdlModule;
            }

            // Create or get cached material instance
            MaterialManager::MaterialInstance* materialInst = nullptr;
            auto instIt = nameToInstance.find(currMatDesc.name);
            if (instIt != nameToInstance.end())
            {
                materialInst = instIt->second;
            }
            else
            {
                materialInst = mMaterialManager.createMaterialInstance(mdlModule, currMatDesc.name.c_str());
                if (!materialInst)
                {
                    STRELKA_ERROR("Failed to create material instance for: {}", currMatDesc.name);
                    continue;
                }
                nameToInstance[currMatDesc.name] = materialInst;
            }

            // Compile material
            auto materialComp = mMaterialManager.compileMaterial(materialInst);
            if (!materialComp)
            {
                STRELKA_ERROR("Failed to compile material: {}", currMatDesc.name);
                continue;
            }
            nameToCompiled[currMatDesc.name] = materialComp;
            compiledMaterials.push_back(materialComp);
        }
        else
        {
            // Handle MaterialX materials
            auto mdlModule = mMaterialManager.createMtlxModule(currMatDesc.code.c_str());
            if (!mdlModule)
            {
                STRELKA_ERROR("Failed to create MaterialX module");
                continue;
            }

            auto materialInst = mMaterialManager.createMaterialInstance(mdlModule, "");
            if (!materialInst)
            {
                STRELKA_ERROR("Failed to create MaterialX instance");
                mMaterialManager.destroyModule(mdlModule);
                continue;
            }

            auto materialComp = mMaterialManager.compileMaterial(materialInst);
            if (!materialComp)
            {
                STRELKA_ERROR("Failed to compile MaterialX material");
                mMaterialManager.destroyMaterialInstance(materialInst);
                mMaterialManager.destroyModule(mdlModule);
                continue;
            }

            compiledMaterials.push_back(materialComp);
        }
    }

    if (compiledMaterials.empty())
    {
        STRELKA_ERROR("No materials were successfully compiled");
        return false;
    }

    // Generate target code for all compiled materials
    auto targetCode = mMaterialManager.generateTargetCode(compiledMaterials.data(), compiledMaterials.size());
    if (!targetCode)
    {
        STRELKA_ERROR("Failed to generate target code");
        return false;
    }

    // Process textures and parameters
    std::vector<Texture> materialTextures;
    materialTextures.reserve(matDescs.size()); // Conservative estimate

    const fs::path resourcePath(getSettings()->getAs<std::string>("resource/searchPath"));

    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        for (const auto& param : matDescs[i].params)
        {
            bool res = false;
            if (param.type == MaterialManager::Param::Type::eTexture)
            {
                std::string texPath(param.value.begin(), param.value.end());
                fs::path fullTextureFilePath = resourcePath / texPath;
                ::Texture tex = loadTextureFromFile(fullTextureFilePath.string());
                materialTextures.push_back(tex);
                int texId = 0;
                int resId = mMaterialManager.registerResource(targetCode, texId);
                assert(resId > 0);
                MaterialManager::Param newParam;
                newParam.name = param.name;
                newParam.type = MaterialManager::Param::Type::eInt;
                newParam.value.resize(sizeof(resId));
                memcpy(newParam.value.data(), &resId, sizeof(resId));
                res = mMaterialManager.setParam(targetCode, i, compiledMaterials[i], newParam);
            }
            else
            {
                res = mMaterialManager.setParam(targetCode, i, compiledMaterials[i], param);
            }
            if (!res)
            {
                STRELKA_ERROR(
                    "Unable to set parameter: {0} for material: {1}", param.name.c_str(), matDescs[i].name.c_str());
                // assert(0);
            }
        }
        mMaterialManager.dumpParams(targetCode, i, compiledMaterials[i]);
    }

    const uint8_t* argData = mMaterialManager.getArgBufferData(targetCode);
    const size_t argDataSize = mMaterialManager.getArgBufferSize(targetCode);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materialArgData), argDataSize));
    CUDA_CHECK(cudaMemcpy((void*)d_materialArgData, argData, argDataSize, cudaMemcpyHostToDevice));

    const uint8_t* roData = mMaterialManager.getReadOnlyBlockData(targetCode);
    const size_t roDataSize = mMaterialManager.getReadOnlyBlockSize(targetCode);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_materialRoData), roDataSize));
    CUDA_CHECK(cudaMemcpy((void*)d_materialRoData, roData, roDataSize, cudaMemcpyHostToDevice));

    const size_t texturesBuffSize = sizeof(Texture) * materialTextures.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texturesData), texturesBuffSize));
    CUDA_CHECK(cudaMemcpy((void*)d_texturesData, materialTextures.data(), texturesBuffSize, cudaMemcpyHostToDevice));

    Texture_handler resourceHandler;
    resourceHandler.num_textures = materialTextures.size();
    resourceHandler.textures = (const Texture*)d_texturesData;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texturesHandler), sizeof(Texture_handler)));
    CUDA_CHECK(cudaMemcpy((void*)d_texturesHandler, &resourceHandler, sizeof(Texture_handler), cudaMemcpyHostToDevice));

    std::unordered_map<MaterialManager::CompiledMaterial*, OptixProgramGroup> compiledToOptixPG;
    for (int i = 0; i < compiledMaterials.size(); ++i)
    {
        if (compiledToOptixPG.find(compiledMaterials[i]) == compiledToOptixPG.end())
        {
            const char* codeData = mMaterialManager.getShaderCode(targetCode, i);
            assert(codeData);
            const size_t codeSize = strlen(codeData);
            assert(codeSize);
            OptixProgramGroup pg = createRadianceClosestHitProgramGroup(mState, codeData, codeSize);
            compiledToOptixPG[compiledMaterials[i]] = pg;
        }

        Material optixMaterial;
        optixMaterial.programGroup = compiledToOptixPG[compiledMaterials[i]];
        optixMaterial.d_argData = d_materialArgData + mMaterialManager.getArgBlockOffset(targetCode, i);
        optixMaterial.d_argDataSize = argDataSize;
        optixMaterial.d_roData = d_materialRoData + mMaterialManager.getReadOnlyOffset(targetCode, i);
        optixMaterial.d_roSize = roDataSize;
        optixMaterial.d_textureHandler = d_texturesHandler;

        mMaterials.push_back(optixMaterial);
    }

    // Clean up material resources
    for (auto& [name, module] : nameToModule)
    {
        mMaterialManager.destroyModule(module);
    }
    for (auto& [name, instance] : nameToInstance)
    {
        mMaterialManager.destroyMaterialInstance(instance);
    }
    for (auto& [name, compiled] : nameToCompiled)
    {
        mMaterialManager.destroyCompiledMaterial(compiled);
    }

    return true;
}

OptiXRender::Material& OptiXRender::getMaterial(int id)
{
    assert(id < mMaterials.size());
    return mMaterials[id];
}
