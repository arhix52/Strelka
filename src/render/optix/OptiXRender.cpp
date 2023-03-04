#include "OptiXRender.h"

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

#include "texture_support_cuda.h"

#include <filesystem>
#include <array>
#include <string>
#include <sstream>
#include <fstream>

#include <log.h>

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

inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        STRELKA_FATAL("CUDA call ({0}) failed with error: {1} {2}:{3}", call, cudaGetErrorString(error), file, line);
        assert(0);
    }
}

inline void cudaSyncCheck(const char* file, unsigned int line)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        STRELKA_FATAL("CUDA error on synchronize with error {0} , {1}:{2}", cudaGetErrorString(error), file, line);
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

#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() cudaSyncCheck(__FILE__, __LINE__)

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

void configureCamera(::Camera& cam, const uint32_t width, const uint32_t height)
{
    cam.setEye({ 0.0f, 0.0f, 2.0f });
    cam.setLookat({ 0.0f, 0.0f, 0.0f });
    cam.setUp({ 0.0f, 1.0f, 3.0f });
    cam.setFovY(45.0f);
    cam.setAspectRatio((float)width / (float)height);
}

static bool readSourceFile(std::string& str, const std::string& filename)
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

OptiXRender::OptiXRender(/* args */)
{
}

OptiXRender::~OptiXRender()
{
}

void OptiXRender::createContext()
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    mState.stream = stream;

    OptixDeviceContext context;
    CUcontext cu_ctx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    if (mEnableValidation)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    }
    else
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    }
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    mState.context = context;
}

bool OptiXRender::compactAccel(CUdeviceptr& buffer,
                               OptixTraversableHandle& handle,
                               CUdeviceptr result,
                               size_t outputSizeInBytes)
{
    bool flag = false;

    size_t compacted_size;
    CUDA_CHECK(cudaMemcpy(&compacted_size, (void*)result, sizeof(size_t), cudaMemcpyDeviceToHost));

    CUdeviceptr compactedOutputBuffer;
    if (compacted_size < outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedOutputBuffer), compacted_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(mState.context, 0, handle, compactedOutputBuffer, compacted_size, &handle));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
        buffer = compactedOutputBuffer;

        flag = true;
    }

    return flag;
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

    OPTIX_CHECK(optixAccelBuild(mState.context, 0, // CUDA stream
                                &accel_options, &curve_input,
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
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        // std::vector<oka::Scene::Vertex>& vertices = mScene->getVertices();
        // std::vector<uint32_t>& indices = mScene->getIndices();

        const CUdeviceptr verticesDataStart = d_vb + mesh.mVbOffset * sizeof(oka::Scene::Vertex);
        CUdeviceptr indicesDataStart = d_ib + mesh.mIndex * sizeof(uint32_t);

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices = mesh.mVertexCount;
        triangle_input.triangleArray.vertexBuffers = &verticesDataStart;
        triangle_input.triangleArray.vertexStrideInBytes = sizeof(oka::Scene::Vertex);
        triangle_input.triangleArray.indexBuffer = indicesDataStart;
        triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
        triangle_input.triangleArray.numIndexTriplets = mesh.mCount / 3;
        // triangle_input.triangleArray.
        triangle_input.triangleArray.flags = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(mState.context, &accel_options, &triangle_input,
                                                 1, // Number of build inputs
                                                 &gas_buffer_sizes));
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

        CUdeviceptr compactedSizeBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>((&compactedSizeBuffer)), sizeof(uint64_t)));

        OptixAccelEmitDesc property = {};
        property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        property.result = compactedSizeBuffer;

        OPTIX_CHECK(optixAccelBuild(mState.context,
                                    0, // CUDA stream
                                    &accel_options, &triangle_input,
                                    1, // num build inputs
                                    d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes, d_gas_output_buffer,
                                    gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                    &property, // emitted property list
                                    1 // num emitted properties
                                    ));

        compactAccel(d_gas_output_buffer, gas_handle, property.result, gas_buffer_sizes.outputSizeInBytes);

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(compactedSizeBuffer)));
    }

    Mesh* rmesh = new Mesh();
    rmesh->d_gas_output_buffer = d_gas_output_buffer;
    rmesh->gas_handle = gas_handle;
    return rmesh;
}

void OptiXRender::createAccelerationStructure()
{
    const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
    const std::vector<oka::Curve>& curves = mScene->getCurves();
    const std::vector<oka::Instance>& instances = mScene->getInstances();
    if (meshes.empty() && curves.empty())
    {
        return;
    }

    // TODO: add proper clear and free resources
    mOptixMeshes.clear();
    for (int i = 0; i < meshes.size(); ++i)
    {
        Mesh* m = createMesh(meshes[i]);
        mOptixMeshes.push_back(m);
    }
    mOptixCurves.clear();
    for (int i = 0; i < curves.size(); ++i)
    {
        Curve* c = createCurve(curves[i]);
        mOptixCurves.push_back(c);
    }

    std::vector<OptixInstance> optixInstances;
    glm::mat3x4 identity = glm::identity<glm::mat3x4>();
    for (int i = 0; i < instances.size(); ++i)
    {
        OptixInstance oi = {};
        const oka::Instance& curr = instances[i];
        if (curr.type == oka::Instance::Type::eMesh)
        {
            oi.traversableHandle = mOptixMeshes[curr.mMeshId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_TRIANGLE;
        }
        else if (curr.type == oka::Instance::Type::eCurve)
        {
            oi.traversableHandle = mOptixCurves[curr.mCurveId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_CURVE;
        }
        else if (curr.type == oka::Instance::Type::eLight)
        {
            oi.traversableHandle = mOptixMeshes[curr.mMeshId]->gas_handle;
            oi.visibilityMask = GEOMETRY_MASK_LIGHT;
        }
        else
        {
            assert(0);
        }
        // fill common instance data
        memcpy(oi.transform, glm::value_ptr(glm::float3x4(glm::rowMajor4(curr.transform))), sizeof(float) * 12);
        oi.sbtOffset = static_cast<unsigned int>(i * RAY_TYPE_COUNT);
        optixInstances.push_back(oi);
    }

    size_t instances_size_in_bytes = sizeof(OptixInstance) * optixInstances.size();
    CUDA_CHECK(cudaMalloc((void**)&mState.d_instances, instances_size_in_bytes));
    CUDA_CHECK(
        cudaMemcpy((void*)mState.d_instances, optixInstances.data(), instances_size_in_bytes, cudaMemcpyHostToDevice));

    OptixBuildInput ias_instance_input = {};
    ias_instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    ias_instance_input.instanceArray.instances = mState.d_instances;
    ias_instance_input.instanceArray.numInstances = static_cast<int>(optixInstances.size());
    OptixAccelBuildOptions ias_accel_options = {};
    ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    ias_accel_options.motionOptions.numKeys = 1;
    ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(
        optixAccelComputeMemoryUsage(mState.context, &ias_accel_options, &ias_instance_input, 1, &ias_buffer_sizes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;

    auto roundUp = [](size_t x, size_t y) { return ((x + y - 1) / y) * y; };

    size_t compactedSizeOffset = roundUp(ias_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_ias_and_compacted_size), compactedSizeOffset + 8));

    CUdeviceptr d_ias_temp_buffer;
    // bool        needIASTempBuffer = ias_buffer_sizes.tempSizeInBytes > state.temp_buffer_size;
    // if( needIASTempBuffer )
    {
        CUDA_CHECK(cudaMalloc((void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes));
    }
    // else
    // {
    // d_ias_temp_buffer = state.d_temp_buffer;
    // }

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
}

void OptiXRender::createModule()
{
    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModuleCompileOptions module_compile_options = {};

    {
        if (mEnableValidation)
        {
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        }
        else
        {
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        }

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 2;
        pipeline_compile_options.numAttributeValues = 2;

        if (mEnableValidation) // Enables debug exceptions during optix launches. This may incur
            // significant performance cost and should only be done during
            // development.
            pipeline_compile_options.exceptionFlags =
                OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        else
        {
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        }

        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags =
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;

        size_t inputSize = 0;
        std::string optixSource;
        const std::string cwdPath = fs::current_path().string();
        const std::string precompiledOptixPath = cwdPath + "\\optix\\render_generated_OptixRender.cu.optixir";

        readSourceFile(optixSource, precompiledOptixPath.c_str());
        const char* input = optixSource.c_str();
        inputSize = optixSource.size();
        char log[2048]; // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(mState.context, &module_compile_options, &pipeline_compile_options,
                                                 input, inputSize, log, &sizeof_log, &module));
    }
    mState.ptx_module = module;
    mState.pipeline_compile_options = pipeline_compile_options;
    mState.module_compile_options = module_compile_options;

    // hair modules
    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    // builtinISOptions.curveEndcapFlags = OPTIX_CURVE_ENDCAP_ON;

    builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    OPTIX_CHECK(optixBuiltinISModuleGet(mState.context, &mState.module_compile_options, &mState.pipeline_compile_options,
                                        &builtinISOptions, &mState.m_catromCurveModule));
}

OptixProgramGroup OptiXRender::createRadianceClosestHitProgramGroup(PathTracerState& state,
                                                                    char const* module_code,
                                                                    size_t module_size)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixModule mat_module = nullptr;
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(state.context, &state.module_compile_options, &state.pipeline_compile_options,
                                             module_code, module_size, log, &sizeof_log, &mat_module));

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = mat_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

    hit_prog_group_desc.hitgroup.moduleIS = mState.m_catromCurveModule;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = 0; // automatically supplied for built-in module

    sizeof_log = sizeof(log);
    OptixProgramGroup ch_hit_group = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hit_prog_group_desc,
                                            /*numProgramGroups=*/1, &program_group_options, log, &sizeof_log,
                                            &ch_hit_group));

    return ch_hit_group;
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
    {
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

        if (mEnableValidation)
        {
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        }
        else
        {
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        }

        char log[2048]; // For error reporting from OptiX creation functions
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(mState.context, &mState.pipeline_compile_options, &pipeline_link_options,
                                            program_groups.data(), program_groups.size(), log, &sizeof_log, &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto& prog_group : program_groups)
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
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
    }
    mState.pipeline = pipeline;
}

void OptiXRender::createSbt()
{
    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.raygen_prog_group, &rg_sbt));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        const uint32_t miss_record_count = RAY_TYPE_COUNT;
        size_t miss_record_size = sizeof(MissSbtRecord) * miss_record_count;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
        std::vector<MissSbtRecord> missGroupDataCpu(miss_record_count);

        MissSbtRecord& ms_sbt = missGroupDataCpu[RAY_TYPE_RADIANCE];
        // ms_sbt.data.bg_color = { 0.5f, 0.5f, 0.5f };
        ms_sbt.data.bg_color = { 0.0f, 0.0f, 0.0f };
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_miss_group, &ms_sbt));

        MissSbtRecord& ms_sbt_occlusion = missGroupDataCpu[RAY_TYPE_OCCLUSION];
        ms_sbt_occlusion.data = { 0.0f, 0.0f, 0.0f };
        OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_miss_group, &ms_sbt_occlusion));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(miss_record), missGroupDataCpu.data(), miss_record_size, cudaMemcpyHostToDevice));

        uint32_t hitGroupRecordCount = RAY_TYPE_COUNT;
        CUdeviceptr hitgroup_record = 0;
        size_t hitgroup_record_size = hitGroupRecordCount * sizeof(HitGroupSbtRecord);
        const std::vector<oka::Instance>& instances = mScene->getInstances();
        std::vector<HitGroupSbtRecord> hitGroupDataCpu(hitGroupRecordCount); // default sbt record
        if (!instances.empty())
        {
            const std::vector<oka::Mesh>& meshes = mScene->getMeshes();
            hitGroupRecordCount = instances.size() * RAY_TYPE_COUNT;
            hitgroup_record_size = sizeof(HitGroupSbtRecord) * hitGroupRecordCount;
            hitGroupDataCpu.resize(hitGroupRecordCount);
            for (int i = 0; i < instances.size(); ++i)
            {
                const oka::Instance& instance = instances[i];
                int hg_index = i * RAY_TYPE_COUNT + RAY_TYPE_RADIANCE;
                HitGroupSbtRecord& hg_sbt = hitGroupDataCpu[hg_index];
                // replace unknown material on default
                const int materialIdx = instance.mMaterialId == -1 ? 0 : instance.mMaterialId;
                const Material& mat = getMaterial(materialIdx);
                if (instance.type == oka::Instance::Type::eLight)
                {
                    hg_sbt.data.lightId = instance.mLightId;
                    const OptixProgramGroup& lightPg = mState.light_hit_group;
                    OPTIX_CHECK(optixSbtRecordPackHeader(lightPg, &hg_sbt));
                }
                else
                {
                    const OptixProgramGroup& hitMaterial = mat.programGroup;
                    OPTIX_CHECK(optixSbtRecordPackHeader(hitMaterial, &hg_sbt));
                }
                // write all needed data for instances
                hg_sbt.data.argData = mat.d_argData;
                hg_sbt.data.roData = mat.d_roData;
                hg_sbt.data.resHandler = mat.d_textureHandler;
                if (instance.type == oka::Instance::Type::eMesh)
                {
                    const oka::Mesh& mesh = meshes[instance.mMeshId];
                    hg_sbt.data.indexCount = mesh.mCount;
                    hg_sbt.data.indexOffset = mesh.mIndex;
                    hg_sbt.data.vertexOffset = mesh.mVbOffset;
                    hg_sbt.data.lightId = -1;
                }

                memcpy(hg_sbt.data.object_to_world, glm::value_ptr(glm::float4x4(glm::rowMajor4(instance.transform))),
                       sizeof(float4) * 4);
                glm::mat4 world_to_object = glm::inverse(instance.transform);
                memcpy(hg_sbt.data.world_to_object, glm::value_ptr(glm::float4x4(glm::rowMajor4(world_to_object))),
                       sizeof(float4) * 4);

                // write data for visibility ray
                hg_index = i * RAY_TYPE_COUNT + RAY_TYPE_OCCLUSION;
                HitGroupSbtRecord& hg_sbt_occlusion = hitGroupDataCpu[hg_index];
                OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_hit_group, &hg_sbt_occlusion));
            }
        }
        else
        {
            // stub record
            HitGroupSbtRecord& hg_sbt = hitGroupDataCpu[RAY_TYPE_RADIANCE];
            OPTIX_CHECK(optixSbtRecordPackHeader(mState.radiance_default_hit_group, &hg_sbt));
            HitGroupSbtRecord& hg_sbt_occlusion = hitGroupDataCpu[RAY_TYPE_OCCLUSION];
            OPTIX_CHECK(optixSbtRecordPackHeader(mState.occlusion_hit_group, &hg_sbt_occlusion));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), hitGroupDataCpu.data(), hitgroup_record_size,
                              cudaMemcpyHostToDevice));

        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = RAY_TYPE_COUNT;
        sbt.hitgroupRecordBase = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = hitGroupRecordCount;
    }
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
        getSharedContext().mFrameNumber = 0;
    }
    mState.params.image_width = width;
    mState.params.image_height = height;
    if (needRealloc)
    {
        getSharedContext().mSettingsManager->setAs<bool>("render/pt/isResized", true);
        if (mState.params.accum)
        {
            CUDA_CHECK(cudaFree((void*)mState.params.accum));
        }
        if (mState.d_params)
        {
            CUDA_CHECK(cudaFree((void*)mState.d_params));
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.d_params), sizeof(Params)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mState.params.accum),
                              mState.params.image_width * mState.params.image_height * sizeof(float4)));
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
        createAccelerationStructure();
        createSbt();
        createLightBuffer();
    }

    const uint32_t width = output->width();
    const uint32_t height = output->height();

    updatePathtracerParams(width, height);

    oka::Camera& camera = mScene->getCamera(1);
    camera.updateAspectRatio(width / (float)height);
    camera.updateViewMatrix();

    View currView;

    currView.mCamMatrices = camera.matrices;

    if (glm::any(glm::notEqual(currView.mCamMatrices.perspective, mPrevView.mCamMatrices.perspective)) ||
        glm::any(glm::notEqual(currView.mCamMatrices.view, mPrevView.mCamMatrices.view)))
    {
        // need reset
        getSharedContext().mFrameNumber = 0;
    }

    ::Camera cam;
    configureCamera(cam, width, height);

    SettingsManager& settings = *getSharedContext().mSettingsManager;

    Params& params = mState.params;
    params.scene.vb = (Vertex*)d_vb;
    params.scene.ib = (uint32_t*)d_ib;
    params.scene.lights = (UniformLight*)d_lights;
    params.scene.numLights = mScene->getLights().size();

    params.image = (float4*)((OptixBuffer*)output)->getNativePtr();
    params.samples_per_launch = settings.getAs<uint32_t>("render/pt/spp");
    params.handle = mState.ias_handle;
    params.cam_eye = make_float3(cam.eye().x, cam.eye().y, cam.eye().z);
    params.max_depth = settings.getAs<uint32_t>("render/pt/depth");

    params.rectLightSamplingMethod = settings.getAs<uint32_t>("render/pt/rectLightSamplingMethod");

    params.viewToWorld = glm::inverse(camera.matrices.view);
    params.clipToView = camera.matrices.invPerspective;

    if (mState.prevParams.rectLightSamplingMethod != params.rectLightSamplingMethod)
    {
        // need reset
        getSharedContext().mFrameNumber = 0;
    }

    params.subframe_index = getSharedContext().mFrameNumber;

    glm::float3 cam_u, cam_v, cam_w;
    cam.UVWFrame(cam_u, cam_v, cam_w);

    params.cam_u = make_float3(cam_u.x, cam_u.y, cam_u.z);
    params.cam_v = make_float3(cam_v.x, cam_v.y, cam_v.z);
    params.cam_w = make_float3(cam_w.x, cam_w.y, cam_w.z);

    params.enableAccumulation = settings.getAs<bool>("render/pt/enableAcc");
    params.debug = settings.getAs<uint32_t>("render/pt/debug");

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mState.d_params), &params, sizeof(params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
        mState.pipeline, mState.stream, mState.d_params, sizeof(Params), &mState.sbt, width, height, /*depth=*/1));
    CUDA_SYNC_CHECK();

    output->unmap();

    getSharedContext().mFrameNumber++;

    mPrevView = currView;
    mState.prevParams = mState.params;
}

void OptiXRender::init()
{
    const char* envUSDPath = std::getenv("USD_DIR");
    mEnableValidation = getSharedContext().mSettingsManager->getAs<bool>("render/enableValidation");

    if (!envUSDPath)
    {
        STRELKA_FATAL("Please, set USD_DIR variable");
        assert(0);
    }
    const std::string usdMdlLibPath = std::string(envUSDPath) + "\\libraries\\mdl\\";
    const std::string cwdPath = fs::current_path().string();
    const std::string mtlxPath = cwdPath + "\\data\\materials\\mtlx";
    const std::string mdlPath = cwdPath + "\\data\\materials\\mdl";

    const char* paths[] = { usdMdlLibPath.c_str(), mtlxPath.c_str(), mdlPath.c_str() };
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
    createAccelerationStructure();
    createModule();
    createProgramGroups();
    createPipeline();
    createSbt();
}

Buffer* OptiXRender::createBuffer(const BufferDesc& desc)
{
    const size_t size = desc.height * desc.width * Buffer::getElementSize(desc.format);
    assert(size != 0);
    void* devicePtr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePtr), size));
    OptixBuffer* res = new OptixBuffer(devicePtr, desc.format, desc.width, desc.height);
    return res;
}

void OptiXRender::createPointsBuffer()
{
    const std::vector<glm::float3>& points = mScene->getCurvesPoint();

    std::vector<float3> data(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        data[i] = make_float3(points[i].x, points[i].y, points[i].z);
    }
    const size_t size = data.size() * sizeof(float3);

    if (d_points)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_points)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_points), data.data(), size, cudaMemcpyHostToDevice));
}

void OptiXRender::createWidthsBuffer()
{
    const std::vector<float>& data = mScene->getCurvesWidths();
    const size_t size = data.size() * sizeof(float);

    if (d_widths)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_widths)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_widths), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_widths), data.data(), size, cudaMemcpyHostToDevice));
}

void OptiXRender::createVertexBuffer()
{
    const std::vector<oka::Scene::Vertex>& vertices = mScene->getVertices();
    const size_t vbsize = vertices.size() * sizeof(oka::Scene::Vertex);

    if (d_vb)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vb)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vb), vbsize));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vb), vertices.data(), vbsize, cudaMemcpyHostToDevice));
}

void OptiXRender::createIndexBuffer()
{
    const std::vector<uint32_t>& indices = mScene->getIndices();
    const size_t ibsize = indices.size() * sizeof(uint32_t);

    if (d_ib)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ib)));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ib), ibsize));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_ib), indices.data(), ibsize, cudaMemcpyHostToDevice));
}

void OptiXRender::createLightBuffer()
{
    const std::vector<Scene::Light>& lightDescs = mScene->getLights();
    const size_t lightBufferSize = lightDescs.size() * sizeof(Scene::Light);
    if (d_lights)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_lights)));
    }
    if (lightBufferSize)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lights), lightBufferSize));
        CUDA_CHECK(
            cudaMemcpy(reinterpret_cast<void*>(d_lights), lightDescs.data(), lightBufferSize, cudaMemcpyHostToDevice));
    }
}

Texture OptiXRender::loadTextureFromFile(std::string& fileName)
{
    int texWidth, texHeight, texChannels;
    stbi_uc* data = stbi_load(fileName.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!data)
    {
        STRELKA_ERROR("Unable to load texture from file: {}", fileName.c_str());
        return Texture();
    }
    // convert to float4 format
    // TODO: add compression here:
    // std::vector<float> floatData(texWidth * texHeight * 4);
    // for (int i = 0; i < texHeight; ++i)
    // {
    //     for (int j = 0; j < texWidth; ++j)
    //     {
    //         const size_t linearPixelIndex = (i * texWidth + j) * 4;

    //         auto remapToFloat = [](const unsigned char v)
    //         {
    //             return float(v) / 255.0f;
    //         };

    //         floatData[linearPixelIndex + 0] = remapToFloat(data[linearPixelIndex + 0]);
    //         floatData[linearPixelIndex + 1] = remapToFloat(data[linearPixelIndex + 1]);
    //         floatData[linearPixelIndex + 2] = remapToFloat(data[linearPixelIndex + 2]);
    //         floatData[linearPixelIndex + 3] = remapToFloat(data[linearPixelIndex + 3]);
    //     }
    // }

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
    return Texture(tex_obj, tex_obj_unfilt, make_uint3(texWidth, texHeight, 1));
}

bool OptiXRender::createOptixMaterials()
{
    std::unordered_map<std::string, MaterialManager::Module*> mNameToModule;
    std::unordered_map<std::string, MaterialManager::MaterialInstance*> mNameToInstance;
    std::unordered_map<std::string, MaterialManager::CompiledMaterial*> mNameToCompiled;

    std::unordered_map<std::string, MaterialManager::TargetCode*> mNameToTargetCode;

    std::vector<MaterialManager::CompiledMaterial*> compiledMaterials;
    MaterialManager::TargetCode* targetCode;

    std::vector<Scene::MaterialDescription>& matDescs = mScene->getMaterials();
    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        oka::Scene::MaterialDescription& currMatDesc = matDescs[i];
        if (currMatDesc.type == oka::Scene::MaterialDescription::Type::eMdl)
        {
            // if (mNameToTargetCode.find(currMatDesc.name) != mNameToTargetCode.end())
            // {
            //     targetCodes.push_back(mNameToTargetCode[currMatDesc.name]);
            //     compiledMaterials.push_back(mNameToCompiled[currMatDesc.name]);
            //     continue;
            // }
            if (mNameToCompiled.find(currMatDesc.name) != mNameToCompiled.end())
            {
                compiledMaterials.push_back(mNameToCompiled[currMatDesc.name]);
                continue;
            }

            MaterialManager::Module* mdlModule = nullptr;
            if (mNameToModule.find(currMatDesc.file) != mNameToModule.end())
            {
                mdlModule = mNameToModule[currMatDesc.file];
            }
            else
            {
                mdlModule = mMaterialManager.createModule(currMatDesc.file.c_str());
                if (mdlModule == nullptr)
                {
                    STRELKA_FATAL("Unable to load MDL file: {}", currMatDesc.file.c_str());
                    assert(0);
                }
                mNameToModule[currMatDesc.file] = mdlModule;
            }
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = nullptr;
            if (mNameToInstance.find(currMatDesc.name) != mNameToInstance.end())
            {
                materialInst = mNameToInstance[currMatDesc.name];
            }
            else
            {
                materialInst = mMaterialManager.createMaterialInstance(mdlModule, currMatDesc.name.c_str());
                mNameToInstance[currMatDesc.name] = materialInst;
            }
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = nullptr;
            // if (mNameToCompiled.find(currMatDesc.name) != mNameToCompiled.end())
            // {
            //     materialComp = mNameToCompiled[currMatDesc.name];
            // }
            // else
            {
                materialComp = mMaterialManager.compileMaterial(materialInst);
                mNameToCompiled[currMatDesc.name] = materialComp;
            }
            assert(materialComp);
            compiledMaterials.push_back(materialComp);
            // MaterialManager::TargetCode* mdlTargetCode = mMaterialManager.generateTargetCode(&materialComp, 1);
            // assert(mdlTargetCode);
            // mNameToTargetCode[currMatDesc.name] = mdlTargetCode;
            // targetCodes.push_back(mdlTargetCode);
        }
        else
        {
            MaterialManager::Module* mdlModule = mMaterialManager.createMtlxModule(currMatDesc.code.c_str());
            assert(mdlModule);
            MaterialManager::MaterialInstance* materialInst = mMaterialManager.createMaterialInstance(mdlModule, "");
            assert(materialInst);
            MaterialManager::CompiledMaterial* materialComp = mMaterialManager.compileMaterial(materialInst);
            assert(materialComp);
            compiledMaterials.push_back(materialComp);
            // MaterialManager::TargetCode* mdlTargetCode = mMaterialManager.generateTargetCode(&materialComp, 1);
            // assert(mdlTargetCode);
            // mNameToTargetCode[currMatDesc.name] = mdlTargetCode;
            // targetCodes.push_back(mdlTargetCode);
        }
    }

    targetCode = mMaterialManager.generateTargetCode(compiledMaterials.data(), compiledMaterials.size());

    std::vector<Texture> materialTextures;

    std::string resourcePath = getSharedContext().mSettingsManager->getAs<std::string>("resource/searchPath");

    for (uint32_t i = 0; i < matDescs.size(); ++i)
    {
        mMaterialManager.dumpParams(targetCode, compiledMaterials[i]);
        for (const auto& param : matDescs[i].params)
        {
            bool res = false;
            if (param.type == MaterialManager::Param::Type::eTexture)
            {
                std::string texPath(param.value.size(), 0);
                memcpy(texPath.data(), param.value.data(), param.value.size());
                // int texId = getTexManager()->loadTextureMdl(texPath);
                ::Texture tex = loadTextureFromFile(resourcePath + "/" + texPath);
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
    return true;
}

OptiXRender::Material& OptiXRender::getMaterial(int id)
{
    assert(id < mMaterials.size());
    return mMaterials[id];
}
