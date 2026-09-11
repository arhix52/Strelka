#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct OpenPbrSampleBenchParams
{
    uint32_t preset;
    uint32_t iterations;
    uint32_t seed;
    uint32_t padding;
};

static double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() & 1u ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name)
{
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function)
    {
        std::fprintf(stderr, "missing Metal function %s\n", name.UTF8String);
        std::exit(2);
    }
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = function;
    descriptor.label = name;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithDescriptor:descriptor
                                                                                 options:MTLPipelineOptionNone
                                                                              reflection:nil
                                                                                   error:&error];
    if (!pipeline)
    {
        std::fprintf(stderr, "pipeline %s: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
        std::exit(2);
    }
    return pipeline;
}

static double run(id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> output,
                  const OpenPbrSampleBenchParams& params,
                  NSUInteger threads,
                  NSString* label,
                  id<MTLBuffer> secondaryOutput = nil)
{
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = label;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = label;
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    if (secondaryOutput)
        [encoder setBuffer:secondaryOutput offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted)
    {
        std::fprintf(stderr, "%s failed: %s\n", label.UTF8String, commandBuffer.error.localizedDescription.UTF8String);
        std::exit(2);
    }
    return (commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1000.0;
}

int main(int argc, char** argv)
{
    @autoreleasepool
    {
        if (argc < 2)
        {
            std::fprintf(stderr,
                         "usage: %s <bench.metallib> [--threads N] [--iterations N] [--repeats N] "
                         "[--once] [--base-prepare] [--capture file.gputrace]\n",
                         argv[0]);
            return 2;
        }

        NSUInteger threads = 65536;
        uint32_t iterations = 32;
        uint32_t repeats = 7;
        bool once = false;
        bool basePrepare = false;
        const char* capturePath = nullptr;
        for (int i = 2; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
                threads = std::strtoull(argv[++i], nullptr, 10);
            else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
                iterations = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
                repeats = uint32_t(std::strtoul(argv[++i], nullptr, 10));
            else if (std::strcmp(argv[i], "--once") == 0)
                once = true;
            else if (std::strcmp(argv[i], "--base-prepare") == 0)
                basePrepare = true;
            else if (std::strcmp(argv[i], "--capture") == 0 && i + 1 < argc)
                capturePath = argv[++i];
            else
            {
                std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
                return 2;
            }
        }
        if (threads == 0 || iterations == 0 || repeats == 0)
        {
            std::fprintf(stderr, "threads, iterations and repeats must be positive\n");
            return 2;
        }
        if (capturePath)
        {
            setenv("MTL_CAPTURE_ENABLED", "1", 1);
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
        {
            std::fprintf(stderr, "no Metal device\n");
            return 2;
        }
        NSError* error = nil;
        NSURL* libraryUrl = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:libraryUrl error:&error];
        if (!library)
        {
            std::fprintf(stderr, "load metallib: %s\n", error.localizedDescription.UTF8String);
            return 2;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> output = [device newBufferWithLength:threads * sizeof(simd_float4)
                                                   options:MTLResourceStorageModeShared];

        static const char* names[] = { "diffuse",        "dielectric",       "metal",          "thick_sss",
                                       "thick_glass",    "thin_glass",       "coat",           "fuzz",
                                       "thin_film_coat", "bathroom_bubbles", "bathroom_water", "mixed" };
        static NSString* functionNames[] = { @"openpbrSampleOnceDiffuse",       @"openpbrSampleOnceDielectric",
                                             @"openpbrSampleOnceMetal",         @"openpbrSampleOnceThickSss",
                                             @"openpbrSampleOnceThickGlass",    @"openpbrSampleOnceThinGlass",
                                             @"openpbrSampleOnceCoat",          @"openpbrSampleOnceFuzz",
                                             @"openpbrSampleOnceThinFilmCoat",  @"openpbrSampleOnceBathroomBubbles",
                                             @"openpbrSampleOnceBathroomWater", @"openpbrSampleOnceMixed" };
        OpenPbrSampleBenchParams params{ 0u, iterations, 0x12345678u, 0u };

        if (basePrepare)
        {
            id<MTLComputePipelineState> reference = makePipeline(device, library, @"openpbrPrepareBaseReference");
            id<MTLComputePipelineState> direct = makePipeline(device, library, @"openpbrPrepareBaseDirect");
            id<MTLComputePipelineState> eval = makePipeline(device, library, @"openpbrPrepareBaseEval");
            id<MTLComputePipelineState> evalSample = makePipeline(device, library, @"openpbrPrepareBaseEvalSample");
            id<MTLComputePipelineState> neePayload = makePipeline(device, library, @"openpbrPrepareBaseNeePayload");
            id<MTLComputePipelineState> neePayloadSample =
                makePipeline(device, library, @"openpbrPrepareBaseNeePayloadSample");
            id<MTLBuffer> referenceOutput = [device newBufferWithLength:threads * sizeof(simd_float4)
                                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> directOutput = [device newBufferWithLength:threads * sizeof(simd_float4)
                                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> shadowOutput = [device newBufferWithLength:threads * 52u
                                                             options:MTLResourceStorageModeShared];

            std::printf("device: %s\n", device.name.UTF8String);
            std::printf("base prepare + one sample: threads=%llu repeats=%u\n",
                        static_cast<unsigned long long>(threads), repeats);
            std::printf("%-12s %10s %10s %10s %12s %12s %12s\n", "preset", "reference", "sample", "eval", "eval+sample",
                        "nee", "nee+sample");
            for (uint32_t preset = 0; preset < 3; ++preset)
            {
                params.preset = preset;
                (void)run(queue, reference, referenceOutput, params, threads, @"base prepare reference warmup");
                (void)run(queue, direct, directOutput, params, threads, @"base prepare direct warmup");
                (void)run(queue, eval, directOutput, params, threads, @"base prepare eval warmup");
                (void)run(queue, evalSample, directOutput, params, threads, @"base prepare eval + sample warmup");
                (void)run(
                    queue, neePayload, directOutput, params, threads, @"base prepare NEE payload warmup", shadowOutput);
                (void)run(queue, neePayloadSample, directOutput, params, threads,
                          @"base prepare NEE payload + sample warmup", shadowOutput);
                std::vector<double> referenceTimes;
                std::vector<double> directTimes;
                std::vector<double> evalTimes;
                std::vector<double> evalSampleTimes;
                std::vector<double> neePayloadTimes;
                std::vector<double> neePayloadSampleTimes;
                for (uint32_t repeat = 0; repeat < repeats; ++repeat)
                {
                    referenceTimes.push_back(
                        run(queue, reference, referenceOutput, params, threads, @"base prepare reference"));
                    directTimes.push_back(run(queue, direct, directOutput, params, threads, @"base prepare direct"));
                    evalTimes.push_back(run(queue, eval, directOutput, params, threads, @"base prepare eval"));
                    evalSampleTimes.push_back(
                        run(queue, evalSample, directOutput, params, threads, @"base prepare eval + sample"));
                    neePayloadTimes.push_back(run(
                        queue, neePayload, directOutput, params, threads, @"base prepare NEE payload", shadowOutput));
                    neePayloadSampleTimes.push_back(run(queue, neePayloadSample, directOutput, params, threads,
                                                        @"base prepare NEE payload + sample", shadowOutput));
                }
                const double referenceMs = median(referenceTimes);
                const double directMs = median(directTimes);
                const double evalMs = median(evalTimes);
                const double evalSampleMs = median(evalSampleTimes);
                const double neePayloadMs = median(neePayloadTimes);
                const double neePayloadSampleMs = median(neePayloadSampleTimes);
                (void)run(queue, direct, directOutput, params, threads, @"base prepare direct validation");
                const simd_float4* expected = static_cast<const simd_float4*>(referenceOutput.contents);
                const simd_float4* actual = static_cast<const simd_float4*>(directOutput.contents);
                float maxAbsError = 0.0f;
                double squaredError = 0.0;
                double absoluteError = 0.0;
                double expectedSum[4] = {};
                double actualSum[4] = {};
                double expectedSquaredSum[4] = {};
                double actualSquaredSum[4] = {};
                for (NSUInteger i = 0; i < threads; ++i)
                {
                    const simd_float4 delta = expected[i] - actual[i];
                    maxAbsError = std::max(maxAbsError, simd_reduce_max(simd_abs(delta)));
                    for (uint32_t component = 0; component < 4u; ++component)
                    {
                        const double d = delta[component];
                        squaredError += d * d;
                        absoluteError += std::abs(d);
                        const double e = expected[i][component];
                        const double a = actual[i][component];
                        expectedSum[component] += e;
                        actualSum[component] += a;
                        expectedSquaredSum[component] += e * e;
                        actualSquaredSum[component] += a * a;
                    }
                }
                const double values = double(threads) * 4.0;
                double maxMeanDelta = 0.0;
                double maxStdDelta = 0.0;
                for (uint32_t component = 0; component < 4u; ++component)
                {
                    const double expectedMean = expectedSum[component] / double(threads);
                    const double actualMean = actualSum[component] / double(threads);
                    const double expectedVariance =
                        std::max(expectedSquaredSum[component] / double(threads) - expectedMean * expectedMean, 0.0);
                    const double actualVariance =
                        std::max(actualSquaredSum[component] / double(threads) - actualMean * actualMean, 0.0);
                    maxMeanDelta = std::max(maxMeanDelta, std::abs(expectedMean - actualMean));
                    maxStdDelta =
                        std::max(maxStdDelta, std::abs(std::sqrt(expectedVariance) - std::sqrt(actualVariance)));
                }
                std::printf(
                    "%-12s %10.3f %10.3f %10.3f %12.3f %12.3f %12.3f  rmse=%.8g mean_abs=%.8g "
                    "max_abs=%.8g mean_delta=%.8g std_delta=%.8g\n",
                    names[preset], referenceMs, directMs, evalMs, evalSampleMs, neePayloadMs, neePayloadSampleMs,
                    std::sqrt(squaredError / values), absoluteError / values, maxAbsError, maxMeanDelta, maxStdDelta);
            }

            if (capturePath)
            {
                MTLCaptureDescriptor* descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = queue;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:capturePath]];
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error])
                {
                    std::fprintf(stderr, "start capture: %s\n", error.localizedDescription.UTF8String);
                    return 2;
                }
                for (uint32_t preset = 0; preset < 3; ++preset)
                {
                    params.preset = preset;
                    (void)run(queue, reference, referenceOutput, params, threads, @"base prepare reference");
                    (void)run(queue, direct, directOutput, params, threads, @"base prepare direct");
                    (void)run(queue, eval, directOutput, params, threads, @"base prepare eval");
                    (void)run(queue, evalSample, directOutput, params, threads, @"base prepare eval + sample");
                    (void)run(
                        queue, neePayload, directOutput, params, threads, @"base prepare NEE payload", shadowOutput);
                    (void)run(queue, neePayloadSample, directOutput, params, threads,
                              @"base prepare NEE payload + sample", shadowOutput);
                }
                [[MTLCaptureManager sharedCaptureManager] stopCapture];
                std::printf("capture: %s\n", capturePath);
            }
            return 0;
        }

        if (once)
        {
            std::vector<id<MTLComputePipelineState>> pipelines;
            pipelines.reserve(12);
            for (NSString* functionName : functionNames)
                pipelines.push_back(makePipeline(device, library, functionName));
            id<MTLComputePipelineState> onceRngPipeline = makePipeline(device, library, @"openpbrSampleOnceRng");
            params.iterations = 1u;

            for (uint32_t preset = 0; preset < 12; ++preset)
            {
                params.preset = preset;
                NSString* label = [NSString stringWithFormat:@"openpbr once warmup: %s", names[preset]];
                (void)run(queue, pipelines[preset], output, params, threads, label);
            }

            std::vector<double> rngTimes;
            for (uint32_t repeat = 0; repeat < repeats; ++repeat)
                rngTimes.push_back(run(queue, onceRngPipeline, output, params, threads, @"rng-only once"));
            const double rngMs = median(rngTimes);

            std::printf("device: %s\n", device.name.UTF8String);
            std::printf("specialized one-sample kernels: threads=%llu repeats=%u\n",
                        static_cast<unsigned long long>(threads), repeats);
            std::printf("%-18s %10s %12s %12s\n", "preset", "GPU ms", "ns/thread", "net ns/thread");
            std::printf("%-18s %10.3f %12.3f %12s\n", "rng-only", rngMs, rngMs * 1.0e6 / double(threads), "-");
            for (uint32_t preset = 0; preset < 12; ++preset)
            {
                params.preset = preset;
                std::vector<double> times;
                for (uint32_t repeat = 0; repeat < repeats; ++repeat)
                {
                    NSString* label = [NSString stringWithFormat:@"openpbr once: %s", names[preset]];
                    times.push_back(run(queue, pipelines[preset], output, params, threads, label));
                }
                const double ms = median(times);
                std::printf("%-18s %10.3f %12.3f %12.3f\n", names[preset], ms, ms * 1.0e6 / double(threads),
                            (ms - rngMs) * 1.0e6 / double(threads));
            }

            if (capturePath)
            {
                MTLCaptureDescriptor* descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = queue;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:capturePath]];
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error])
                {
                    std::fprintf(stderr, "start capture: %s\n", error.localizedDescription.UTF8String);
                    return 2;
                }
                for (uint32_t preset = 0; preset < 12; ++preset)
                {
                    params.preset = preset;
                    NSString* label = [NSString stringWithFormat:@"openpbr once: %s", names[preset]];
                    (void)run(queue, pipelines[preset], output, params, threads, label);
                }
                [[MTLCaptureManager sharedCaptureManager] stopCapture];
                std::printf("capture: %s\n", capturePath);
            }

            const simd_float4 checksum = static_cast<const simd_float4*>(output.contents)[threads / 2];
            std::printf("checksum: %.7g %.7g %.7g %.7g\n", checksum.x, checksum.y, checksum.z, checksum.w);
            return 0;
        }

        id<MTLComputePipelineState> samplePipeline = makePipeline(device, library, @"openpbrSampleBench");
        id<MTLComputePipelineState> rngPipeline = makePipeline(device, library, @"openpbrSampleBenchRng");
        for (uint32_t warmup = 0; warmup < 2; ++warmup)
            (void)run(queue, samplePipeline, output, params, threads, @"warmup");

        std::vector<double> rngTimes;
        for (uint32_t repeat = 0; repeat < repeats; ++repeat)
            rngTimes.push_back(run(queue, rngPipeline, output, params, threads, @"rng-only"));
        const double rngMs = median(rngTimes);
        const double samples = double(threads) * double(iterations);

        std::printf("device: %s\n", device.name.UTF8String);
        std::printf("threads=%llu iterations=%u samples/dispatch=%.0f repeats=%u\n",
                    static_cast<unsigned long long>(threads), iterations, samples, repeats);
        std::printf("%-18s %10s %12s %12s\n", "preset", "GPU ms", "ns/sample", "net ns/sample");
        std::printf("%-18s %10.3f %12.3f %12s\n", "rng-only", rngMs, rngMs * 1.0e6 / samples, "-");
        for (uint32_t preset = 0; preset < 12; ++preset)
        {
            params.preset = preset;
            std::vector<double> times;
            for (uint32_t repeat = 0; repeat < repeats; ++repeat)
            {
                NSString* label = [NSString stringWithFormat:@"openpbr sample: %s", names[preset]];
                times.push_back(run(queue, samplePipeline, output, params, threads, label));
            }
            const double ms = median(times);
            std::printf("%-18s %10.3f %12.3f %12.3f\n", names[preset], ms, ms * 1.0e6 / samples,
                        (ms - rngMs) * 1.0e6 / samples);
        }

        if (capturePath)
        {
            MTLCaptureDescriptor* descriptor = [MTLCaptureDescriptor new];
            descriptor.captureObject = queue;
            descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
            descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:capturePath]];
            if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error])
            {
                std::fprintf(stderr, "start capture: %s\n", error.localizedDescription.UTF8String);
                return 2;
            }
            for (uint32_t preset = 0; preset < 12; ++preset)
            {
                params.preset = preset;
                NSString* label = [NSString stringWithFormat:@"openpbr sample: %s", names[preset]];
                (void)run(queue, samplePipeline, output, params, threads, label);
            }
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            std::printf("capture: %s\n", capturePath);
        }

        const simd_float4 checksum = static_cast<const simd_float4*>(output.contents)[threads / 2];
        std::printf("checksum: %.7g %.7g %.7g %.7g\n", checksum.x, checksum.y, checksum.z, checksum.w);
    }
    return 0;
}
