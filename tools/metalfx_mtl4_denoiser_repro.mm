// Minimal reproducer: MTLFXTemporalDenoisedScaler cannot be created for Metal 4.
//
//   [descriptor newTemporalDenoisedScalerWithDevice:device compiler:compiler]
//
// aborts inside MPSGraph with
//
//   MPSGraphExecutable.mm:3467: failed assertion
//   'Incompatible shape for parameter at index 0'
//
// while the Metal 3 constructor, [descriptor newTemporalDenoisedScalerWithDevice:],
// succeeds on the byte-identical descriptor. Observed on macOS 26.5.2 (25F84) on
// both Apple M1 Pro and Apple M4 Pro, so it is not tied to one GPU generation.
//
// The failure is in the denoised scaler alone. Both neighbours are fine through
// the same compiler, which is what rules out the compiler and the environment:
//
//   MTL4FXSpatialScaler                 OK
//   MTL4FXTemporalScaler                OK
//   MTL4FXTemporalDenoisedScaler        assert
//
// Apple has confirmed this as a framework bug and asked for feedback; FB22575333
// tracks it, filed from developer.apple.com/forums/thread/819276, where the same
// call aborts on an A17 Pro with a different symptom again
// ("-[AGXG16XFamilyHeap baseObject]: unrecognized selector"). The recommended
// workaround is the Metal 3 constructor. Note that supportsMetal4FX: answers YES
// on every machine above, so it cannot be used to decide.
//
// Ruled out by bisection, each tried on its own and in combination: every texture
// format the descriptor accepts, input and output sizes including 1:1, autoExposure
// on and off, requiresSynchronousInitialization on and off, a plain MTL4Compiler
// and one carrying a pipeline data set serializer, and creating the Metal 3 scaler
// first from the same descriptor. All three descriptors report supportsDevice = YES.
//
// Build and run:
//   clang++ -std=c++20 -fobjc-arc -framework Metal -framework MetalFX \
//           -framework Foundation tools/metalfx_mtl4_denoiser_repro.mm -o /tmp/repro
//   /tmp/repro
//
// Expected once fixed: three OK lines and "survived".

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <cstdio>

int main()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    printf("device=%s compiler=%s\n", device.name.UTF8String, compiler ? "ok" : "nil");
    printf("supportsDevice: spatial=%d temporal=%d temporalDenoised=%d\n",
           (int)[MTLFXSpatialScalerDescriptor supportsDevice:device],
           (int)[MTLFXTemporalScalerDescriptor supportsDevice:device],
           (int)[MTLFXTemporalDenoisedScalerDescriptor supportsDevice:device]);

    @autoreleasepool
    {
        MTLFXSpatialScalerDescriptor* spatial = [MTLFXSpatialScalerDescriptor new];
        spatial.colorTextureFormat = MTLPixelFormatRGBA16Float;
        spatial.outputTextureFormat = MTLPixelFormatRGBA16Float;
        spatial.inputWidth = 512;
        spatial.inputHeight = 384;
        spatial.outputWidth = 1024;
        spatial.outputHeight = 768;
        printf("MTL4 spatial            -> %s\n",
               [spatial newSpatialScalerWithDevice:device compiler:compiler] ? "OK" : "nil");
        fflush(stdout);

        MTLFXTemporalScalerDescriptor* temporal = [MTLFXTemporalScalerDescriptor new];
        temporal.colorTextureFormat = MTLPixelFormatRGBA16Float;
        temporal.depthTextureFormat = MTLPixelFormatR32Float;
        temporal.motionTextureFormat = MTLPixelFormatRG16Float;
        temporal.outputTextureFormat = MTLPixelFormatRGBA16Float;
        temporal.inputWidth = 512;
        temporal.inputHeight = 384;
        temporal.outputWidth = 1024;
        temporal.outputHeight = 768;
        printf("MTL4 temporal           -> %s\n",
               [temporal newTemporalScalerWithDevice:device compiler:compiler] ? "OK" : "nil");
        fflush(stdout);

        MTLFXTemporalDenoisedScalerDescriptor* denoised = [MTLFXTemporalDenoisedScalerDescriptor new];
        denoised.colorTextureFormat = MTLPixelFormatRGBA16Float;
        denoised.depthTextureFormat = MTLPixelFormatR32Float;
        denoised.motionTextureFormat = MTLPixelFormatRG16Float;
        denoised.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
        denoised.specularAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
        denoised.normalTextureFormat = MTLPixelFormatRGBA16Float;
        denoised.roughnessTextureFormat = MTLPixelFormatR16Float;
        denoised.outputTextureFormat = MTLPixelFormatRGBA16Float;
        denoised.inputWidth = 512;
        denoised.inputHeight = 384;
        denoised.outputWidth = 1024;
        denoised.outputHeight = 768;

        // Works.
        printf("Metal 3 temporalDenoised -> %s\n",
               [denoised newTemporalDenoisedScalerWithDevice:device] ? "OK" : "nil");
        fflush(stdout);

        // Aborts.
        printf("MTL4 temporalDenoised   -> ");
        fflush(stdout);
        id<MTL4FXTemporalDenoisedScaler> broken =
            [denoised newTemporalDenoisedScalerWithDevice:device compiler:compiler];
        printf("%s\n", broken ? "OK" : "nil");
    }

    printf("survived\n");
    return 0;
}
