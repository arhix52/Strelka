
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
