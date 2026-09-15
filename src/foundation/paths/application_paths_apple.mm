#include "application_paths.h"

#import <Foundation/Foundation.h>

namespace
{

std::filesystem::path searchPath(NSSearchPathDirectory directory)
{
    @autoreleasepool
    {
        const NSArray<NSString*>* paths = NSSearchPathForDirectoriesInDomains(directory, NSUserDomainMask, YES);
        if (paths.count == 0)
        {
            return std::filesystem::temp_directory_path();
        }
        return std::filesystem::path(paths.firstObject.fileSystemRepresentation);
    }
}

} // namespace

std::filesystem::path oka::applicationSupportDirectory()
{
    return searchPath(NSApplicationSupportDirectory) / "Strelka";
}

std::filesystem::path oka::applicationCacheDirectory()
{
    return searchPath(NSCachesDirectory) / "Strelka";
}

std::filesystem::path oka::applicationLogDirectory()
{
    return searchPath(NSLibraryDirectory) / "Logs" / "Strelka";
}
