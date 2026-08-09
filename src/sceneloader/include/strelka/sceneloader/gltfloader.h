#pragma once

#include <strelka/scene/scene.h>

#include <loadprogress.h>

#include <string>

namespace oka
{

class GltfLoader
{
private:
    LoadProgress* mProgress = nullptr;

public:
    explicit GltfLoader(){}

    /// Where to report progress and where to read cancellation from. Optional:
    /// nothing here reports if it is left unset, which is what the headless path
    /// wants. The object must outlive the load -- the editor owns one for the
    /// lifetime of the app for exactly that reason.
    void setProgress(LoadProgress* progress)
    {
        mProgress = progress;
    }

    /// Returns false on a malformed file and also on a load that was cancelled
    /// through the progress object, in which case `mScene` is left partial and
    /// the caller is expected to discard it.
    bool loadGltf(const std::string& modelPath, Scene& mScene);
};
} // namespace oka
