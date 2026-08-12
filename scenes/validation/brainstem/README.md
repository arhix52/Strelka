# BrainStem — skinning regression

Articulated robot with skeletal animation. Used as the validation dataset's
skinning / animation guard: the CPU bounds test loads it, and the CLI smoke
renders it at mid-clip so a broken Metal 4 skinning or BLAS refit path fails
before anything subtler does.

## Provenance

`BrainStem.glb` is the Khronos [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
model of the same name (Smith Micro / Poser). Licence: Poser EULA — see
`LICENSE.md`.

## Pose

`BrainStem.toml` pins every animation to half its clip via `animation_time`.
That is the only way the headless path actually runs the skinning dispatch:
leaving time at the clip start keeps `animations[i].current` equal to the
target, so the renderer never marks the skeleton dirty.