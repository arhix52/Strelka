#!/usr/bin/env python3
"""
Render a production .blend in Cycles as a reference for Strelka's beauty pass.

    blender -b <file.blend> -P tools/feature_tests/beauty_ref.py -- --out FILE
            [--width 640] [--height 480] [--merge] [--cpu] [--seconds 60]

Unlike the feature-test harness, which builds its scenes and therefore knows
their exposure exactly, this renders a scene someone else authored. So it fixes
only the things that would otherwise make the two images incomparable for
reasons that have nothing to do with light transport:

  - Standard view transform, no look, no exposure, gamma 1. Filmic or AgX would
    put a tone curve in the reference that Strelka does not apply, and the whole
    comparison would then be measuring that curve.
  - Denoising off. A denoised reference hides exactly the noise-floor difference
    that says whether the two renderers agree or merely look similar.
  - Clamping off. Clamped indirect light silently darkens caustics and bright
    bounces, which is the first thing a comparison would blame on the BSDF.

Bounce depth is left alone here and matched from the Strelka side instead: the
scene's own value is the one the author intended, and reading it back out of the
render is how the TOML gets set.
"""

import bpy
import os
import sys


def merge_scenes():
    """Fold every other scene into the active one -- see export_scene.py."""
    target = bpy.context.scene
    for sc in bpy.data.scenes:
        if sc is target or sc.name.startswith("__"):
            continue
        for coll in list(sc.collection.children):
            if coll.name not in target.collection.children:
                target.collection.children.link(coll)
        for ob in list(sc.collection.objects):
            if ob.name not in target.collection.objects:
                target.collection.objects.link(ob)
    bpy.context.view_layer.update()


def pick_device(sc, want_cpu):
    """Metal GPU unless asked otherwise.

    Asked otherwise matters: on unified memory the GPU allocation comes out of
    the same pool as the scene, and a 50 M triangle forest is killed outright
    (SIGKILL, not a Cycles error) rather than falling back. CPU is slower and
    finishes.
    """
    if want_cpu:
        sc.cycles.device = "CPU"
        return
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        found = False
        for dev in prefs.devices:
            dev.use = True
            found = found or dev.type == "METAL"
        sc.cycles.device = "GPU" if found else "CPU"
    except (KeyError, AttributeError, TypeError):
        sc.cycles.device = "CPU"


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    def opt(name, default, cast=str):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    out = os.path.abspath(opt("--out", "/tmp/beauty.exr"))
    width = opt("--width", 640, int)
    height = opt("--height", 480, int)
    seconds = opt("--seconds", 60.0, float)
    if "--merge" in argv:
        merge_scenes()

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    pick_device(sc, "--cpu" in argv)

    # A wall-clock ceiling rather than a sample count: the point is a reference
    # that always arrives, and the noise it arrives with is visible in the
    # comparison anyway.
    sc.cycles.time_limit = seconds
    sc.cycles.samples = 4096
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.blur_glossy = 0.0
    sc.cycles.sample_clamp_indirect = 0.0
    sc.cycles.sample_clamp_direct = 0.0

    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.exr_codec = "ZIP"

    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)

    # Printed so the Strelka side can be matched to it rather than guessed at.
    print("BEAUTY %s %dx%d camera=%s device=%s bounces=%d/%d/%d/%d world=%s"
          % (out, width, height,
             sc.camera.name if sc.camera else None, sc.cycles.device,
             sc.cycles.max_bounces, sc.cycles.diffuse_bounces,
             sc.cycles.glossy_bounces, sc.cycles.transmission_bounces,
             sc.world.name if sc.world else None))


main()
