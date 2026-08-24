"""Beer-Lambert through a slab of known thickness, against Cycles and against algebra.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/sss_slab.py
    tools/feature_tests/run_strelka.sh
    /Applications/Blender.app/Contents/Resources/<v>/python/bin/python3.x \
        tools/feature_tests/sss_slab_read.py

Why a slab and not another sphere.

`31_subsurface_absorbing` says the disagreement with Cycles is entirely in paths
that cross the body without a single collision, and that our transmitted half is
3.3x too bright. What it cannot say is whether that is a defect or the mean chord
of a cosine-distributed entry into a sphere, because on a sphere the path length
is a distribution rather than a number: every candidate scale factor that fixed
the shadowed half wrecked the lit one, which is what a geometry confound looks
like.

A flat plate lit from above and viewed from below removes it. The medium is
almost purely absorbing -- base colour 0.05, so a path either crosses or dies --
and the top face is held at a fixed height while the thickness grows downward, so
every row receives identical irradiance. What reaches the camera is then
transmittance and nothing else.

That makes the row gradeable **without a reference**. The entry and exit lobes,
the light, the solid angle and the camera are common to every thickness, so they
cancel in a ratio:

    T(d1) / T(d2) = exp(-sigma_t * (d1 - d2))

is exact whatever those factors are. Strelka can therefore be graded against
algebra, which no ladder row has been able to do for this feature, and separately
against Cycles for the absolute level.

Deliberately no stage: a floor and a back wall would put bounced light on the
underside of the plate and that is the quantity being measured.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# build_features needs bpy, so it is imported where it is used rather than here:
# sss_slab_read.py takes MFP, THICKNESSES and slab_name() from this file and runs
# under a plain Python.

# One mean free path for every channel, so the three carry the same answer and a
# per-channel disagreement would stand out rather than being folded into a mean.
MFP = 0.2

# Optical depths of 0.25, 0.5, 1, 2 and 4. Wide enough that a wrong effective
# extinction cannot hide, and stopping at 4 because exp(-4) is already close
# enough to the noise floor of a 512-sample render.
THICKNESSES = [0.05, 0.10, 0.20, 0.40, 0.80]

# The top face, held fixed so irradiance does not change with thickness.
TOP_Z = 1.5
# Wide enough that the camera below sees no edge: at 45 degrees from 1.1 m the
# visible extent on the plate is 0.91 m.
WIDTH = 1.6


def slab_name(thickness):
    return "93_slab_%03d" % round(thickness * 100)


def make_builder(thickness):
    def build(tex):
        import bpy
        import build_features

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, TOP_Z - thickness * 0.5))
        obj = bpy.context.object
        obj.name = "Slab"
        obj.scale = (WIDTH, WIDTH, thickness)
        obj.data.materials.append(build_features.new_material(
            "sss_slab", base_color=(0.05, 0.05, 0.05, 1.0), roughness=1.0, metallic=0.0,
            specular=0.0, subsurface=1.0, subsurface_radius=(MFP, MFP, MFP),
            subsurface_scale=1.0, subsurface_anisotropy=0.0))

    return build


if __name__ == "__main__":
    import build_features

    build_features.SSS_RADIUS = (MFP, MFP, MFP)
    build_features.SSS_SCALE = 1.0

    for thickness in THICKNESSES:
        name = slab_name(thickness)
        build_features.SCENES.append((name, make_builder(thickness), True))
        build_features.GLTF_PATCHERS[name] = build_features.patch_subsurface
        # Under the plate, looking up at its centre. Radiance is distance
        # invariant, so the thickness moving the bottom face costs nothing.
        build_features.SCENE_CAMERAS[name] = ((0.0, -0.05, 0.4), (0.0, 0.0, TOP_Z), 45.0)

    for thickness in THICKNESSES:
        sys.argv = [sys.argv[0], "--", "--only", slab_name(thickness)]
        build_features.main()
