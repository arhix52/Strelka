"""A semi-infinite scattering half-space under a uniform sky, for the third estimator.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/sss_halfspace.py
    tools/feature_tests/run_strelka.sh
    /Applications/Blender.app/Contents/Resources/<v>/python/bin/python3.x \
        tools/feature_tests/sss_halfspace_read.py

Why a third one.

`sss_slab.py` grades how much light crosses a body, against algebra. Nothing
grades the *angle* it leaves at, and that is where entry 16's residual lives:
`31_subsurface_absorbing` reads 1.174 at the centre of a sphere and 0.883 at its
rim, which is an angular redistribution, and every structural difference with
Cycles has already been eliminated. Two renderers disagreeing says nothing about
which is wrong.

The angular distribution of light leaving a semi-infinite, isotropically
scattering medium is one of the few things in transport with a closed form.
Chandrasekhar's H-function gives the bidirectional reflectance

    f_r(mu, mu0) = (omega / 4 pi) * H(mu) H(mu0) / (mu + mu0)

and under uniform illumination the emergent radiance is therefore

    L(mu) = (omega / 2) H(mu) * integral_0^1 mu0 H(mu0) / (mu + mu0) dmu0

with H itself the solution of H(mu) = 1 + (omega/2) mu H(mu) int_0^1 H(mu')/(mu+mu') dmu'.
`sss_halfspace_read.py` solves that by iteration and compares both renderers
against it.

A sphere is the geometry because it presents every view angle at once: at the
centre of the disc the surface faces the camera and mu = 1, at the rim mu -> 0.
The same disc the residual was measured on, now with a curve to hold it against.

The medium is optically enormous -- a mean free path of 0.01 against a radius of
0.48, so about 96 free paths across -- because the closed form is for a
half-space. Uniform sky and nothing else in the scene: no floor to bounce off, no
analytic light whose falloff would have to be matched.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NAME = "95_halfspace"
MFP = 0.01
RADIUS = 0.48
# Strongly scattering, so the emergent distribution is far from Lambertian and
# the shape has something to say. The base colour is what both renderers map to a
# single-scattering albedo, through the same van de Hulst fit; the reader takes
# the resulting alpha out of the glTF rather than assuming it.
BASE = 0.6


def build(tex):
    import bpy

    import build_features

    world = bpy.data.worlds.new("UniformSky")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = world

    obj = build_features.sphere("HalfSpace", (0.0, 0.0, 0.75), radius=RADIUS)
    obj.data.materials.append(build_features.new_material(
        "sss_halfspace", base_color=(BASE, BASE, BASE, 1.0), roughness=1.0, metallic=0.0,
        specular=0.0, subsurface=1.0, subsurface_radius=(MFP, MFP, MFP),
        subsurface_scale=1.0, subsurface_anisotropy=0.0))


if __name__ == "__main__":
    import build_features

    build_features.SSS_RADIUS = (MFP, MFP, MFP)
    build_features.SSS_SCALE = 1.0
    # Not `True`: that is what makes main() add the rect key light and write a
    # sidecar naming it. This row is lit by the sky alone, and the sidecar is
    # written below.
    build_features.SCENES.append((NAME, build, "env"))
    build_features.GLTF_PATCHERS[NAME] = build_features.patch_subsurface
    # Close in, so the disc is wide enough to resolve the angle across it.
    build_features.SCENE_CAMERAS[NAME] = ((0.0, -2.2, 0.75), (0.0, 0.0, 0.75), 45.0)

    sys.argv = [sys.argv[0], "--", "--only", NAME]
    build_features.main()

    root = os.path.join(HERE, "..", "..", "scenes", "feature_tests", NAME)
    with open(os.path.join(root, NAME + "_light.json"), "w") as f:
        json.dump({"lights": [], "environment": {"color": [1.0, 1.0, 1.0], "intensity": 1.0}},
                  f, indent=4)
    print("  lights -> sidecar (uniform environment, no analytic light)")
