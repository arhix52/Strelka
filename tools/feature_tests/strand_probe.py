"""One isolated strand: the instrument that separates a hair lobe from hair transport.

28_hair is a groom, and a groom is optically dense -- most of what leaves it has
scattered off several strands. That makes it a poor place to ask whether the BSDF
is right, because two opposite errors there cancel in every aggregate metric: for
a while this row sat at rel 0.042 with ratio 1.001 while its shell was 4.5% bright
against a scalp 2.6% dark, and neither number said why.

This scene removes the transport. One strand, black background, no floor to bounce
light back, lit by the same key light and exported through the same sidecar path as
the ladder. The strand is horizontal, grown along +X, so the light above arrives
broadside: longitudinal angle near zero, which is where the R lobe is best
conditioned. A vertical strand would put the light along the fibre axis, the one
geometry where every hair model is degenerate.

Two things to read off it, both in strand_measure output:

  * The integrated cross-section against Cycles. This is the lobe, with no
    inter-strand transport left in it, and it should agree to about a percent.
  * The same number as a function of max_depth. A single convex fibre with nothing
    else in the scene has nowhere to send light that could come back, so it must
    stop changing after the second bounce. If it keeps climbing, transmitted rays
    are re-entering the strand they just left and buying a second whole-fibre
    event -- the Chiang lobe's T factor already covers the crossing. That is how
    the defect in entry 3 of docs/open-defects.md was finally cornered: 1.00 at
    depth 3 rising to 1.29 at depth 8, on one strand, in an empty room.

Usage:

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/strand_probe.py
    cd build/Release && for d in 1 2 3 4 8; do \
        ./StrelkaCLI -c /tmp/strand_probe/strand_d$d.toml; done
    blender -b --python-expr "exec(open('tools/feature_tests/strand_measure.py').read())"
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_features as bf  # noqa: E402
import bpy  # noqa: E402

NAME = "strand"
OUT = os.environ.get("PROBE_OUT", "/tmp/strand_probe")
SPP = int(os.environ.get("PROBE_SPP", "4096"))
DEPTHS = (1, 2, 3, 4, 8)


def main():
    os.makedirs(OUT, exist_ok=True)

    # Broadside view of a strand that runs along +X at z = 0.5.
    bf.SCENE_CAMERAS[NAME] = ((0.5, -1.2, 0.5), (0.5, 0.0, 0.5), 45.0)
    bf.STRELKA_SCENE_SPP[NAME] = SPP

    bf.reset_scene()
    scene = bpy.context.scene
    scene.cycles.samples = SPP
    # Round curves through the control points, matching what the sidecar hands
    # Strelka. Cycles defaults to camera-facing ribbons with two subdivisions.
    scene.cycles_curves.shape = "THICK"
    scene.cycles_curves.subdivisions = 0

    bf.add_camera(NAME)
    bf.add_key_light()

    scalp = bf.new_material("scalp", base_color=(0.0, 0.0, 0.0, 1.0),
                            roughness=1.0, metallic=0.0, specular=0.0)
    hair_mat = bf.new_hair_material("hair0")

    # A 2 mm emitter, rotated so its normal is +X and the hair grows horizontally.
    bpy.ops.mesh.primitive_plane_add(size=0.002, location=(0.0, 0.0, 0.5),
                                     rotation=(0.0, math.radians(90.0), 0.0))
    emitter = bpy.context.object
    emitter.name = "Emitter"
    emitter.data.materials.clear()
    emitter.data.materials.append(scalp)
    emitter.data.materials.append(hair_mat)

    emitter.modifiers.new("Hair", type="PARTICLE_SYSTEM")
    s = emitter.particle_systems[0].settings
    s.type = "HAIR"
    s.use_advanced_hair = True
    s.count = 1
    s.hair_length = 1.0
    s.hair_step = 5
    s.display_step = 3
    s.render_step = 3
    # Deliberately fat -- 20 px of cross-section -- so the *profile* can be
    # compared and not only the total. Untapered and open-tipped for the same
    # reason: a constant radius makes every row of the image the same geometry.
    s.root_radius = 0.04
    s.tip_radius = 0.04
    s.radius_scale = 1.0
    s.shape = 0.0
    s.use_close_tip = False
    s.child_type = "NONE"
    s.material = 2
    s.use_rotations = False

    gltf_path = os.path.join(OUT, NAME + ".gltf")
    bf.export_gltf(gltf_path, export_lights=False)
    with open(gltf_path) as f:
        doc = json.load(f)
    bf.patch_hair(doc)
    with open(gltf_path, "w") as f:
        json.dump(doc, f)
    bf.export_feature_hair(OUT, NAME)
    bf.write_light_json(os.path.join(OUT, NAME + "_light.json"))

    toml = os.path.join(OUT, NAME + ".toml")
    bf.write_toml(toml, NAME, gltf_rel=gltf_path,
                  out_rel=os.path.join(OUT, NAME + "_strelka.exr"))
    # A config per depth, so the convergence ladder is one loop over StrelkaCLI.
    with open(toml) as f:
        base = f.read()
    for d in DEPTHS:
        with open(os.path.join(OUT, "%s_d%d.toml" % (NAME, d)), "w") as f:
            f.write(base.replace("max_depth = 8", "max_depth = %d" % d)
                        .replace(NAME + "_strelka.exr", "%s_d%d.exr" % (NAME, d)))

    bf.render_cycles(os.path.join(OUT, NAME + "_cycles.exr"))
    print("probe -> %s (%d spp, depths %s)"
          % (OUT, SPP, ", ".join(str(d) for d in DEPTHS)))


if __name__ == "__main__":
    main()
