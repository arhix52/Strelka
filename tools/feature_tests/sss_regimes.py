"""The two subsurface rows `25_subsurface` cannot be: a different mean free path.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/sss_regimes.py
    tools/feature_tests/run_strelka.sh
    SSS_SCENE=30_subsurface_translucent \
        /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/feature_tests/sss_probe.py

Both rows are `s25_subsurface` with one constant changed, registered from here
rather than copied, so they cannot drift from the row they are variants of. They
write into the ordinary scene root, so `run_strelka.sh` and `compare.py` grade
them with no arguments; they are deliberately absent from `SCENES` so a plain
`build_features.py` run does not pay for two more Cycles references.

Why `25_subsurface` needs company at all.

**It passes with subsurface switched off.** Strip `STRELKA_materials_subsurface`
from its glTF and render the same base colours as plain Lambertian diffuse, and
that control grades rel 0.060 against the Cycles subsurface reference -- against
0.057 for the real thing. The row cannot tell the two apart, and neither can the
eye, so a subsurface regression that left the albedo mapping intact would not
move it.

That is by construction rather than by accident. At mean free paths of
0.05 / 0.025 / 0.015 against a sphere of radius 0.48 the medium is optically
thick, and the van de Hulst mapping the extension uses is *defined* to make a
thick medium reproduce a chosen diffuse albedo. So the row measures the mapping,
which is worth measuring, and says nothing about the transport underneath it.

`29_subsurface_skin` -- Blender's own skin preset, a Subsurface Radius of
(1, 0.2, 0.1) at a Subsurface Scale of 0.005. Against the same sphere that is one
hundred to one thousand mean free paths instead of ten to sixty, at a
single-scattering albedo that rounds to one in red. It is thicker still, so it
inherits the blindness above; what it guards is the walk's *length*. Russian
roulette does not fire in a medium that absorbs nothing, so the step ceiling is
the only thing that ends those walks and every walk it ends is discarded energy.
If that starts happening the ratio falls, which this row would show and no
shorter-walk row would.

`30_subsurface_translucent` -- mean free paths comparable to the body, so light
crosses it instead of turning round inside the first millimetre. This is the row
that actually grades the feature: the diffuse control that scores 0.060 on
`25_subsurface` scores ratio 0.910 here, against 1.045 for the walk. It fails
with subsurface off, which is the only property that makes a row a test of it.
It also fails against Cycles, which is entry 16 of docs/open-defects.md.

`31_subsurface_absorbing` -- the instrument that localised entry 16. Same body at
the same mean free path, but an albedo near zero, so a path either crosses
without a single collision or dies. It separates the two halves of the disagreement
cleanly: the lit side reads 1.03 and the shadowed side 3.30. Whatever fixes 16
has to bring the second to one without moving the first.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_features  # noqa: E402  (path set above)

# (name, subsurface_radius, subsurface_scale). Blender spells the skin preset as
# two numbers and the exported radius is the product; writing it that way is what
# proves `patch_subsurface` applies the scale. It did not, until this file needed
# it -- with the scale pinned at 1 everywhere else the omission was invisible, and
# here it would have been a factor of 200 between the two sides, read as a
# shading error.
# (name, subsurface_radius, subsurface_scale, base_colour). The colour is what
# separates the last row from the third: at an albedo near zero almost nothing
# scatters, so what crosses the body is Beer-Lambert and nothing else, and the
# entry and exit lobes are the only things left that can disagree.
ROWS = [
    ("29_subsurface_skin", (1.0, 0.2, 0.1), 0.005, None),
    ("30_subsurface_translucent", (0.40, 0.25, 0.15), 1.0, None),
    ("31_subsurface_absorbing", (0.8, 0.8, 0.8), 1.0, 0.05),
]

if __name__ == "__main__":
    def builder_for(base):
        if base is None:
            return build_features.s25_subsurface

        def build(tex):
            build_features.add_stage()
            for i, pos in enumerate(build_features.row_positions(5)):
                obj = build_features.sphere("A%d" % i, pos, radius=0.48)
                obj.data.materials.append(build_features.new_material(
                    "sss%d" % i, base_color=(base, base, base, 1.0), roughness=1.0,
                    metallic=0.0, specular=0.0, subsurface=1.0,
                    subsurface_radius=build_features.SSS_RADIUS,
                    subsurface_scale=build_features.SSS_SCALE, subsurface_anisotropy=0.0))

        return build

    for name, _, _, base in ROWS:
        build_features.SCENES.append((name, builder_for(base), True))
        build_features.GLTF_PATCHERS[name] = build_features.patch_subsurface

    for name, radius, scale, _ in ROWS:
        # The builder and the patcher read these at call time, so one main() per
        # row with the constants set immediately before is what keeps them apart.
        build_features.SSS_RADIUS = radius
        build_features.SSS_SCALE = scale
        sys.argv = [sys.argv[0], "--", "--only", name]
        build_features.main()
