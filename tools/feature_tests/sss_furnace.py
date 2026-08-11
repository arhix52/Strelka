"""White-furnace check for the subsurface random walk.

    python3 tools/feature_tests/sss_furnace.py            # build the scenes
    build/Release/StrelkaCLI --config /tmp/furnace_<r>.toml

A closed body whose medium has a single-scattering albedo of 1 absorbs nothing,
so under a uniform environment of radiance 1 it has to render as exactly 1
whatever its density. Anything below that is energy the walk lost, and the
deficit as a function of mean free path says whether the loss is per step.

Derived from 25_subsurface: same sphere and camera, with the stage removed and
the rect light replaced by the uniform sky.
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "scenes", "feature_tests", "25_subsurface")
OUT = os.path.join(HERE, "..", "..", "scenes", "feature_tests", "_sss_furnace")

RADII = [1.0, 0.5, 0.25, 0.12, 0.06]

# Van de Hulst says a semi-infinite isotropic medium of this single-scattering
# albedo has exactly this diffuse albedo, so under a sky of radiance 1 the sphere
# has to read the diffuse value back. Same inversion build_features.py applies.
ALBEDOS = [0.18, 0.35, 0.55, 0.75, 0.85]


def vdh(diffuse):
    def forward(alpha):
        s = (max(1.0 - alpha, 0.0)) ** 0.5
        return (1.0 - s) * (1.0 - 0.139 * s) / (1.0 + 1.17 * s)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if forward(mid) < diffuse:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build(radius, tag, alpha=1.0, reference=1.0):
    doc = json.load(open(os.path.join(SRC, "25_subsurface.gltf")))
    # Camera and one sphere: the stage would absorb what the sphere lets through
    # and a furnace has to be the object alone.
    keep = {i for i, n in enumerate(doc["nodes"]) if n.get("name") in ("Camera", "SSS0")}
    doc["scenes"][0]["nodes"] = sorted(keep)
    for mat in doc["materials"]:
        if not mat.get("name", "").startswith("sss"):
            continue
        mat["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        mat["extensions"]["STRELKA_materials_subsurface"] = {
            "subsurfaceFactor": 1.0,
            "scatterColor": [alpha] * 3,
            "scatterRadius": [radius] * 3,
            "anisotropy": 0.0,
            "scatterReference": [reference] * 3,
        }
    name = f"furnace_{tag}"
    json.dump(doc, open(os.path.join(OUT, name + ".gltf"), "w"))
    # The buffer URI is the one the export wrote; keep the name it asks for.
    shutil.copy(os.path.join(SRC, "25_subsurface.bin"), os.path.join(OUT, "25_subsurface.bin"))

    # No analytic lights: a textureless environment is carried on the miss colour,
    # which is the uniform sky this test needs.
    json.dump({"environment": {"color": [1.0, 1.0, 1.0], "intensity": 1.0}},
              open(os.path.join(OUT, name + "_light.json"), "w"))

    toml = f"""[scene]
path = "{os.path.abspath(os.path.join(OUT, name + '.gltf'))}"

[output]
path = "{os.path.abspath(os.path.join(OUT, name + '.exr'))}"
width = 256
height = 256

[render]
integrator = "pt"
spp = 1024
spp_per_launch = 1
max_depth = 64
sampler = "sobol"

[camera]
index = 0
position = [-1.850000, 1.150000, 3.000000]
target = [-1.850000, 0.750000, 0.000000]
fov = 45.0

[tonemap]
type = "none"
gamma = 0.0
"""
    path = f"/tmp/{name}.toml"
    open(path, "w").write(toml)
    return path


os.makedirs(OUT, exist_ok=True)
for r in RADII:
    print(build(r, str(r).replace(".", "p")))
# Optically thick enough that the sphere answers as a half-space, which is the
# configuration the Van de Hulst fit describes.
for a in ALBEDOS:
    print(build(0.02, "albedo" + str(a).replace(".", "p"), vdh(a)))
# Raw single-scattering albedos, to read off how many times the walk applies it.
for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print(build(0.02, "raw" + str(a).replace(".", "p"), a))
# The lobe stays selectable but the walk's own albedo is driven to nothing, so
# whatever is left is light that reached the camera without the medium taking
# its cut.
print(build(0.02, "bypass", 0.9, 100.0))
