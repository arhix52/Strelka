# MaterialX sample scenes

The geometry, materials and HDRIs the MaterialX distribution ships, wired up as
Strelka scenes. Nothing here is downloaded and nothing is copied: MaterialX is
already a submodule, so every asset is a **symlink** into
`third_party/materialx/resources/`. What is checked in is the part that is ours
— a `.toml` and a `<stem>_light.json` per scene, a few hundred bytes in total
against 40 MB of assets.

```bash
cd build/Release
./StrelkaCLI -c ../../scenes/materialx/chess_set/chess_set.toml -o /tmp/chess.png
```

## Why these and not the validation ladder

The ladder is graded against Cycles and every row is a glTF material. These are
the opposite: authored MaterialX documents that Strelka's loader turns into
**OpenPBR** materials, so they exercise the model the ladder cannot reach. They
have no reference and are not a pass/fail gate — they are what you look at when
a change to the OpenPBR path needs to be seen rather than measured. For a
number, use `tools/parity/run_openpbr.py` and `run_openpbr_examples.py`.

| scene | what it is | materials |
|---|---|---|
| `chess_set` | the Open Chess Set, from `standard_surface_chess_set.mtlx` | 18 OpenPBR — marble subsurface, brass, glass finials, 43 maps |
| `shaderball` | the MaterialX shader ball | 1, the glTF material the `.glb` carries. No `.mtlx` is bound to it |
| `boombox` | the glTF sample BoomBox | **none** — renders white, see below |

`chess_set` is the useful one. It is the asset `docs/open-defects.md` entry 14
is written about, and the only thing in the tree that drives OpenPBR subsurface
from a texture rather than a constant.

## boombox has no material, and that is recorded rather than fixed

It renders as untextured white, and there are two separate reasons — the first
answered here, the second not.

`boombox.glb` declares no glTF materials at all (`"materials": null`), so there
is no material *name* for the loader to match `Material_boombox` against.
`boombox.mtlx` is therefore not a symlink to the MaterialX example but a copy of
it with a `<look>` added, which binds by *node* name instead
(`materialx_loader.cpp:1399`); the mesh node is called `BoomBox`.

That gets as far as the second reason, which is a gap in the loader:

    MaterialX .../boombox.mtlx: 'Material_boombox' is a <gltf_pbr>,
    which this loader does not map

`materialx_loader.cpp:1145-1155` handles `standard_surface` and
`open_pbr_surface`. Supporting `<gltf_pbr>` is not a third input table: the
document routes its maps through `gltf_colorimage`, `gltf_image`,
`gltf_normalmap` and a `separate3` with per-channel `output="outz"` selection,
and the loader's image handling knows `image`, `tiledimage` and `normalmap` with
no multioutput routing. That is a feature, and it is deliberately not here.

The `<look>` is kept anyway, precisely because it is what produces that warning.
Without it the document binds nothing and says nothing, and an unbound material
renders identically to an unmapped one — which is how this cost an investigation
once already.

Nothing else in the scene set depends on it: `boombox` is geometry, a camera and
an environment, and it is useful as those.

## Things that are the way they are for a reason

**The scene file is a symlink, and the sidecars are not.** Every sidecar is
found by the scene's stem — `<stem>_light.json`, `<stem>.mtlx`,
`<stem>_openpbr.json`, `<stem>_curves.bin`, `<stem>_camera.json` — so the scene
path must not be resolved through the link, or the whole search moves to
`third_party/` and the scene loads with none of them. `HeadlessApp.cpp` uses
`fs::absolute(...).lexically_normal()` rather than `fs::weakly_canonical()` for
exactly this; the symptom when it did not was a single line,
`No light in scene, adding default distant light`.

**`chess_set/` is a symlink too**, to the texture directory beside the `.mtlx`,
because the document names its maps as `chess_set/piece_base_color.jpg`
relative to itself.

**The cameras are authored in the `.toml`.** None of the three `.glb` files
carries one, and a camera fitted to the bounding box frames the chess set badly
— it is 0.705 square and 0.169 tall, so two thirds of the frame would be felt.

Derive them in the editor (`Frame selection`, then `Dump camera`) rather than
from the glTF accessor extents, which are pre-transform: `boombox`'s mesh
measures ±0.01 but its node carries `scale: [100, 100, 100]`, so a camera placed
from those numbers sits a hundred times too close — inside the model, which
renders as a black frame with a horizon across it and looks like a broken scene
rather than a misplaced camera.

**ACES and a stated exposure**, unlike the validation scenes, which use
`tonemap = "none"` because a curve would corrupt what they are graded against.
These are looked at, and the default photometric exposure renders them to
exactly zero in 8 bits (see CLAUDE.md).

**First run of `chess_set` takes about ten minutes.** 43 JPEGs at 0.62 GB
decoded and uploaded, once; the texture cache makes every run after it 25 s at
512 spp. It is not a hang, though it looks exactly like one.
