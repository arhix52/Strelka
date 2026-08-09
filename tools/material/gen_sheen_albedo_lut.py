#!/usr/bin/env python3
"""
Generate the Charlie sheen directional-albedo table baked into
src/material/include/strelka/material/sheen_albedo_lut.h.

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        -P tools/material/gen_sheen_albedo_lut.py

(Blender only because it ships numpy; any python with numpy works.)

E(mu, roughness) is the hemispherical integral of the sheen BRDF weighted by the
cosine -- what fraction of the light arriving from `mu` the sheen layer sends
back. KHR_materials_sheen needs it to scale the base layer down by what the
fabric already reflected; without it a towel reflects more light than fell on it.

Note the table exceeds 1 at low roughness and grazing angles. That is not a bug
in the integration, it is Ashikhmin's visibility term, which the extension is
specified against and which does not conserve energy on its own. The renderer
normalises by it -- see sheen_albedo() -- rather than pretending otherwise.
"""

import numpy as np

N = 16


def directional_albedo(mu, roughness, ntheta=1024, nphi=1024):
    alpha = max(roughness * roughness, 1e-3)
    inv_a = 1.0 / alpha
    mu = max(mu, 1e-4)
    sv = np.sqrt(max(1.0 - mu * mu, 0.0))
    V = np.array([sv, 0.0, mu])

    th = (np.arange(ntheta) + 0.5) / ntheta * (np.pi / 2)
    ph = (np.arange(nphi) + 0.5) / nphi * (2 * np.pi)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    st, ct = np.sin(TH), np.cos(TH)
    L = np.stack([st * np.cos(PH), st * np.sin(PH), ct], axis=-1)

    H = L + V
    H = H / np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-9)
    NdotH = np.clip(H[..., 2], 0.0, 1.0)
    sin2h = np.maximum(1.0 - NdotH * NdotH, 1e-7)

    D = (2.0 + inv_a) * np.power(sin2h, inv_a * 0.5) / (2 * np.pi)
    Vis = 1.0 / (4.0 * (ct + mu - ct * mu) + 1e-7)
    return float(np.sum(D * Vis * ct * st) * (np.pi / 2 / ntheta) * (2 * np.pi / nphi))


def main():
    mus = [(i + 0.5) / N for i in range(N)]
    rghs = [(j + 0.5) / N for j in range(N)]
    table = [[directional_albedo(m, r) for m in mus] for r in rghs]

    rows = "\n".join(
        "    " + ", ".join(f"{v:.5f}f" for v in row) + ("," if j + 1 < N else "")
        for j, row in enumerate(table)
    )
    print(f"// peak E = {max(max(r) for r in table):.5f}")
    print(rows)


if __name__ == "__main__":
    main()
