"""Grade the emergent angular distribution against Chandrasekhar, not against Cycles.

    /Applications/Blender.app/Contents/Resources/<v>/python/bin/python3.x \
        tools/feature_tests/sss_halfspace_read.py

For a semi-infinite, isotropically scattering medium the bidirectional
reflectance has a closed form,

    f_r(mu, mu0) = (omega / 4 pi) * H(mu) H(mu0) / (mu + mu0),

so under a uniform sky the emergent radiance is

    L(mu) = (omega / 2) H(mu) * integral_0^1 mu0 H(mu0) / (mu + mu0) dmu0.

H is the solution of the non-linear integral equation below and is found here by
fixed-point iteration, which converges in a few dozen sweeps for any albedo under
one.

Both renders are read off the same disc and normalised at the centre, because the
absolute level is what the other two instruments already grade; what is being
compared here is the *shape*, and normalising removes every constant that a
half-space model would not know about -- the sky's units, the entry Fresnel, the
solid angle of a pixel.

mu is taken as sqrt(1 - (r/R)^2), the orthographic reading of the disc. The camera
is 2.2 m from a 0.48 m sphere, so the true view angle differs from that by under a
degree except in the last few percent of the radius, which the table stops short
of.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from exr_io import load_exr  # noqa: E402  (path set above)

NAME = "95_halfspace"
ROOT = os.path.join(HERE, "..", "..", "scenes", "feature_tests", NAME)
# Quadrature for the H equation and for the integral over the incident hemisphere.
NODES = 256


def chandrasekhar_h(omega, nodes=NODES, sweeps=200):
    """H(mu) for isotropic scattering, by fixed-point iteration.

    H(mu) = 1 + (omega/2) mu H(mu) integral_0^1 H(mu') / (mu + mu') dmu'
    """
    mu = (np.arange(nodes) + 0.5) / nodes
    w = 1.0 / nodes
    h = np.ones(nodes)
    for _ in range(sweeps):
        # kernel[i, j] = 1 / (mu_i + mu_j)
        kernel = 1.0 / (mu[:, None] + mu[None, :])
        integral = (kernel * h[None, :]).sum(axis=1) * w
        h_next = 1.0 / (1.0 - 0.5 * omega * mu * integral)
        if np.max(np.abs(h_next - h)) < 1e-12:
            h = h_next
            break
        h = h_next
    return mu, h


def emergent_radiance(omega, mu_query):
    """L(mu) under uniform incident radiance of one, up to a constant."""
    mu, h = chandrasekhar_h(omega)
    w = 1.0 / len(mu)
    h_at = np.interp(mu_query, mu, h)
    integral = np.array([(mu * h / (m + mu)).sum() * w for m in mu_query])
    return 0.5 * omega * h_at * integral


def disc_profile(path, cx, cy, radius, edges):
    """Mean radiance in annuli of the disc, returned against mu at each annulus."""
    img = load_exr(path)[..., :3].mean(axis=2)
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        out.append(float(img[m].mean()) if m.any() else float("nan"))
    return np.array(out)


def find_disc(path):
    """Centre and radius of the sphere, from the pixels brighter than the sky."""
    img = load_exr(path)[..., :3].mean(axis=2)
    h, w = img.shape
    # The sphere is whatever differs from the sky, in either direction: a medium
    # of albedo 0.95 under a sky of 1 emerges *darker* than the sky, so a
    # brighter-than test finds nothing at all.
    sky = np.median(np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]]))
    mask = np.abs(img - sky) > 0.05 * sky
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = float(xx[mask].mean()), float(yy[mask].mean())
    radius = math.sqrt(mask.sum() / math.pi)
    return cx, cy, radius


def main():
    import json

    with open(os.path.join(ROOT, NAME + ".gltf")) as f:
        doc = json.load(f)
    mat = next(m for m in doc["materials"] if m.get("name", "").startswith("sss"))
    omega = float(mat["extensions"]["STRELKA_materials_subsurface"]["scatterColor"][0])

    ref_path = os.path.join(ROOT, NAME + "_cycles.exr")
    our_path = os.path.join(ROOT, NAME + "_strelka.exr")
    cx, cy, radius = find_disc(ref_path)

    # Stop at 0.94 of the radius: past that a pixel straddles the silhouette and
    # averages sky into the sphere, which is a measurement artefact and not a
    # disagreement.
    edges = np.linspace(0.0, 0.94, 13)
    mid = 0.5 * (edges[:-1] + edges[1:])
    mu = np.sqrt(np.clip(1.0 - mid ** 2, 0.0, 1.0))

    ref = disc_profile(ref_path, cx, cy, radius, edges)
    ours = disc_profile(our_path, cx, cy, radius, edges)
    analytic = emergent_radiance(omega, mu)

    ref /= ref[0]
    ours /= ours[0]
    analytic /= analytic[0]

    print("single-scattering albedo %.4f, %d free paths across the body\n" % (omega, 96))
    print("%7s%9s%11s%10s%11s%10s" % ("r/R", "mu", "analytic", "cycles", "strelka", "ours/an"))
    for i in range(len(mu)):
        print("%7.3f%9.3f%11.4f%10.4f%11.4f%10.4f"
              % (mid[i], mu[i], analytic[i], ref[i], ours[i], ours[i] / analytic[i]))

    def rms(a):
        return float(np.sqrt(np.mean((a / analytic - 1.0) ** 2)))

    print("\nrms departure from the closed form:  cycles %.4f   strelka %.4f"
          % (rms(ref), rms(ours)))


if __name__ == "__main__":
    main()
