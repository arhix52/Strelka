# Reconstruction-filter validation

This scene isolates pixel reconstruction from textures, lighting, depth of field, and denoising. Its panels contain a Siemens star, a zone plate, a vertical frequency sweep, subpixel diagonal lines, 4/2/1-pixel checkerboards, and a diagonal grating.

Render `reconstruction_filter.toml` with `--reconstruction-filter box`, `tent`, `mitchell`, or `lanczos2` for a static comparison. Render `reconstruction_filter_shifted.toml` as well to compare stability after an exact half-pixel camera translation. Above-Nyquist detail should converge toward a stable gray instead of producing false bands or shimmer; Mitchell and Lanczos 2 should retain noticeably more of the resolvable line detail than Tent.

Signed filters must not depend on launch batching. Compare linear EXRs at a fixed total SPP, for example:

```bash
build/Release/StrelkaCLI -c scenes/validation/reconstruction_filter/reconstruction_filter.toml \
  -o /tmp/lanczos-b1.exr --spp 64 --spp-per-launch 1 --reconstruction-filter lanczos2
build/Release/StrelkaCLI -c scenes/validation/reconstruction_filter/reconstruction_filter.toml \
  -o /tmp/lanczos-b64.exr --spp 64 --spp-per-launch 64 --reconstruction-filter lanczos2
```

The images can differ by float summation order, but should otherwise be the same estimator.
