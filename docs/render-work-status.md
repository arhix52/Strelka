# Metal render-work audit

Status: PASS. Remaining OPEN items are closed; no estimator or procedural-shape math changed.

- Expected work remains `O(active*spp*depth) + O(H*candidates) + O(H*neighbors)`.
- Guides and radiance share first-surface traversal; guides do not add primary/extend rays.
- Empty guide queue: 320×240 dispatched threads `76,800 -> 0`; 1920×1080 `2,073,600 -> 0`.
- Nonempty queue dispatches only compact work: `11,028/11,072` active/dispatched at 320p;
  `396,724/396,736` at 1080p.
- ReSTIR first-bounce NEE is zero. Candidate and reuse queries are zero. Final visibility is `<= H`.
- Static post-warm-up BLAS/TLAS build/refit is `0/0/0`.
- 512 moving lights, 32 frames, NEE and ReSTIR: BLAS/TLAS builds `0/0`; TLAS refits `32`, max `1/frame`;
  uploads `32`, mappings `32`, both max `1/frame`; each render pipeline dispatch count is `32`.
- Spatial upscale call graph: `HeadlessApp::run -> renderSync -> render -> Metal4 commit/wait -> readback blit`.
- Runtime readback CB ID 1: creation/encoder/copy/end/commit/wait/readback = `1/1/1/1/1/1/1`.
- The apparent second commit is a distinct Managed-output sync CB; it is not created for Shared output on Apple silicon.
- ReSTIR storage is 296 B/pixel: reservoirs 96, surface history 64, shading point 136.
- At 1080p, NEE ReSTIR-only allocation is `585.4 -> 0 MiB`; total wavefront allocation `1231.5 -> 654.0 MiB`.
- ReSTIR total is `1231.5 -> 1239.4 MiB`; the +7.9 MiB is the compact guide index queue.

Commits:

- `02ac340` — skip empty guide traversal dispatches.
- `afb4f9c` — allocate ReSTIR history only when enabled.
- `16adca0` — add moving-light and command-buffer audit probes.

Validation: 11 focused render-work cases / 28 assertions and 33 buffer-layout assertions pass;
full Debug `ctest` result is recorded in the JSON ledger.
