# Metal render-work audit

Status: PASS for the measured static Metal wavefront path; 3 OPEN items remain.

- Expected work: `O(active*spp*depth) + O(H*candidates) + O(H*neighbors)`.
- 320×240, 1 spp, depth 1, ReSTIR c4/T/S2: 76,800 primary/extend rays, H=53,248,
  212,992 candidates, 66,215 spatial merges, 34,204 final visibility rays, 111,004 total queries.
- Guides OFF/ON: primary and extend stay 76,800/76,800; guide-only rays are zero on Cornell.
- Candidate and temporal/spatial reuse queries are zero; final visibility is <= H.
- ReSTIR first-bounce NEE is zero; depth-4 secondary NEE remains enabled (89,083 samples).
- Static post-load frame: BLAS/TLAS build/refit = 0/0/0; light upload remains change-gated.
- History uses frame-parity ping-pong; no whole-reservoir copy and one fused clear per frame.
- No all-light loop remains in extend, shadow, materials, or reservoir reconstruction.

Confirmed fixes:

- `efb4ea8`: temporal reuse fused into first-hit shade; spatial/final only run on bounce 0.
  ReSTIR depth-4 dispatches: 42 -> 32; ray/intersection counts unchanged.
- `96634c6`: disabled/zero-neighbor spatial pass skipped; depth-1 dispatches: 11 -> 10.
- `1ce668e`: miss evaluates compact infinite lights, not every analytic light.
  512 finite rect lights: 11,807,744 rejected inspections -> 0; 2.24 -> 1.99 ms.
- `45d4bf3`, `ac08f53`, `e23fb80`, `d0055a3`: Debug-only audit JSON and 7 focused invariants.

Timing medians (5 short Release runs, ms): guides off/on 0.7/0.9; NEE/ReSTIR c1 0.7/1.0;
320×240/1080p 1.0/23.3; depth 1/4 1.0/2.6; candidates 1/4 1.0/1.5;
neighbors 0/2/4 0.9/1.0/1.5. Output EXRs were byte-identical across math-preserving fixes.

OPEN: empty guide dispatch needs a measured compact-queue A/B; moving-light row lacks a headless driver;
the unmeasured spatial-upscale readback has a duplicate `commit()` call. Full details and ledger are in
`docs/render-work-audit.json`.
