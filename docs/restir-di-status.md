Phase: ReSTIR DI performance triage complete; exact procedural geometry unchanged.
Benchmark: Apple M4 Pro, Release, 1920x1080, depth 4, 1 spp/frame, static camera; 8 warm-up + 32 measured frames, median of 5 launches. Quality uses a 128-NEE-frame equal-time budget and 512-spp reference.
| config (C/T/S/N) | uniform ms / rMSE / last | distributed ms / rMSE / last | occluded ms / rMSE / last | ReSTIR B/px |
|---|---:|---:|---:|---:|
| NEE | 65.06 / .00658 / 1.522 | 74.52 / .00460 / .872 | 27.14 / .00300 / .425 | 0* |
| 1/0/0/0 | 78.74 / .01450 / 1.388 | 83.71 / .00521 / .797 | 29.33 / .00296 / .348 | 296* |
| 1/1/0/0 | 86.07 / .01551 / 1.390 | 85.83 / .00530 / .878 | 30.67 / .00298 / .355 | 296* |
| 1/1/1/2 | 95.10 / .01282 / 1.386 | 88.20 / .00547 / .777 | 30.88 / .00319 / .651 | 296* |
| 1/1/1/4 | 105.23 / .01831 / 1.397 | 91.62 / .00567 / .804 | 30.87 / .00299 / .503 | 296* |
| 2/1/1/2 | 104.00 / .01108 / 1.378 | 97.87 / .00604 / .739 | 31.14 / .00290 / .223 | 296* |
| 4/1/1/2 | 119.47 / .01213 / 1.379 | 108.08 / .00651 / .712 | 31.64 / .00282 / .182 | 296* |
| 8/1/1/4 | 159.49 / .01618 / 1.383 | 129.48 / .00745 / .591 | 32.65 / .00349 / .293 | 296* |
Stage ms for 1/T/S2 (uniform / distributed / occluded): extend 6.97/29.22/16.55; initial net .69/2.66/.22; temporal 7.03/3.91/.23; spatial 13.04/4.78/.23; final 7.60/3.91/.21; shadow 1.92/10.74/2.17; accumulation .74/.56/.52.
Traversal: 1.122 IFT calls/shadow on sphere control; restart=0 in benchmark scenes, .0527/ray in alpha test; alpha shadow +1.41 ms; IFT traversal +0.36 ms vs rectangle triangle control. Candidate/reconstruction kernels bind no AS. *Buffers are allocated in NEE too: reservoirs 96, temporal 64, work 136 B/px (585.35 MiB at 1080p), so current physical delta is 0.
Decision: Pareto ReSTIR preset 2/T/S2; only occluded scene crosses NEE (-3.3% rMSE), no general crossover. Experimental/default OFF.
