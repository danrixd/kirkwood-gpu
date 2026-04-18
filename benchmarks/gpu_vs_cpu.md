# GPU vs CPU scaling benchmark

Integrator: **yoshida4**. 500 composite steps per run; each Yoshida4 step does 3 force evaluations.

| N particles | CPU wall (s) | GPU wall (s) | CPU throughput | GPU throughput | GPU speedup |
|------------:|-------------:|-------------:|---------------:|---------------:|------------:|
|       1,000 |         0.28 |         2.88 |       1.77e+06 |       1.73e+05 |        0.1x |
|      10,000 |         3.49 |         2.40 |       1.43e+06 |       2.09e+06 |        1.5x |
|     100,000 |        47.44 |         2.78 |       1.05e+06 |       1.80e+07 |       17.1x |
|   1,000,000 |       222.33 |         5.18 |       9.00e+05 |       9.66e+07 |      107.3x |

Throughput is reported in (particles × composite-steps) per second, so wall-clock time for a run of length `T` years at step `h` years is

    wall ≈ N * (T / h) / throughput.

## Reading the table

- Below ~10⁴ particles the GPU is **kernel-launch bound**: a handful of tiny elementwise kernels per step dominate the timeline regardless of N.
- At N ≳ 10⁵ the GPU is memory-bandwidth bound and scales as O(N).
- At N = 10⁶ the RTX 3080 Ti reaches ≈10⁸ particle-steps/s — ~100× the NumPy CPU path, which is the break-even point for a 10⁵-year run.

## Hardware / software

- CPU run: `numpy` (gpu=False)
- GPU run: `cupy` (gpu=True)
