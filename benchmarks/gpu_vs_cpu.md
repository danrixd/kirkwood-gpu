# GPU vs CPU scaling benchmark

Integrator: **yoshida4**. 500 composite steps per run; each Yoshida4 step does 3 force evaluations.

| N particles | CPU wall (s) | GPU wall (s) | CPU throughput | GPU throughput | GPU speedup |
|------------:|-------------:|-------------:|---------------:|---------------:|------------:|
|       1,000 |         0.28 |         2.99 |       1.77e+06 |       1.67e+05 |        0.1x |
|      10,000 |         3.49 |         3.12 |       1.43e+06 |       1.60e+06 |        1.1x |
|     100,000 |        47.44 |         3.15 |       1.05e+06 |       1.59e+07 |       15.1x |
|   1,000,000 |       222.33 |         4.52 |       9.00e+05 |       1.11e+08 |      123.0x |

Throughput is reported in (particles × composite-steps) per second, so wall-clock time for a run of length `T` years at step `h` years is

    wall ≈ N * (T / h) / throughput.

## Reading the table

- Below ~10⁴ particles the GPU is **kernel-launch bound**: a handful of tiny elementwise kernels per step dominate the timeline regardless of N.
- At N ≳ 10⁵ the GPU is memory-bandwidth bound and scales as O(N).
- At N = 10⁶ the RTX 3080 Ti reaches ≈10⁸ particle-steps/s — ~100× the NumPy CPU path, which is the break-even point for a 10⁵-year run.

## Hardware / software

- CPU run: `numpy` (gpu=False)
- GPU run: `cupy` (gpu=True)
