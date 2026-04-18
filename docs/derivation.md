# CR3BP: Hamiltonian, Jacobi integral, resonance locations

This note covers the numerical-algorithmic side of the problem: the rotating-frame Hamiltonian that makes the CR3BP a one-degree-of-freedom-plus-time system with a single integral, the resonance-location formula used to label the Kirkwood gaps, and a short argument for why a symplectic integrator is mandatory for 10⁵-year integrations.

## 1. Setup

The planar circular restricted three-body problem (CR3BP) describes a massless test particle in the gravitational field of two primaries — here the Sun ($M_1$) and Jupiter ($M_2$) — that move on a circular Keplerian orbit about their common barycenter. The test particle has zero mass, so it does not perturb the primaries. Units are chosen such that

$$G\, M_\odot = 4\pi^2, \qquad [\text{length}] = \text{AU}, \qquad [\text{time}] = \text{yr},$$

so Kepler's third law for Earth reduces to $n^2 a^3 = 1$ at $a = 1$ AU, $T = 1$ yr. Jupiter's mass ratio is $q \equiv M_J / M_\odot = 1/1047.3486 \approx 9.547 \times 10^{-4}$. In the CR3BP it is more natural to use $\mu \equiv M_J / (M_\odot + M_J) = q/(1+q)$, placing the Sun at $(-\mu\, a_J, 0)$ and Jupiter at $((1-\mu)\, a_J, 0)$ in the rotating frame.

The Sun-Jupiter mean motion sets the rotation rate of the rotating frame:

$$n = \sqrt{\frac{G(M_\odot + M_J)}{a_J^3}}.$$

For $a_J = 5.2044$ AU this gives a Jupiter period $T_J = 2\pi / n \approx 11.86$ yr.

## 2. Rotating-frame Hamiltonian

Let $(x, y)$ be Cartesian coordinates in the frame co-rotating with the primaries at angular velocity $n\,\hat z$. Define the effective potential

$$U_{\text{eff}}(x, y) = -\frac{1}{2} n^2 (x^2 + y^2) - \frac{GM_\odot}{r_1} - \frac{GM_J}{r_2},$$

where $r_1 = \sqrt{(x + \mu a_J)^2 + y^2}$ and $r_2 = \sqrt{(x - (1-\mu)a_J)^2 + y^2}$. The equations of motion of a test particle in the rotating frame are

$$\ddot x - 2n\,\dot y = -\partial_x U_{\text{eff}},$$
$$\ddot y + 2n\,\dot x = -\partial_y U_{\text{eff}}.$$

The Coriolis cross-terms $\mp 2n\dot y, \pm 2n\dot x$ carry no explicit position dependence, so they do no work, which gives rise to the single conserved quantity.

## 3. The Jacobi integral

Multiplying the EOM by $\dot x$ and $\dot y$ and adding, the Coriolis terms cancel and we are left with

$$\frac{d}{dt}\left[\tfrac{1}{2}(\dot x^2 + \dot y^2) + U_{\text{eff}}\right] = 0.$$

The quantity

$$C_J = n^2(x^2 + y^2) + \frac{2GM_\odot}{r_1} + \frac{2GM_J}{r_2} - (\dot x^2 + \dot y^2)$$

is the **Jacobi integral**. In the non-integrable CR3BP it is the *only* isolating integral; conservation of $C_J$ to machine precision is therefore the strongest numerical check we can run.

For an inertial-frame integration it is convenient to use the frame-independent form

$$C_J = 2\,(n L_z - E), \qquad L_z = x\dot y - y\dot x, \qquad E = \tfrac{1}{2}v^2 - \tfrac{GM_\odot}{r_1} - \tfrac{GM_J}{r_2},$$

which follows from $v_{\text{rot}}^2 = v_{\text{inertial}}^2 + n^2 r^2 - 2 n L_z$. This is the formula used in `kirkwood_gpu.physics.jacobi_constant`, avoiding any need to transform coordinates at diagnostic time.

## 4. Mean-motion resonance locations

A test particle in a $p\!:\!q$ **interior** mean-motion resonance satisfies $T_{\text{ast}} / T_J = q/p$ with $p > q$. Kepler's third law gives

$$\frac{a_{\text{ast}}}{a_J} = \left(\frac{q}{p}\right)^{2/3},$$

reproduced by `physics.resonance_semimajor_axis`. The 3:1 and 2:1 resonances that produce the dominant Kirkwood gaps sit at

| Resonance | $a$ (AU)  | Physical feature           |
|-----------|-----------|-----------------------------|
| 3:1       | **2.502** | Kirkwood gap                |
| 5:2       | 2.825     | Kirkwood feature            |
| 7:3       | 2.958     | Kirkwood feature            |
| 2:1       | **3.278** | Kirkwood gap (Hecuba gap)   |
| 3:2       | 3.971     | Hilda group (stable libration) |

at $a_J = 5.2044$ AU. The 3:1 is the one whose **chaotic** origin Wisdom (1982) first demonstrated; the 2:1 mechanism is subtler and involves secular resonance sweeping.

## 5. Why symplectic matters here

For a non-integrable Hamiltonian system integrated over $10^5$ periods of the perturber, the *phenomenon we are trying to see* is a slow diffusive depletion of phase space near the resonances. If our integrator has secular drift in the energy (or Jacobi constant), that drift is indistinguishable from — and swamps — the physical signal.

A generic order-$k$ explicit integrator such as RK4 has a local truncation error that is $O(h^{k+1})$ per step, but the *global* energy error grows as $O(N \cdot h^{k+1}) = O(T \cdot h^k)$: **linearly with integration time**. For $T \sim 10^5$ yr and $h \sim 0.06$ yr, even RK4's tiny per-step error compounds into visible drift long before the gaps finish carving.

A symplectic integrator, by contrast, preserves an *augmented* "shadow Hamiltonian" $\tilde H = H + h^2 H_2 + \dots$ exactly. The error on the *true* energy $H$ therefore oscillates with bounded amplitude $O(h^k)$ rather than drifting linearly. For the Störmer-Verlet leapfrog ($k=2$) and the 4th-order Yoshida composition used as the default in this repo, the Jacobi constant stays bounded for the entire $10^5$-year integration, as the `test_jacobi_constant.py` test and the per-run diagnostic both confirm.

This is not a cosmetic choice. Using RK4 at the same step size would, over $10^5$ years, drift $C_J$ by a factor of order unity — more than enough to artificially fill or empty the very resonance zones whose statistics are the headline result.

![drift](drift_comparison.png)

The figure above (generated by `docs/make_drift_figure.py`) shows the point directly: on a Kepler circular orbit at $a = 2.5$ AU with $dt = T/100$, RK4 drifts linearly in time (reaching $3.4 \times 10^{-5}$ at 200 orbits), while leapfrog and Yoshida4 oscillate with bounded amplitude ($1.7 \times 10^{-7}$ and $6 \times 10^{-14}$ respectively — Yoshida4 is essentially at machine precision). Scale this up to the $10^5$-year horizon and the RK4 line crosses the physical-signal amplitude well before the gaps have finished carving.

## 6. The integration stack

- **Frame.** Inertial barycentric frame. The Sun and Jupiter move on analytic circles at $n$; we never integrate them.
- **Integrators (three, all symplectic).**
  - **Yoshida 4 (default).** 4th-order composition of KDK leapfrog: three substeps per composite step with weights $w_1 = (2 - 2^{1/3})^{-1}$, $w_0 = 1 - 2 w_1$. General-purpose; doesn't exploit the near-Kepler structure.
  - **Leapfrog (Störmer-Verlet).** 2nd-order KDK. Faster per step than Yoshida4 but needs smaller $h$ for the same accuracy.
  - **Wisdom-Holman map.** Operator-split in *heliocentric* coordinates: the Hamiltonian is written as $H = H_{\text{Kepler}} + H_{\text{pert}}$, and the step is `drift(h/2) · kick(h) · drift(h/2)`, where `drift` advances each particle along its instantaneous Kepler orbit *analytically* (f/g functions from Danby 1988) and `kick` is the usual Jupiter-perturbation force evaluation. Because the Kepler drift is exact, the step size is limited only by the perturbation timescale, not by the asteroid orbital period. In practice this lets us take $h = T_J / 20$ instead of $T_J / 200$ — a 10× step-size gain.
- **Step sizes.** $h_{\text{Yoshida4}} = T_J / 200 \approx 0.059$ yr (~50 substeps per asteroid orbit at $a = 2.5$ AU). $h_{\text{WH}} = T_J / 20 \approx 0.59$ yr (one substep every ~5 asteroid orbits, but the Kepler drift advances them exactly).
- **Vectorization.** The test particles form a single batch of size $N$. The acceleration is a purely data-parallel elementwise kernel over $N$; the Wisdom-Holman Kepler drift is also fully elementwise, implemented as a single fused CuPy `ElementwiseKernel` (the Newton iteration for eccentric anomaly is per-particle in-kernel, no Python-level loop).
- **Back-end.** A single `xp` alias switches between NumPy (CPU fallback) and CuPy (GPU). All physics code is backend-agnostic.

## 7. Wisdom-Holman: why the analytic Kepler drift matters

The Wisdom-Holman map (Wisdom & Holman 1991) is the right tool for this exact problem. Here is the argument in three lines.

For the CR3BP, written in heliocentric canonical coordinates $(\vec\rho, \vec p)$ where the Sun sits at the origin, the Hamiltonian splits cleanly as

$$H = \underbrace{\tfrac{1}{2}|\vec p|^2 - \frac{GM_\odot}{|\vec\rho|}}_{H_K\text{: pure Kepler}} \;+\; \underbrace{-\frac{GM_J}{|\vec\rho - \vec R_J(t)|} - \frac{GM_J\,\vec\rho\cdot\vec R_J(t)}{|\vec R_J|^3}}_{H_{\text{pert}}\text{: Jupiter direct + indirect}},$$

where $\vec R_J(t)$ is the known analytic heliocentric position of Jupiter. The two pieces are separately integrable: $H_K$ is just Kepler's problem, advanced by the f/g functions in closed form; $H_{\text{pert}}$ is a pure *kick* (depends only on positions, integrated trivially). The leapfrog of the two maps — `drift(h/2)  · kick(h) ·  drift(h/2)` — is a 2nd-order symplectic integrator for $H$ with per-step error $O(h^3 |H_{\text{pert}}|/|H_K|)$. Crucially, the *Kepler* part carries zero discretization error at any $h$, so $h$ is capped only by the *perturbation* dynamics. Since $|H_{\text{pert}}|/|H_K| \sim m_J / m_\odot \approx 10^{-3}$, this gains roughly an order of magnitude in allowable step size versus a generic leapfrog on the full Hamiltonian — $h \sim T_J / 20$ rather than $T_J / 200$.

Trade-off: each Wisdom-Holman step costs ~2–3× a Yoshida4 step (a Newton solve for the eccentric anomaly dominates). With a 10× step-size gain, the net speedup is ~3–5× for the same Jacobi-conservation budget. That is what turns a 10⁶-year run from 26 hours of Yoshida4 into ~5 hours of Wisdom-Holman on the same GPU.

## References

- Wisdom, J. (1982). *The origin of the Kirkwood gaps: a mapping for asteroidal motion near the 3/1 commensurability.* AJ 87, 577–593.
- Wisdom, J. (1983). *Chaotic behavior and the origin of the 3/1 Kirkwood gap.* Icarus 56, 51–74.
- Murray, C. D. & Dermott, S. F. (1999). *Solar System Dynamics.* Cambridge. Ch. 3, 8, 9.
- Yoshida, H. (1990). *Construction of higher order symplectic integrators.* Phys. Lett. A 150, 262–268.
- Wisdom, J. & Holman, M. (1991). *Symplectic maps for the N-body problem.* AJ 102, 1528–1538.
- Danby, J. M. A. (1988). *Fundamentals of Celestial Mechanics*, 2nd ed., Ch. 6 (f/g functions and Kepler advance).
- Hairer, Lubich, Wanner (2006). *Geometric Numerical Integration*, Ch. IX (backward-error analysis of symplectic integrators).
