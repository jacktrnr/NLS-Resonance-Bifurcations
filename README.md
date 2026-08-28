# NLS-Resonance-Bifurcations

Numerical code for **"Resonance-induced nonlinear bound states"**  
*J. C. Turner and M. I. Weinstein, Nonlinearity (2026)*  
DOI: [10.1088/1361-6544/ae9de1](https://doi.org/10.1088/1361-6544/ae9de1)

## What this does

Computes nonlinear bound states of the focusing 1D NLS/GP equation

$$-\psi'' + V(x)\psi - |\psi|^2\psi = E\psi$$

with compactly supported potentials $V$, and their connection to the linear scattering data of $H_V = -\partial_x^2 + V$.

The code:
1. Finds **bound state poles** and **scattering resonance poles** (zeros of $w(k)$) and **transmission resonances** (zeros of $s_-(k)$) via quadratic eigenvalue problems
2. Traces **bifurcation branches** of nonlinear bound states using pseudo-arclength continuation
3. Verifies the **asymptotic predictions** of the paper (threshold convergence rates)

## Reproducing the paper

Each script in `reproduce/` generates one result from the paper:

| Script | Paper result | What it verifies |
|--------|-------------|------------------|
| `reproduce/table1.jl` | Table 1 | $E(\varepsilon)/\varepsilon \to -2/\pi^2$, $x_R(\varepsilon) \to 3/4$ |
| `reproduce/figure9_scattering.jl` | Figure 9 (left panels) | Poles/zeros of $w(k)$, $s_-(k)$ for square well at three depths |

Run any script from the repo root:

```bash
julia reproduce/table1.jl
```

## Structure

```
halfline/          Half-line problem (Dirichlet at x=0, support [0,b])
  core.jl            ODE shooting, norms, BifurcationKit continuation
  resonances.jl      Companion QEP for linear resonances
  potentials.jl      Compactly supported potentials (square, smooth, etc.)
  plotting.jl        Visualization
  dynamics.jl        Split-step time propagation
  run.jl             Full pipeline driver

fullline/          Full-line problem (support [a,b], soliton tails both sides)
  NLSBifurcation.jl  Module definition
  scattering.jl      QEP for w(k) and s₋(k) zeros (outgoing + transmission BCs)
  shooting.jl        ODE integration with ζ-parametrization
  continuation.jl    Seed finding + BifurcationKit + scattering-informed discovery
  potentials.jl      15+ potentials including paper_potential(α,β) from eq (5.1)
  spectral.jl        L₊/L₋ stability operators
  dynamics.jl        Split-step with absorbing BCs
  run.jl             Full pipeline driver

reproduce/         Paper reproduction scripts
  table1.jl          Table 1 verification
  figure9_scattering.jl   Figure 9 scattering data
```

## Dependencies

Julia 1.10+ with:

```julia
using Pkg
Pkg.add(["OrdinaryDiffEq", "BifurcationKit", "Accessors",
          "Arpack", "FFTW", "Plots", "LaTeXStrings", "Colors", "JLD2"])
```

Or activate the project environment: `julia --project=fullline/`

## Methods

**Scattering data** — The linear eigenvalue problems for $w(k) = 0$ (outgoing radiation BCs) and $s_-(k) = 0$ (transmission BCs) are discretized by finite differences on $[a,b]$ with ghost-point elimination of the Robin-type boundary conditions $\psi' = \pm ik\psi$. This yields a quadratic eigenvalue problem $(A + ik B)\psi = k^2\psi$, linearized to a $2N\times 2N$ companion matrix.

**Nonlinear bound states** — Solutions are found by ODE shooting through the potential region and matching to the exact homoclinic (soliton) tail at $x = b$. The Hamiltonian residual $H = \frac{1}{2}(\psi')^2 + \frac{1}{2}E\psi^2 + \frac{1}{4}\psi^4$ vanishes on the homoclinic orbit; its zeros in $(\zeta, E)$-space give bound states. Branches are continued in $E$ using pseudo-arclength continuation (BifurcationKit.jl).

**Verification** — Asymptotic predictions from Theorem 4.4 are tested by extracting $(E(\varepsilon), x_R(\varepsilon))$ along the continuation branch near threshold and comparing to the predicted rates.

## License

MIT
