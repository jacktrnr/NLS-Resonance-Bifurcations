# NLSBifurcations.jl

Numerical code for **"Resonance-induced nonlinear bound states"**  
J. C. Turner and M. I. Weinstein, *Nonlinearity* (2026)  
DOI: [10.1088/1361-6544/ae9de1](https://doi.org/10.1088/1361-6544/ae9de1)

## Overview

Computes nonlinear bound states of the focusing 1D NLS/GP equation

$$-\psi'' + V(x)\psi - |\psi|^2\psi = E\psi$$

with compactly supported potentials, and their connection to the linear scattering data of $H_V = -\partial_x^2 + V$.

The code:
1. Finds **bound state poles** and **scattering resonance poles** (zeros of $w(k)$) and **transmission resonances** (zeros of $s_-(k)$) via quadratic eigenvalue problems
2. Traces **nonlinear bifurcation branches** $\mathcal{N}[\psi_E]$ vs $E$ using pseudo-arclength continuation, with automatic fold detection and recovery
3. Verifies the **asymptotic predictions** of the paper (threshold convergence rates, Theorem 4.4)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jacktrnr/NLS-Resonance-Bifurcations")
```

Or clone and activate locally:

```bash
git clone https://github.com/jacktrnr/NLS-Resonance-Bifurcations.git
cd NLS-Resonance-Bifurcations
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

```julia
using NLSBifurcations

# Define a potential on [-1, 1]
a, b = -1.0, 1.0
Vfun = square_well(a, b, -2.0)

# Compute linear scattering data
scat = compute_all_scattering_data(a, b, Vfun; N=500, k_max=4.0)

# Trace complete nonlinear branches (automatic fold recovery)
branches = trace_complete_branches(a, b, Vfun;
    E_list=[-0.3, -0.8, -1.3], p_min=-4.0)
```

### Paper potential (eq 5.1)

```julia
V = paper_potential(3.0, 0.0)     # Symmetric (Figure 7)
V = paper_potential(24.0, -11.0)  # Asymmetric (Figure 10)
```

## Reproducing paper results

| Script | Paper result |
|--------|-------------|
| `examples/table1.jl` | Table 1 — threshold resonance asymptotics |
| `examples/figure9.jl` | Figure 9 — square well bifurcation diagrams |
| `examples/figure9_complete.jl` | Figure 9(a) — complete branch topology with fold |

```bash
julia --project=. examples/table1.jl
```

## Methods

**Scattering data** — Discretize $-\psi'' + V\psi = k^2\psi$ on $[a,b]$ with ghost-point elimination of the Robin BCs ($\psi' = \pm ik\psi$). Outgoing BCs give zeros of $w(k)$; transmission BCs give zeros of $s_-(k)$. Both yield quadratic eigenvalue problems $(A + ikB)\psi = k^2\psi$, linearized to $2N \times 2N$ companion matrices.

**Nonlinear continuation** — Bound states are found by shooting with the $\zeta$-parametrized initial amplitude $c = \sqrt{-2E}\tanh\zeta$, matching to soliton tails via the Hamiltonian residual. Branches are continued in $E$ using PALC (BifurcationKit.jl). When the continuation stalls at a fold, the code automatically searches for seeds nearby and continues past it.

**Verification** — Theorem 4.4 predictions ($E(\varepsilon)/\varepsilon \to -\frac{1}{2}U_\star(b)^2$, $x_R(\varepsilon) \to$ finite limit) are confirmed numerically in `examples/table1.jl`.

## Package structure

```
src/
  NLSBifurcations.jl   Main module
  scattering.jl         QEP for w(k) and s₋(k) zeros
  potentials.jl         Compactly supported potentials + paper_potential(α,β)
  continuation.jl       Seed finding, PALC continuation, fold recovery
  shooting.jl           ODE integration
  glue.jl               Soliton tail matching
  spectral.jl           L₊/L₋ stability operators
  dynamics.jl           Split-step time evolution

examples/              Paper reproduction scripts
halfline/              Half-line code (Dirichlet BC, used for Table 1)
test/                  Package tests
```

## License

MIT
