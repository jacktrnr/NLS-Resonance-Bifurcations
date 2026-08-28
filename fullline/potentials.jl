##################
# potentials.jl
# A large collection of compactly-supported 1D potentials on [a,b].
# Each constructor returns a callable V(x) that is exactly zero outside (a,b).
##################

# Utility: clamp ξ into (0,1) safely (used for profiles)
_clamp01(ξ) = ξ < 0 ? 0.0 : (ξ > 1 ? 1.0 : ξ)

# -----------------------
# Basic / piecewise
# -----------------------

"""
    square_well(a, b, V0)

Constant potential V = V0 on (a,b), 0 outside.
"""
square_well(a::Real, b::Real, V0::Real) = x -> (a < x < b ? V0 : 0.0)


"""
    two_step_potential(a, b, α; split=0.1)

Two-step compact potential:

- V = α on [a, b - split)
- V = -1.0 on [b - split, b)
- V = 0 elsewhere

Keyword:
- split : width of the rightmost step (default 0.1)
"""
function two_step_potential(a::Real, b::Real, α::Real; split::Real=0.1)
    return x -> begin
        if !(a < x < b)
            0.0
        elseif x < b - split
            α
        else
            -1.0
        end
    end
end


function three_step_potential(a::Real, b::Real, α::Real; left_frac::Real=0.25, right_frac::Real=0.25, peak::Real=1.0)
    L = b - a
    left_end  = a + left_frac * L
    right_start = b - right_frac * L
    return x -> begin
        if !(a < x < b)
            0.0
        elseif x < left_end
            peak
        elseif x >= right_start
            peak
        else
            α
        end
    end
end


# -----------------------
# Smooth / bump-based
# -----------------------

# basic smooth compact bump φ(ξ) = exp(-((ξ-c)*s)^2) * window(ξ)
# window uses cos-based taper to vanish at 0 and 1 smoothly
function _smooth_profile(ξ::Real, c::Real, s::Real)
    # compact window that vanishes at endpoints (C^∞ not necessary)
    window = 0.5 * (1 + cos(π * (2 * _clamp01(ξ) - 1)))  # vanishes at ξ=0,1
    return exp(-((ξ - c) * s)^2) * window
end

"""
    single_well(a, b, V0; skew=0.5, sharpness=4.0)

Smooth asymmetric single-well supported on (a,b).
`skew` in (0,1) moves the minimum; `sharpness` controls steepness.
"""
function single_well(a::Real, b::Real, V0::Real; skew::Real=0.5, sharpness::Real=4.0)
    width = b - a
    # precompute normalization on grid
    ξs = range(0.0, 1.0; length=400)
    prof = _smooth_profile.(ξs, skew, sharpness)
    profmax = maximum(prof)
    return x -> begin
        if !(a < x < b)
            0.0
        else
            ξ = (x - a) / width
            (V0 * _smooth_profile(ξ, skew, sharpness) / profmax)
        end
    end
end


"""
    smooth_double_well(a, b, V0; separation=0.25, sharpness=6.0)

Two smooth Gaussian-like wells inside (a,b), compactly tapered to zero.
"""
function smooth_double_well(a::Real, b::Real, V0::Real; separation::Real=0.25, sharpness::Real=6.0)
    width = b - a
    c1 = 0.5 - separation / 2
    c2 = 0.5 + separation / 2
    ξs = range(0.0, 1.0; length=500)
    profsum = [_smooth_profile(ξ, c1, sharpness) + _smooth_profile(ξ, c2, sharpness) for ξ in ξs]
    norm = maximum(profsum)
    return x -> begin
        if !(a < x < b)
            0.0
        else
            ξ = (x - a) / width
            V0 * (_smooth_profile(ξ, c1, sharpness) + _smooth_profile(ξ, c2, sharpness)) / norm
        end
    end
end


"""
    gaussian_bump(a, b, A; center_frac=0.5, sigma_frac=0.1)

A Gaussian bump truncated and scaled to be supported on (a,b).
- `A` amplitude (can be negative for well).
- `center_frac` fraction (0..1) for center position.
- `sigma_frac` controls width relative to (b-a).
"""
function gaussian_bump(a::Real, b::Real, A::Real; center_frac::Real=0.5, sigma_frac::Real=0.1)
    width = b - a
    x0 = a + center_frac * width
    σ = max(sigma_frac * width, 1e-12)
    return x -> begin
        if !(a < x < b)
            0.0
        else
            A * exp(-((x - x0)^2) / (2 * σ^2))
        end
    end
end


"""
    cosine_well(a, b, V0; n_periods=1)

Smooth cosine-shaped well supported on (a,b):
V(x) = V0 * sin^2(nπ (x-a)/(b-a)) inside (a,b), zero outside.
"""
cosine_well(a::Real, b::Real, V0::Real; n_periods::Int=1) = x -> begin
    if !(a < x < b)
        0.0
    else
        ξ = (x - a) / (b - a)
        V0 * sin(n_periods * π * ξ)^2
    end
end


"""
    polynomial_bump(a, b, A; degree=4)

Compact polynomial bump on (a,b): A * ξ^p (1-ξ)^p, with p = degree/2.
"""
function polynomial_bump(a::Real, b::Real, A::Real; degree::Int=4)
    p = max(1, Int(round(degree/2)))
    return x -> begin
        if !(a < x < b)
            0.0
        else
            ξ = (x - a) / (b - a)
            A * ξ^p * (1 - ξ)^p
        end
    end
end


"""
    tent_potential(a, b, A)

Piecewise linear tent (triangular) potential supported on (a,b).
"""
function tent_potential(a::Real, b::Real, A::Real)
    mid = 0.5 * (a + b)
    return x -> begin
        if !(a < x < b)
            0.0
        elseif x <= mid
            A * (x - a) / (mid - a)
        else
            A * (b - x) / (b - mid)
        end
    end
end


# -----------------------
# Asymmetric / skewed constructions
# -----------------------

"""
    skewed_single_well(a, b, V0; skew=0.3, sharpness=6.0)

An asymmetric smooth well that decays faster to one side than the other.
"""
function skewed_single_well(a::Real, b::Real, V0::Real; skew::Real=0.3, sharpness::Real=6.0)
    width = b - a
    # left and right profile shapes differ
    left_sharp = sharpness * (1.0 + (0.5 - skew))
    right_sharp = sharpness * (1.0 + (skew - 0.5))
    # normalize on grid
    ξs = range(0.0, 1.0; length=400)
    prof = [ (_smooth_profile(ξ, skew, left_sharp) * (ξ <= skew ? 1.0 : 0.0) +
              _smooth_profile(ξ, skew, right_sharp) * (ξ > skew ? 1.0 : 0.0)) for ξ in ξs ]
    norm = maximum(prof)
    return x -> begin
        if !(a < x < b)
            0.0
        else
            ξ = (x - a) / width
            (V0 * ((_smooth_profile(ξ, skew, left_sharp) * (ξ <= skew ? 1.0 : 0.0)) +
                   (_smooth_profile(ξ, skew, right_sharp) * (ξ >  skew ? 1.0 : 0.0))) / norm)
        end
    end
end


"""
    asymmetric_step_array(a, b, vals::AbstractVector)

Build a piecewise constant potential on (a,b) with `length(vals)` equal subintervals.
"""
function asymmetric_step_array(a::Real, b::Real, vals::AbstractVector)
    n = length(vals)
    cuts = [a + (i-1)/n * (b - a) for i in 1:(n+1)]
    return x -> begin
        if !(a < x < b)
            0.0
        else
            # find index
            t = (x - a) / (b - a)
            idx = clamp(Int(floor(t * n)) + 1, 1, n)
            return vals[idx]
        end
    end
end


# -----------------------
# Multi-well / structured
# -----------------------

"""
    multi_gaussian_wells(a, b, V0s::AbstractVector; centers_frac=nothing, σ_frac=0.05)

Sum of multiple Gaussian bumps placed in (a,b). V0s gives amplitudes (can be negative).
"""
function multi_gaussian_wells(a::Real, b::Real, V0s::AbstractVector; centers_frac=nothing, σ_frac::Real=0.05)
    m = length(V0s)
    width = b - a
    centers = centers_frac === nothing ? [ (i-0.5)/m for i in 1:m ] : centers_frac
    σ = max(σ_frac * width, 1e-12)
    return x -> begin
        if !(a < x < b)
            0.0
        else
            ξ = (x - a) / width
            s = zero(eltype(V0s))
            for (j, V0) in enumerate(V0s)
                c = centers[j]
                s += V0 * exp(-((ξ - c)^2) / (2 * (σ/width)^2))
            end
            return s
        end
    end
end


# -----------------------
# Misc / test / random
# -----------------------

"""
    random_bumps(a, b, n::Int; rng=Random.GLOBAL_RNG, max_amp=1.0, max_width_frac=0.2)

Generate a sum of `n` random compact Gaussian bumps inside (a,b). Useful for testing.
"""
function random_bumps(a::Real, b::Real, n::Int; rng=Random.GLOBAL_RNG, max_amp::Real=1.0, max_width_frac::Real=0.2)
    width = b - a
    amps = rand(rng, n) .* (2 .* max_amp) .- max_amp
    centers = rand(rng, n)
    widths = rand(rng, n) .* max_width_frac .+ 0.01
    return x -> begin
        if !(a < x < b)
            0.0
        else
            ξ = (x - a) / width
            s = 0.0
            for i in 1:n
                s += amps[i] * exp(-((ξ - centers[i])^2) / (2 * widths[i]^2))
            end
            return s
        end
    end
end


"""
    sech2_well(a, b, V0; center_frac=0.5, k=1.0)

Truncated sech^2 well on (a,b) (smooth and localized).
"""
function sech2_well(a::Real, b::Real, V0::Real; center_frac::Real=0.5, k::Real=1.0)
    width = b - a
    x0 = a + center_frac * width
    return x -> begin
        if !(a < x < b)
            0.0
        else
            V0 * (1 / cosh(k * (x - x0))^2)
        end
    end
end


"""
    double_well(a, b, V0)

Polynomial double-well supported on (a,b).
"""
double_well(a, b, V0) = x -> begin
    if a ≤ x ≤ b
        mid = (a+b)/2
        return V0 * ((x-a)*(x-b)) * ((x-mid)^2)
    else
        return 0.0
    end
end


# =============================================================================
# PAPER POTENTIAL (5.1) — Turner & Weinstein, Nonlinearity 2026
# =============================================================================

"""
    paper_potential(α, β)

The 2-parameter family from equation (5.1) of the paper:

    V(x; α, β) = -α W(x; β) / max_x W(x; β)

where

    W(x; β) = exp(-¼(x - β)²) · ½(1 + cos(πx)),   |x| ≤ 1

and V(x) = 0 for |x| > 1. The support is always [-1, 1].

Parameters:
  - α > 0: well depth
  - β: asymmetry parameter (β = 0 gives symmetric well; β = -11 in Fig 10)

# Examples
```julia
V = paper_potential(3.0, 0.0)     # Figure 7: symmetric
V = paper_potential(3.0, -11.0)   # Figure 10: asymmetric
```
"""
function paper_potential(α::Real, β::Real)
    # Precompute max of W on [-1, 1]
    function W(x)
        abs(x) > 1.0 && return 0.0
        return exp(-0.25 * (x - β)^2) * 0.5 * (1.0 + cos(π * x))
    end

    # Find max on a fine grid
    xs = range(-1.0, 1.0; length=10000)
    Wmax = maximum(W.(xs))
    Wmax = max(Wmax, 1e-30)  # safety

    return x -> begin
        if abs(x) > 1.0
            0.0
        else
            -α * W(x) / Wmax
        end
    end
end

"""
    find_threshold_alpha(β; α_range=(0.1, 50.0), tol=1e-6, N=2000)

Find α_★(β) at which the linear operator H_V develops a zero-energy
(threshold) resonance, i.e., w(0) = 0.

Uses bisection on the Wronskian evaluated at k = 0.
"""
function find_threshold_alpha(β::Real; α_range=(0.1, 50.0), tol::Real=1e-6, N::Int=2000)
    a, b = -1.0, 1.0

    function w_at_zero(α)
        Vfun = paper_potential(α, β)
        # At k=0: f₊(x) = 1 for x > b, f₋(x) = 1 for x < a
        # Integrate -u'' + V u = 0 from both sides and compute Wronskian
        u0_plus = ComplexF64[1.0, 0.0]  # f₊(b) = 1, f₊'(b) = 0
        u0_minus = ComplexF64[1.0, 0.0]  # f₋(a) = 1, f₋'(a) = 0

        function f!(du, u, p, x)
            du[1] = u[2]
            du[2] = Vfun(x) * u[1]
        end

        # f₊ integrated backwards from b
        prob_p = ODEProblem(f!, u0_plus, (b, a))
        sol_p = solve(prob_p, Tsit5(); reltol=1e-12, abstol=1e-14, save_everystep=false)

        # f₋ integrated forwards from a
        prob_m = ODEProblem(f!, u0_minus, (a, b))
        sol_m = solve(prob_m, Tsit5(); reltol=1e-12, abstol=1e-14, save_everystep=false)

        # Evaluate at midpoint x = 0
        # For f₊ at x=0: we need to get values at an intermediate point
        # Simpler: evaluate Wronskian at x = a (left boundary)
        fp_a = sol_p.u[end]    # f₊ at x = a
        fm_a = u0_minus        # f₋ at x = a: (1, 0)

        w = real(fm_a[1] * fp_a[2] - fm_a[2] * fp_a[1])
        return w
    end

    # Bisect
    α_lo, α_hi = α_range
    w_lo = w_at_zero(α_lo)
    w_hi = w_at_zero(α_hi)

    if w_lo * w_hi > 0
        @warn "w does not change sign in α range [$α_lo, $α_hi]. Try wider range."
        return NaN
    end

    for _ in 1:100
        α_mid = 0.5 * (α_lo + α_hi)
        w_mid = w_at_zero(α_mid)

        if abs(w_mid) < tol || (α_hi - α_lo) < tol
            return α_mid
        end

        if w_lo * w_mid < 0
            α_hi = α_mid
        else
            α_lo = α_mid
            w_lo = w_mid
        end
    end

    return 0.5 * (α_lo + α_hi)
end
