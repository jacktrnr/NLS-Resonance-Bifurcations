###############################################
# scattering.jl — Linear scattering data via QEP
###############################################
#
# Finds zeros of w(k) and s₋(k) for H_V = -∂² + V(x) on [a,b]
# using quadratic eigenvalue problems (QEP) with ghost-point BCs.
#
# Problem 1 — Outgoing radiation (zeros of w(k)):
#   -ψ'' + Vψ = k²ψ on [a,b]
#   ψ'(a) = -ik ψ(a)     (outgoing left)
#   ψ'(b) = +ik ψ(b)     (outgoing right)
#
# Problem 2 — Right transmission (zeros of s₋(k)):
#   -ψ'' + Vψ = k²ψ on [a,b]
#   ψ'(a) = +ik ψ(a)     (transmission left)
#   ψ'(b) = +ik ψ(b)     (transmission right)
#
# Both yield QEPs: (A + ik·B) ψ = k² ψ
# Linearized to 2N×2N companion: C [ψ; kψ] = k [ψ; kψ]
#
# Reference: Section 2, eqs (2.12)–(2.14) of the paper.
#
###############################################

using LinearAlgebra, Printf

# =============================================================================
# QEP CONSTRUCTION
# =============================================================================

"""
    _build_scattering_qep(a, b, Vfun, N; problem=:outgoing)

Build the N×N matrices A and B for the QEP (A + ik·B)ψ = k²ψ.

`problem`:
  - `:outgoing`      — zeros of w(k): ψ'(a) = -ik ψ(a), ψ'(b) = ik ψ(b)
  - `:transmission`  — zeros of s₋(k): ψ'(a) = +ik ψ(a), ψ'(b) = ik ψ(b)
"""
function _build_scattering_qep(a, b, Vfun, N::Int; problem::Symbol=:outgoing)
    h = (b - a) / (N + 1)
    x = [a + j * h for j in 1:N]

    # Standard FD Laplacian (-d²/dx²) tridiagonal
    A = zeros(Float64, N, N)
    for j in 1:N
        A[j, j] = 2.0 / h^2 + Vfun(x[j])
        if j > 1
            A[j, j-1] = -1.0 / h^2
        end
        if j < N
            A[j, j+1] = -1.0 / h^2
        end
    end

    # Ghost-point modifications:
    #
    # Right BC (both problems): ψ'(b) = ik ψ(b)
    #   Ghost: ψ_{N+1} = ψ_{N-1} + 2ikh ψ_N
    #   Row N: (2-2ikh)ψ_N/h² - 2ψ_{N-1}/h² + V_N ψ_N = k² ψ_N
    A[N, N] = 2.0 / h^2 + Vfun(x[N])
    if N > 1
        A[N, N-1] = -2.0 / h^2
    end

    # Left BC depends on problem type:
    if problem == :outgoing
        # ψ'(a) = -ik ψ(a)
        # Ghost: ψ₀ = ψ₂ + 2ikh ψ₁
        # Row 1: (2-2ikh)ψ₁/h² - 2ψ₂/h² + V₁ψ₁ = k²ψ₁
        A[1, 1] = 2.0 / h^2 + Vfun(x[1])
        if N > 1
            A[1, 2] = -2.0 / h^2
        end
    elseif problem == :transmission
        # ψ'(a) = +ik ψ(a)
        # Ghost: ψ₀ = ψ₂ - 2ikh ψ₁
        # Row 1: (2+2ikh)ψ₁/h² - 2ψ₂/h² + V₁ψ₁ = k²ψ₁
        A[1, 1] = 2.0 / h^2 + Vfun(x[1])
        if N > 1
            A[1, 2] = -2.0 / h^2
        end
    else
        error("Unknown problem type: $problem. Use :outgoing or :transmission.")
    end

    # B matrix (the ik-linear coefficient)
    B = zeros(Float64, N, N)

    # Right BC contributes B[N,N] = -2/h in both cases
    B[N, N] = -2.0 / h

    # Left BC:
    if problem == :outgoing
        # From (2 - 2ikh)/h²: the ik coefficient is -2/h
        B[1, 1] = -2.0 / h
    elseif problem == :transmission
        # From (2 + 2ikh)/h²: the ik coefficient is +2/h
        B[1, 1] = +2.0 / h
    end

    return A, B, x
end

"""
    _solve_companion_qep(A, B; k_max=10.0)

Solve the QEP (A + ik·B)ψ = k²ψ via companion linearization.

Returns all eigenvalues k and eigenvectors (upper N block).
"""
function _solve_companion_qep(A, B; k_max::Real=10.0)
    N = size(A, 1)

    # Companion: [0  I; A  iB] [ψ; w] = k [ψ; w]  where w = kψ
    C = zeros(ComplexF64, 2N, 2N)
    C[1:N, (N+1):(2N)] = I(N)
    C[(N+1):(2N), 1:N] = A
    C[(N+1):(2N), (N+1):(2N)] = im .* B

    F = eigen(C)

    # Filter by |k| < k_max
    mask = abs.(F.values) .< k_max
    k_vals = F.values[mask]
    vecs = F.vectors[:, mask]

    return k_vals, vecs
end

# =============================================================================
# PUBLIC API
# =============================================================================

"""
    compute_w_zeros(a, b, Vfun; N=400, k_max=10.0)

Find zeros of w(k) — bound state poles and scattering resonances.
Solves the outgoing radiation QEP on [a,b].

Returns vector of NamedTuples with fields:
  k, κ (= Im(k)), E (= -κ² for purely imaginary k), type ∈ {:bound, :resonance}
"""
function compute_w_zeros(a, b, Vfun; N::Int=400, k_max::Real=10.0, imag_tol::Real=0.01)
    A, B, x = _build_scattering_qep(a, b, Vfun, N; problem=:outgoing)
    k_vals, vecs = _solve_companion_qep(A, B; k_max=k_max)

    results = NamedTuple[]
    for (i, k) in enumerate(k_vals)
        # Keep only those on or very near the imaginary axis
        abs(real(k)) > imag_tol * max(abs(imag(k)), 1.0) && continue
        # Skip near-zero
        abs(k) < 1e-6 && continue

        κ = imag(k)
        E = -κ^2  # for k = iκ: k² = -κ²
        type = κ > 0 ? :bound : :resonance

        # Extract eigenfunction (upper N block)
        u = vecs[1:N, i]
        # Normalize max = 1
        u ./= maximum(abs.(u))

        push!(results, (; k, κ, E, type, u_vec=u, x))
    end

    # Sort by distance from real axis
    sort!(results; by=r -> abs(r.κ))
    return results
end

"""
    compute_s_minus_zeros(a, b, Vfun; N=400, k_max=10.0)

Find zeros of s₋(k) — right transmission resonances.
Solves the transmission QEP on [a,b].

Returns vector of NamedTuples with fields:
  k, κ (= Im(k)), E, type=:transmission
"""
function compute_s_minus_zeros(a, b, Vfun; N::Int=400, k_max::Real=10.0, imag_tol::Real=0.05)
    A, B, x = _build_scattering_qep(a, b, Vfun, N; problem=:transmission)
    k_vals, vecs = _solve_companion_qep(A, B; k_max=k_max)

    results = NamedTuple[]
    for (i, k) in enumerate(k_vals)
        # Keep near-imaginary axis OR near-real axis
        # Transmission resonances can be on the real axis too
        on_imag = abs(real(k)) < imag_tol * max(abs(imag(k)), 1.0)
        on_real = abs(imag(k)) < imag_tol * max(abs(real(k)), 1.0) && abs(real(k)) > 0.1

        (!on_imag && !on_real) && continue
        abs(k) < 1e-6 && continue

        κ = imag(k)
        E = real(k^2)

        u = vecs[1:N, i]
        u ./= maximum(abs.(u))

        push!(results, (; k, κ, E, type=:transmission, u_vec=u, x))
    end

    sort!(results; by=r -> abs(r.κ))
    return results
end

"""
    compute_all_scattering_data(a, b, Vfun; N=400, k_max=5.0)

Compute complete scattering data: bound state poles, scattering resonance
poles, and transmission resonances.

Returns a NamedTuple with fields:
  bound_poles, scat_poles, trans_res (each a vector of NamedTuples)
"""
function compute_all_scattering_data(a, b, Vfun; N::Int=400, k_max::Real=5.0, verbose::Bool=true)
    verbose && println("  Computing w(k) zeros (outgoing QEP, N=$N)...")
    w_res = compute_w_zeros(a, b, Vfun; N=N, k_max=k_max)

    verbose && println("  Computing s₋(k) zeros (transmission QEP, N=$N)...")
    s_res = compute_s_minus_zeros(a, b, Vfun; N=N, k_max=k_max)

    bound_poles = filter(r -> r.type == :bound, w_res)
    scat_poles  = filter(r -> r.type == :resonance, w_res)
    trans_res   = s_res

    if verbose
        println("\n  Scattering data summary:")
        println("    Bound state poles (w=0, κ>0): $(length(bound_poles))")
        for r in bound_poles
            @printf("      k = %.6fi,  E = %.6f\n", r.κ, r.E)
        end
        println("    Scattering resonance poles (w=0, κ<0): $(length(scat_poles))")
        for r in scat_poles
            @printf("      k = %.6fi,  E = %.6f\n", r.κ, r.E)
        end
        println("    Transmission resonances (s₋=0): $(length(trans_res))")
        for r in trans_res
            @printf("      k = %.4f + %.4fi\n", real(r.k), imag(r.k))
        end
    end

    return (; bound_poles, scat_poles, trans_res)
end

# =============================================================================
# THRESHOLD DETECTION
# =============================================================================

"""
    find_threshold_alpha_qep(β; α_range=(0.1, 50.0), N=400, tol=1e-6)

Find α_★(β) where a scattering resonance pole crosses k=0 (threshold).
Uses the smallest |κ| of the outgoing QEP: at threshold, κ → 0.
"""
function find_threshold_alpha_qep(β::Real; α_range=(0.1, 50.0), N::Int=300, tol::Real=1e-4)
    a_dom, b_dom = -1.0, 1.0

    function smallest_κ(α)
        Vfun = paper_potential(α, β)
        res = compute_w_zeros(a_dom, b_dom, Vfun; N=N, k_max=3.0, imag_tol=0.1)
        # Find the pole closest to the real axis in the lower half-plane
        scat = filter(r -> r.κ < -1e-4, res)
        isempty(scat) && return -10.0  # no resonance found → below threshold
        return maximum(r -> r.κ, scat)  # closest to zero (least negative)
    end

    # At threshold, a resonance pole crosses from κ<0 to κ>0 (becomes bound state)
    # Search for α where the pole is at κ≈0
    α_lo, α_hi = α_range
    κ_lo = smallest_κ(α_lo)
    κ_hi = smallest_κ(α_hi)

    # If both below zero, we need a wider range where one has a bound state
    # Actually: as α increases, poles move up. We look for sign change.
    # Better: just bisect on whether a bound state exists
    function has_bound_state(α)
        Vfun = paper_potential(α, β)
        res = compute_w_zeros(a_dom, b_dom, Vfun; N=N, k_max=3.0, imag_tol=0.1)
        bound = filter(r -> r.κ > 0.01, res)
        return !isempty(bound)
    end

    if has_bound_state(α_lo)
        @warn "α_lo already has a bound state. Lower α_range[1]."
        return α_lo
    end
    if !has_bound_state(α_hi)
        @warn "α_hi has no bound state. Raise α_range[2]."
        return α_hi
    end

    for _ in 1:60
        α_mid = 0.5 * (α_lo + α_hi)
        if has_bound_state(α_mid)
            α_hi = α_mid
        else
            α_lo = α_mid
        end
        (α_hi - α_lo) < tol && break
    end

    return 0.5 * (α_lo + α_hi)
end

# =============================================================================
# PRINTING
# =============================================================================

"""Print formatted scattering data."""
function print_scattering_summary(data; title="Scattering data for H_V")
    println("\n" * "="^60)
    println(title)
    println("="^60)

    println("\n  ● Bound state poles (w(k)=0, Im(k)>0):")
    if isempty(data.bound_poles)
        println("    (none)")
    else
        for r in data.bound_poles
            @printf("    k = %8.5fi   →   E = %9.5f\n", r.κ, r.E)
        end
    end

    println("\n  ● Scattering resonance poles (w(k)=0, Im(k)<0):")
    if isempty(data.scat_poles)
        println("    (none)")
    else
        for r in data.scat_poles
            @printf("    k = %8.5fi   →   E = %9.5f\n", r.κ, r.E)
        end
    end

    println("\n  ✕ Transmission resonances (s₋(k)=0):")
    if isempty(data.trans_res)
        println("    (none)")
    else
        for r in data.trans_res
            @printf("    k = %7.4f + %7.4fi\n", real(r.k), imag(r.k))
        end
    end
    println()
end
