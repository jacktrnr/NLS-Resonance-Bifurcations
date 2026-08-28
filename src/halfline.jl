###############################################
# core.jl - Half-line NLS with ψ(0) = 0
###############################################
#
# Equation: -u'' + V(x)u - u³ = Eu  on x > 0
# BC: ψ(0) = 0
# Shooting parameter: β = ψ'(0)
#
# Match to homoclinic tail: ψ → ±A sech(κ(x - x_R)) as x → ∞
# where A = √(-2E), κ = √(-E)
#
###############################################

using LinearAlgebra, SparseArrays, Arpack
using OrdinaryDiffEq
using OrdinaryDiffEq.SciMLBase: ReturnCode
using BifurcationKit
using Accessors: @optic

# =============================================================================
# GLOBAL HELPERS
# =============================================================================

"""κ(E) = √(-E) for E < 0"""
κ(E) = sqrt(-E)

"""Soliton amplitude A = √(-2E)"""
A_soliton(E) = sqrt(-2E)

# =============================================================================
# SHOOTING FROM ORIGIN
# =============================================================================

"""
    shoot_from_origin(b, E, Vfun, β; N=2000)

Integrate -u'' + V(x)u - u³ = Eu from x=0 to x=b with:
    u(0) = 0
    u'(0) = β

Returns (x, u, v) where v = u'.
"""
function shoot_from_origin(b, E, Vfun, β; N=2000)
    β = abs(β)
    u0 = [0.0, abs(β)]

    function f!(du, u, p, x)
        du[1] = u[2]
        du[2] = (Vfun(x) - E) * u[1] - u[1]^3
    end

    prob = ODEProblem(f!, u0, (0.0, b))
    sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-12,
                saveat=range(0.0, b; length=N+1))

    if sol.retcode != ReturnCode.Success
        return Float64[], Float64[], Float64[]
    end

    x = sol.t
    U = reduce(hcat, sol.u)
    return x, U[1,:], U[2,:]
end

"""
    F_residual(b, E, Vfun, β; N=2000)

Hamiltonian residual at x=b:
    F(β, E) = ½(ψ'(b))² + ½E(ψ(b))² + ¼(ψ(b))⁴

For a valid homoclinic solution, F = 0.
"""
function F_residual(b, E, Vfun, β; N=2000)
    x, u, v = shoot_from_origin(b, E, Vfun, abs(β); N=N)

    if isempty(x)
        return NaN
    end

    ub, vb = u[end], v[end]

    H = 0.5 * vb^2 + 0.5 * E * ub^2 + 0.25 * ub^4

    A = A_soliton(E)
    if abs(ub) > A + 1e-8
        return H/ub + 100.0 * (abs(ub) - A)^2
    end

    return H/ub
end

# =============================================================================
# GLUING TO HOMOCLINIC TAIL
# =============================================================================

"""
    glue_solution(b, E, x, u, v; Xmax=50.0, NR=1000)

Given interior solution (x, u, v) on [0, b], glue the right homoclinic tail.

The tail has form: ψ(x) = ± S(x - x_shift; E)
where S(ξ; E) = √(-2E) sech(√(-E) ξ) is the soliton profile,
± is determined by sign of u(b), and x_shift direction is determined by v(b)/u(b).

Returns (xfull, ψfull) on [0, Xmax].
"""
function glue_solution(b, E, x, u, v; Xmax=50.0, NR=1000)
    if isempty(x) || E >= 0
        return Float64[], Float64[]
    end

    κ = sqrt(-E)
    A = sqrt(-2E)

    ub = u[end]
    vb = v[end]

    if abs(ub) > A + 1e-10
        return Float64[], Float64[]
    end

    s = sign(ub)
    if s == 0
        s = sign(vb)
    end
    if s == 0
        s = 1.0
    end

    if abs(ub) < 1e-14
        if vb > 0
            x_shift = b + 10.0/κ
        else
            x_shift = b - 10.0/κ
        end
    else
        ratio = A / abs(ub)
        ratio = max(ratio, 1.0 + 1e-10)

        y0 = acosh(ratio)

        y_pos = y0
        y_neg = -y0

        deriv_pos = -s * κ * A * sech(y_pos) * tanh(y_pos)
        deriv_neg = -s * κ * A * sech(y_neg) * tanh(y_neg)

        if abs(deriv_pos - vb) < abs(deriv_neg - vb)
            x_shift = b - y0/κ
        else
            x_shift = b + y0/κ
        end
    end

    xR = range(b, Xmax; length=NR)
    ψR = s .* A .* sech.(κ .* (xR .- x_shift))

    xfull = vcat(x, xR[2:end])
    ψfull = vcat(u, ψR[2:end])

    return xfull, ψfull
end

# =============================================================================
# NORMS
# =============================================================================

"""
    compute_L2_norm(b, E, x, u, v; Xmax=50.0)

Compute ∫₀^∞ |ψ|² dx using numerical integration on [0,b]
plus analytic tail integral.
"""
function compute_L2_norm(b, E, x, u, v; Xmax=50.0)
    if isempty(x) || E >= 0
        return NaN
    end

    κv = κ(E)
    A = A_soliton(E)

    ub = u[end]
    vb = v[end]

    if abs(ub) > A + 1e-10
        return NaN
    end

    dx = x[2] - x[1]
    N_bulk = dx * (sum(abs2, u) - 0.5*(u[1]^2 + u[end]^2))

    if abs(ub) < 1e-14
        N_tail = 0.0
    else
        arg = -vb / (κv * ub)
        arg = clamp(arg, -1 + 1e-12, 1 - 1e-12)
        N_tail = (A^2 / κv) * (1 - arg)
    end

    return N_bulk + N_tail
end

"""
    compute_H1_norm(b, E, x, u, v; Xmax=50.0)

Compute ∫₀^∞ (|ψ|² + |ψ'|²) dx.
"""
function compute_H1_norm(b, E, x, u, v; Xmax=50.0)
    if isempty(x) || E >= 0
        return NaN
    end

    κv = κ(E)
    A = A_soliton(E)

    ub = u[end]
    vb = v[end]

    if abs(ub) > A + 1e-10
        return NaN
    end

    dx = x[2] - x[1]

    N_bulk = dx * (sum(abs2, u) - 0.5*(u[1]^2 + u[end]^2))
    D_bulk = dx * (sum(abs2, v) - 0.5*(v[1]^2 + v[end]^2))

    if abs(ub) < 1e-14
        N_tail = 0.0
        D_tail = 0.0
    else
        arg = -vb / (κv * ub)
        arg = clamp(arg, -1 + 1e-12, 1 - 1e-12)

        N_tail = (A^2 / κv) * (1 - arg)
        D_tail = (A^2 * κv / 3) * (1 - arg^3)
    end

    return (N_bulk + N_tail) + (D_bulk + D_tail)
end

# =============================================================================
# SEED FINDING
# =============================================================================

"""
    find_seeds(b, Vfun; E_list, β_max=10.0, nβ=500, N=2000, tol=1e-8)

For each E in E_list, scan β ∈ (0, β_max] for zeros of F(β, E).

Returns vector of named tuples: (; β, E, p=(E=E,))
"""
function find_seeds(b, Vfun; E_list, β_max=10.0, nβ=500, N=2000, tol=1e-8)
    seeds = []

    β_grid = range(1e-4, β_max; length=nβ)

    for E0 in E_list
        println("Searching at E = $E0...")

        F_vals = [F_residual(b, E0, Vfun, β; N=N) for β in β_grid]

        for i in 1:(length(β_grid)-1)
            β1, β2 = β_grid[i], β_grid[i+1]
            F1, F2 = F_vals[i], F_vals[i+1]

            if !isfinite(F1) || !isfinite(F2)
                continue
            end

            found = false
            β_star = NaN

            if F1 * F2 < 0
                β_star = bisect_F(b, E0, Vfun, β1, β2; N=N, tol=tol)
                found = true
            elseif abs(F1) < tol
                β_star = β1
                found = true
            end

            if found && isfinite(β_star)
                is_dup = any(abs(s.β - β_star) < 1e-4 && abs(s.E - E0) < 1e-6
                            for s in seeds)
                if !is_dup
                    push!(seeds, (; β=β_star, E=E0, p=(E=E0,)))
                    println("  Found: β = $(round(β_star, digits=5))")
                end
            end
        end
    end

    println("Total seeds: $(length(seeds))")
    return seeds
end

"""Bisection for F(β, E) = 0"""
function bisect_F(b, E, Vfun, β_lo, β_hi; N=2000, tol=1e-10, maxiter=60)
    F_lo = F_residual(b, E, Vfun, β_lo; N=N)
    F_hi = F_residual(b, E, Vfun, β_hi; N=N)

    if F_lo * F_hi > 0
        return NaN
    end

    for _ in 1:maxiter
        β_mid = 0.5 * (β_lo + β_hi)
        F_mid = F_residual(b, E, Vfun, β_mid; N=N)

        if abs(F_mid) < tol || (β_hi - β_lo) < tol
            return β_mid
        end

        if F_lo * F_mid < 0
            β_hi = β_mid
            F_hi = F_mid
        else
            β_lo = β_mid
            F_lo = F_mid
        end
    end

    return 0.5 * (β_lo + β_hi)
end

# =============================================================================
# CONTINUATION
# =============================================================================

"""
    continue_from_seeds(seeds, b, Vfun; ...)

Continue branches in E from each seed using BifurcationKit.
"""
function continue_from_seeds(seeds, b, Vfun;
                            N=2000,
                            p_min=-10.0,
                            p_max=-1e-10,
                            ds=0.001,
                            dsmin=1e-6,
                            dsmax=0.01,
                            max_steps=1000,
                            β_min=1e-3,
                            verbose=1)

    branches = []

    println("\nContinuing $(length(seeds)) seed(s)...")

    for (idx, seed) in enumerate(seeds)
        verbose > 0 && println("\n--- Seed $idx: E=$(seed.E), β=$(round(seed.β, digits=4)) ---")

        lensE = @optic _.E

        F(β, p) = begin
            E_val = typeof(p) <: Number ? p : p.E
            return [F_residual(b, E_val, Vfun, β[1]; N=N)]
        end

        rec(β, p; k...) = begin
            E_val = typeof(p) <: Number ? p : p.E
            return (; β=β[1], E=E_val)
        end

        prob = BifurcationProblem(F, [seed.β], seed.p, lensE;
                                 record_from_solution=rec)

        opts = ContinuationPar(
            ds=ds,
            dsmin=dsmin,
            dsmax=dsmax,
            p_min=p_min,
            p_max=p_max,
            max_steps=max_steps,
            newton_options=NewtonPar(tol=1e-8, max_iterations=15),
            nev=0,
        )

        last_br = Ref{Any}(nothing)

        finalise_solution = (z, tau, step, contResult; k...) -> begin
            last_br[] = contResult
            β_val = z.u[1]
            step ≤ 1 && return true
            return abs(β_val) > β_min
        end

        br = try
            continuation(prob, PALC(), opts;
                        bothside=true,
                        verbosity=verbose,
                        finalise_solution=finalise_solution)
        catch e
            @warn "Continuation failed for seed $idx: $e"
            last_br[]
        end

        if br !== nothing && !isempty(br.branch)
            n_pts = length(br.branch)
            Es = [sol.param for sol in br.branch]
            verbose > 0 && println("✓ Branch: $n_pts points, E ∈ [$(round(minimum(Es),digits=3)), $(round(maximum(Es),digits=3))]")
        else
            verbose > 0 && println("✗ Failed")
            br = (; branch=[])
        end

        push!(branches, br)
    end

    return branches
end

# =============================================================================
# LINEAR STABILITY OPERATORS
# =============================================================================

"""Simple piecewise-linear interpolation"""
function linear_interp(xsrc::AbstractVector,
                       ysrc::AbstractVector,
                       xtarget::AbstractVector)
    n = length(xtarget)
    y = similar(xtarget)
    j = 1
    for i in 1:n
        x = xtarget[i]
        if x <= xsrc[1]
            y[i] = ysrc[1]
            continue
        elseif x >= xsrc[end]
            y[i] = ysrc[end]
            continue
        end
        while xsrc[j+1] < x
            j += 1
        end
        t = (x - xsrc[j]) / (xsrc[j+1] - xsrc[j])
        y[i] = (1 - t) * ysrc[j] + t * ysrc[j+1]
    end
    return y
end

"""
    compute_Lpm_eigenvalues(b, E, Vfun, β; nev=3, Ngrid=4000, Xmax=150.0)

Compute the smallest `nev` eigenvalues of L₊ and L₋
using second-order finite differences with Dirichlet BCs.
"""
function compute_Lpm_eigenvalues(b, E, Vfun, β;
                                 nev=3,
                                 Ngrid=12000,
                                 Xmax=150.0)

    x = range(0.0, Xmax; length=Ngrid)
    h = x[2] - x[1]

    x_int, u_int, v_int = shoot_from_origin(b, E, Vfun, β; N=2000)
    xfull, ψfull = glue_solution(b, E, x_int, u_int, v_int; Xmax=Xmax)
    ψ = linear_interp(xfull, ψfull, x)

    xi = x[2:end-1]
    ψi = ψ[2:end-1]
    n  = length(xi)

    main = fill(2.0 / h^2, n)
    off  = fill(-1.0 / h^2, n-1)
    Δ = Tridiagonal(off, main, off)

    V = Vfun.(xi)
    Vplus  = V .- E .- 3.0 .* ψi.^2
    Vminus = V .- E .- 1.0 .* ψi.^2

    Lplus  = Δ + Diagonal(Vplus)
    Lminus = Δ + Diagonal(Vminus)

    λ_plus  = sort(eigvals(Lplus))[1:nev]
    λ_minus = sort(eigvals(Lminus))[1:nev]

    return λ_plus, λ_minus
end

"""Compute eigenfunctions of L₊ and L₋"""
function compute_Lpm_eigenfunctions(b, E, Vfun, β;
                                    nev=3,
                                    Ngrid=12000,
                                    Xmax=150.0)
    x = range(0.0, Xmax; length=Ngrid)
    h = x[2] - x[1]

    x_int, u_int, v_int = shoot_from_origin(b, E, Vfun, β; N=2000)
    xfull, ψfull = glue_solution(b, E, x_int, u_int, v_int; Xmax=Xmax)
    ψ = linear_interp(xfull, ψfull, x)

    xi = x[2:end-1]
    ψi = ψ[2:end-1]
    n  = length(xi)

    main = fill(2.0 / h^2, n)
    off  = fill(-1.0 / h^2, n-1)
    Δ = Tridiagonal(off, main, off)

    V = Vfun.(xi)
    Lplus  = Δ + Diagonal(V .- E .- 3.0 .* ψi.^2)
    Lminus = Δ + Diagonal(V .- E .- 1.0 .* ψi.^2)

    Fp = eigen(Symmetric(Matrix(Lplus)))
    Fm = eigen(Symmetric(Matrix(Lminus)))

    idx_p = sortperm(Fp.values)[1:nev]
    idx_m = sortperm(Fm.values)[1:nev]

    return (; x=collect(xi), ψ=ψi,
             λ_plus=Fp.values[idx_p], ϕ_plus=Fp.vectors[:, idx_p],
             λ_minus=Fm.values[idx_m], ϕ_minus=Fm.vectors[:, idx_m])
end

"""Track spectrum along a branch on a grid of `n_grid` E values."""
function track_spectrum_branch(branch, b, Vfun;
                               nev=2,
                               n_grid=50,
                               Ngrid=12000,
                               Xmax=150.0)

    isempty(branch.branch) &&
        return (; Es=Float64[], evals_plus=zeros(0,0), evals_minus=zeros(0,0))

    # Build a uniform grid of E values spanning the branch, then pick the
    # nearest branch point for each.  Always include the first and last
    # branch points (these are the "endpoints of the E data").
    all_Es = [sol.param for sol in branch.branch]
    E_lo, E_hi = extrema(all_Es)

    E_grid = range(E_lo, E_hi; length=max(n_grid, 2))

    # For each grid E, find the branch index with the closest E value
    indices = Set{Int}()
    push!(indices, 1)                        # first branch point
    push!(indices, length(branch.branch))    # last branch point
    for Eg in E_grid
        _, idx = findmin(abs.(all_Es .- Eg))
        push!(indices, idx)
    end
    indices = sort(collect(indices))
    sols = [branch.branch[i] for i in indices]

    Es = Float64[]
    evals_plus  = fill(NaN, nev, length(sols))
    evals_minus = fill(NaN, nev, length(sols))

    println("Tracking spectrum on $(length(sols)) points (FD, tridiagonal)...")

    for (k, sol) in enumerate(sols)
        E = sol.param
        β = sol.β
        push!(Es, E)

        try
            λp, λm = compute_Lpm_eigenvalues(
                b, E, Vfun, β;
                nev=nev,
                Ngrid=Ngrid,
                Xmax=Xmax
            )

            evals_plus[:, k]  .= λp
            evals_minus[:, k] .= λm

            if k == 1 || k % 5 == 0 || k == length(sols)
                println("  [$k/$(length(sols))] E=$(round(E,digits=6)) ",
                        "λ₀⁺=$(round(λp[1],digits=8)) ",
                        "λ₀⁻=$(round(λm[1],digits=8)) ",
                        "Ngrid=$Ngrid Xmax=$Xmax")
            end

        catch e
            @warn "Spectrum computation failed at E=$E" exception=e
        end
    end

    return (; Es=Es, evals_plus=evals_plus, evals_minus=evals_minus)
end

"""Compatibility stub for building L₊/L₋ operators"""
function build_Lpm_operators(b, E, Vfun, β;
                             Ngrid=12000,
                             Xmax=150.0)

    x = range(0.0, Xmax; length=Ngrid)
    x_int, u_int, v_int = shoot_from_origin(b, E, Vfun, β; N=2000)
    xfull, ψfull = glue_solution(b, E, x_int, u_int, v_int; Xmax=Xmax)
    ψ = linear_interp(xfull, ψfull, x)

    return x[2:end-1], ψ[2:end-1], nothing, nothing
end
