##################
# spectral.jl   #
##################

# Linear stability operators L+ and L- for full-line NLS solitons.
# Adapted from Dec 2025 half-line code to the full-line domain [-Xmax, Xmax].

"""
    linear_interp(xsrc, ysrc, xtarget)

Simple piecewise-linear interpolation of (xsrc, ysrc) onto xtarget grid.
"""
function linear_interp(xsrc::AbstractVector,
                       ysrc::AbstractVector,
                       xtarget::AbstractVector)
    n = length(xtarget)
    y = similar(xtarget, eltype(ysrc))
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
        while j < length(xsrc) - 1 && xsrc[j+1] < x
            j += 1
        end
        t = (x - xsrc[j]) / (xsrc[j+1] - xsrc[j])
        y[i] = (1 - t) * ysrc[j] + t * ysrc[j+1]
    end
    return y
end

"""
    compute_Lpm_eigenvalues(a, b, E, Vfun, c, slope_sign;
                            nev=3, Ngrid=6000, Xmax=50.0)

Compute the smallest `nev` eigenvalues of L+ and L- on the full line [-Xmax, Xmax]
using second-order finite differences with Dirichlet BCs at both ends.

L+ = -d²/dx² + (V(x) - E) - 3 psi0²
L- = -d²/dx² + (V(x) - E) -   psi0²

Returns `(lambda_plus, lambda_minus)` each a Vector of length `nev`.
"""
function compute_Lpm_eigenvalues(a, b, E, Vfun, c, slope_sign;
                                 nev=3,
                                 Ngrid=6000,
                                 Xmax=50.0)

    # FD grid on [-Xmax, Xmax]
    x = range(-Xmax, Xmax; length=Ngrid)
    h = x[2] - x[1]

    # Build the glued soliton profile on [a,b] + tails
    x_int, u_int, v_int = integrate_support(a, b, E, Vfun;
                                            N=7000, c=c, slope_sign=slope_sign)
    if isempty(x_int)
        return fill(NaN, nev), fill(NaN, nev)
    end

    xfull, psifull, _ = glue_full_solution(a, b, E, x_int, u_int, v_int; Xmax=Xmax)
    if isempty(xfull)
        return fill(NaN, nev), fill(NaN, nev)
    end

    # Interpolate onto FD grid
    psi = linear_interp(collect(Float64, real.(xfull)),
                        collect(Float64, real.(psifull)),
                        collect(x))

    # Interior points (Dirichlet BCs: psi=0 at endpoints)
    xi  = x[2:end-1]
    psi_i = psi[2:end-1]
    n   = length(xi)

    # Tridiagonal Laplacian: -d²/dx²
    main = fill(2.0 / h^2, n)
    off  = fill(-1.0 / h^2, n - 1)
    Delta = Tridiagonal(off, main, off)

    # Potential diagonal
    V = [Vfun(xi_k) for xi_k in xi]
    Vplus  = V .- E .- 3.0 .* psi_i.^2
    Vminus = V .- E .- 1.0 .* psi_i.^2

    Lplus  = Delta + Diagonal(Vplus)
    Lminus = Delta + Diagonal(Vminus)

    # Full diagonalization (dense) — fine for Ngrid up to ~10k
    lambda_plus  = sort(eigvals(Symmetric(Matrix(Lplus))))[1:nev]
    lambda_minus = sort(eigvals(Symmetric(Matrix(Lminus))))[1:nev]

    return lambda_plus, lambda_minus
end

"""
    track_spectrum_branch(branch, a, b, Vfun;
                          nev=2, n_grid=50, Ngrid_FD=6000, Xmax=50.0)

Track eigenvalues of L+ and L- along an entire continuation branch.

Samples the branch at `n_grid` uniformly spaced E values (always including
the first and last branch points), computing the spectrum at each.

Returns `(; Es, evals_plus, evals_minus)` where `evals_plus[j,k]` is the
j-th eigenvalue at the k-th sample point.
"""
function track_spectrum_branch(branch, a, b, Vfun;
                               nev=2,
                               n_grid=50,
                               Ngrid_FD=6000,
                               Xmax=50.0)

    isempty(branch.branch) &&
        return (; Es=Float64[], evals_plus=zeros(0, 0), evals_minus=zeros(0, 0))

    # Build uniform E grid spanning the branch
    all_Es = [sol.param for sol in branch.branch]
    E_lo, E_hi = extrema(all_Es)

    E_grid = range(E_lo, E_hi; length=max(n_grid, 2))

    # For each grid E, find the nearest branch point
    indices = Set{Int}()
    push!(indices, 1)
    push!(indices, length(branch.branch))
    for Eg in E_grid
        _, idx = findmin(abs.(all_Es .- Eg))
        push!(indices, idx)
    end
    indices = sort(collect(indices))
    sols = [branch.branch[i] for i in indices]

    Es = Float64[]
    evals_plus  = fill(NaN, nev, length(sols))
    evals_minus = fill(NaN, nev, length(sols))

    println("Tracking spectrum on $(length(sols)) points (FD grid=$Ngrid_FD, Xmax=$Xmax)...")

    for (k, sol) in enumerate(sols)
        E = sol.param
        c = sol.c
        ss = sol.slope_sign
        push!(Es, E)

        try
            lp, lm = compute_Lpm_eigenvalues(a, b, E, Vfun, c, ss;
                                              nev=nev, Ngrid=Ngrid_FD, Xmax=Xmax)
            evals_plus[:, k]  .= lp
            evals_minus[:, k] .= lm

            if k == 1 || k % 5 == 0 || k == length(sols)
                println("  [$k/$(length(sols))] E=$(round(E, digits=6)) " *
                        "lam0+=$(round(lp[1], digits=8)) " *
                        "lam0-=$(round(lm[1], digits=8))")
            end
        catch e
            @warn "Spectrum computation failed at E=$E" exception=e
        end
    end

    return (; Es=Es, evals_plus=evals_plus, evals_minus=evals_minus)
end
