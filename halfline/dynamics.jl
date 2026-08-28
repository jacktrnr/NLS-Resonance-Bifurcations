###############################################
# dynamics.jl - Time dynamics via split-step
###############################################
#
# Evolves the time-dependent NLS on the half-line:
#   iψₜ = -ψ'' + V(x)ψ - |ψ|²ψ,   x > 0
#   ψ(0,t) = 0                       (Dirichlet BC)
#
# Uses a split-step method with the Discrete Sine Transform (DST)
# to handle the kinetic operator with Dirichlet BCs at both ends.
#
# The initial condition is a bound state ψ₀(x) plus a perturbation:
#   ψ(x,0) = (1 + ε₁) ψ₀(x) + ε₂ h(x)
# where h(x) is a derivative-of-Gaussian with a tail correction
# to satisfy h(0) = 0:
#   g(x)  = exp(-(x - x_c)² / 2σ²)        (Gaussian centered at peak of ψ₀)
#   g'(x) = -(x - x_c)/σ² · g(x)
#   h(x)  = g'(x) - g'(0) · exp(-κx)       (κ = √(-E), soliton decay rate)
#
###############################################

using FFTW
using Plots
using LaTeXStrings

# =============================================================================
# SPLIT-STEP INTEGRATOR
# =============================================================================

"""
    splitstep_evolve(ψ0, x, Vx, dt, Nt; save_every=1)

Evolve iψₜ = -ψ'' + V(x)ψ - |ψ|²ψ on an interior grid `x` (Dirichlet at
both endpoints) using symmetric split-step with DST.

`ψ0` and `Vx` are vectors on the interior grid (excluding boundary zeros).
Returns `(t_saves, ψ_saves)` where `ψ_saves[k]` is the wavefunction at `t_saves[k]`.
"""
function splitstep_evolve(ψ0::Vector{ComplexF64}, x, Vx, dt, Nt; save_every=10)
    n = length(x)
    dx = x[2] - x[1]
    L = x[end] + dx  # effective domain length (grid goes from dx to L-dx)

    # DST-I eigenvalues for -d²/dx² with Dirichlet BCs:
    #   k_m = mπ/L,  eigenvalue = k_m²,   m = 1, ..., n
    k2 = [(m * π / L)^2 for m in 1:n]

    # Kinetic propagator in sine space (full step)
    kinetic_full = exp.(-im .* k2 .* dt)

    ψ = copy(ψ0)

    t_saves = Float64[]
    ψ_saves = Vector{ComplexF64}[]

    push!(t_saves, 0.0)
    push!(ψ_saves, copy(ψ))

    for step in 1:Nt
        # --- Half-step potential + nonlinear (real space) ---
        @. ψ = ψ * exp(-im * (Vx - abs2(ψ)) * dt / 2)

        # --- Full-step kinetic (sine space) ---
        ψ_hat = FFTW.r2r(real.(ψ), FFTW.RODFT00) .+ im .* FFTW.r2r(imag.(ψ), FFTW.RODFT00)
        @. ψ_hat = ψ_hat * kinetic_full
        # Inverse DST-I: same transform, scaled by 1/(2(n+1))
        scale = 1.0 / (2 * (n + 1))
        ψ = scale .* (FFTW.r2r(real.(ψ_hat), FFTW.RODFT00) .+ im .* FFTW.r2r(imag.(ψ_hat), FFTW.RODFT00))

        # --- Half-step potential + nonlinear (real space) ---
        @. ψ = ψ * exp(-im * (Vx - abs2(ψ)) * dt / 2)

        if step % save_every == 0 || step == Nt
            push!(t_saves, step * dt)
            push!(ψ_saves, copy(ψ))
        end
    end

    return t_saves, ψ_saves
end

# =============================================================================
# MAIN DRIVER
# =============================================================================

"""
    run_dynamics(seeds, branches, b, Vfun;
                 use_endpoint=false,
                 Xmax=30.0, Ngrid=2048, N_ode=3000,
                 Tmax=50.0, dt=1e-3,
                 ε_scale=0.05, ε_pert=0.05, σ_pert=1.0,
                 save_every=50, fps=20,
                 results_dir="results", label="dynamics")

Run time dynamics for a perturbed bound state.

- `use_endpoint=false`: use the branch point near Estart (first point on branch).
- `use_endpoint=true`:  use the branch endpoint whose E is closest to 0.

Initial condition:
    ψ(x,0) = (1 + ε_scale) ψ₀(x)  +  ε_pert · h(x)
where h(x) is a derivative-of-Gaussian (centered at the peak of ψ₀, width σ_pert)
with a tail correction to enforce h(0) = 0:
    g(x)  = exp(-(x - x_c)² / 2σ²)
    g'(x) = -(x - x_c)/σ² · g(x)
    h(x)  = g'(x) - g'(0) · exp(-κx),   κ = √(-E)
"""
function run_dynamics(seeds, branches, b, Vfun;
                      use_endpoint=false,
                      Xmax=30.0, Ngrid=2048, N_ode=3000,
                      Tmax=50.0, dt=1e-3,
                      ε_scale=0.05, ε_pert=0.05, σ_pert=1.0,
                      save_every=50, fps=20,
                      results_dir="results", label="dynamics")

    # --- Find the bound state on the first valid branch ---
    E0 = nothing
    β0 = nothing
    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue

        if use_endpoint
            # Pick the branch point whose E is closest to 0
            _, idx = findmin(abs(sol.param) for sol in br.branch)
            sol = br.branch[idx]
            println("  Using Branch $i endpoint closest to E=0: " *
                    "E = $(sol.param), β = $(round(sol.β, digits=5))")
        else
            # Use the first branch point (near Estart)
            sol = br.branch[1]
            println("  Using Branch $i starting point: " *
                    "E = $(sol.param), β = $(round(sol.β, digits=5))")
        end

        E0 = sol.param
        β0 = sol.β
        break
    end

    if E0 === nothing
        @warn "No valid branch found — skipping dynamics."
        return nothing
    end

    # --- Build the bound state on a uniform grid ---
    x_ode, u_ode, v_ode = shoot_from_origin(b, E0, Vfun, β0; N=N_ode)
    if isempty(x_ode)
        @warn "Shooting failed for dynamics initial condition."
        return nothing
    end

    xfull, ψfull = glue_solution(b, E0, x_ode, u_ode, v_ode; Xmax=Xmax)
    if isempty(xfull)
        @warn "Gluing failed for dynamics initial condition."
        return nothing
    end

    # Uniform interior grid (excludes x=0 and x=Xmax where ψ=0)
    dx = Xmax / (Ngrid + 1)
    x_grid = [j * dx for j in 1:Ngrid]

    # Interpolate bound state onto the grid
    ψ_bound = linear_interp(xfull, ψfull, x_grid)
    Vx = [Vfun(xi) for xi in x_grid]

    # --- Build perturbation h(x): derivative-of-Gaussian + tail correction ---
    # Center the Gaussian at the peak of |ψ₀|
    _, i_peak = findmax(abs.(ψ_bound))
    x_c = x_grid[i_peak]
    σ = σ_pert
    κv = sqrt(-E0)   # soliton decay rate

    # g'(x) = -(x - x_c)/σ² · exp(-(x - x_c)²/(2σ²))
    gprime = [-(xi - x_c) / σ^2 * exp(-(xi - x_c)^2 / (2σ^2)) for xi in x_grid]
    gprime_0 = x_c / σ^2 * exp(-x_c^2 / (2σ^2))   # = -g'(0) evaluated analytically

    # h(x) = g'(x) - g'(0) · exp(-κx)   so that h(0) = 0
    h_pert = [gprime[j] - (-gprime_0) * exp(-κv * x_grid[j]) for j in 1:Ngrid]
    # Note: g'(0) = -gprime_0 since gprime = -(x-x_c)/σ² · g  →  at x=0: -(-x_c)/σ² · g(0) = x_c/σ² · g(0)
    # Correction: g'(0) = gprime evaluated at x=0, but x_grid starts at dx, not 0.
    # Use the analytic value: g'(0) = x_c/σ² · exp(-x_c²/(2σ²))
    gprime_at_0 = x_c / σ^2 * exp(-x_c^2 / (2σ^2))
    h_pert = [gprime[j] - gprime_at_0 * exp(-κv * x_grid[j]) for j in 1:Ngrid]

    # Normalize h so that ε_pert controls the amplitude relative to ψ₀
    h_max = maximum(abs, h_pert)
    if h_max > 0
        h_pert .*= maximum(abs, ψ_bound) / h_max
    end

    # --- Perturb ---
    ψ_init = ComplexF64.((1.0 + ε_scale) .* ψ_bound .+ ε_pert .* h_pert)

    N_mass0 = dx * sum(abs2, ψ_init)
    println("  Grid: $Ngrid interior points, dx = $(round(dx, digits=5))")
    println("  Tmax = $Tmax, dt = $dt, $(round(Int, Tmax/dt)) steps")
    println("  Perturbation: ε_scale = $ε_scale, ε_pert = $ε_pert, σ_pert = $σ_pert")
    println("  Gaussian peak at x_c = $(round(x_c, digits=3)), κ = $(round(κv, digits=5))")
    @printf("  Initial mass: %.6f\n", N_mass0)

    # --- Evolve ---
    Nt = round(Int, Tmax / dt)
    t_saves, ψ_saves = splitstep_evolve(ψ_init, x_grid, Vx, dt, Nt;
                                         save_every=save_every)

    println("  Evolution complete: $(length(t_saves)) frames saved.")

    # --- Check mass conservation ---
    N_final = dx * sum(abs2, ψ_saves[end])
    @printf("  Final mass:   %.6f  (drift = %.2e)\n", N_final, abs(N_final - N_mass0))

    # --- Build GIF ---
    println("  Generating GIF...")
    ymax = 1.3 * maximum(maximum(abs.(ψ)) for ψ in ψ_saves)

    anim = @animate for (k, ψk) in enumerate(ψ_saves)
        t = t_saves[k]
        Nk = dx * sum(abs2, ψk)

        plot(x_grid, abs.(ψk);
             color=:blue, lw=2,
             xlabel=L"x", ylabel=L"|\psi(x,t)|",
             title=@sprintf("t = %.2f,  N = %.6f", t, Nk),
             ylims=(0, ymax),
             xlims=(0, min(Xmax, 3*b + 15)),
             legend=false, size=(600, 350))

        # Show bound state for reference
        plot!(x_grid, abs.(ψ_bound);
              color=:gray, lw=1.5, ls=:dash, alpha=0.5)

        vline!([b]; color=:gray40, ls=:dash, lw=1, alpha=0.4)
    end

    gif_dir = joinpath(results_dir, "dynamics")
    mkpath(gif_dir)
    gif_path = joinpath(gif_dir, "$(label).gif")
    gif(anim, gif_path; fps=fps)

    println("  GIF saved to: $gif_path")
    return gif_path
end
