###############################################
# dynamics.jl - Time dynamics via split-step
###############################################
#
# Evolves the time-dependent NLS on the full line:
#   iψₜ = -ψ'' + V(x)ψ - |ψ|²ψ,   x ∈ (-Xmax, Xmax)
#   ψ(-Xmax,t) = ψ(Xmax,t) = 0      (Dirichlet BCs)
#
# Uses a symmetric split-step method with the Discrete Sine Transform (DST-I)
# to handle the kinetic operator.
#
# Optional complex absorbing potential (CAP) near ±Xmax to simulate
# radiating BCs: outgoing waves are damped instead of reflected.
#
# Three initial condition options:
#   :soliton     — perturbed bound state from continuation branch
#   :groundstate — ground eigenfunction of -∂²/∂x² on [a,b], scaled to N[Emin]
#   :gaussian    — A·exp(-x²/(2σ²)), scaled to N[Emin]
#
###############################################

using FFTW
using Printf

# =============================================================================
# ABSORBING LAYER
# =============================================================================

"""
    build_absorbing_layer(x_grid, Xmax, width, strength; power=3)

Build a smooth absorbing potential W(x) ≥ 0 that ramps up near ±Xmax:

    W(x) = W0 · ((|x| - x_abs)/(Xmax - x_abs))^p   for |x| > x_abs
    W(x) = 0                                          otherwise

where x_abs = Xmax - width.

The effective complex potential is V(x) - i·W(x). In the split-step,
this produces damping exp(-W·dt/2) at each half-step, absorbing outgoing
radiation instead of reflecting it off the hard walls.
"""
function build_absorbing_layer(x_grid, Xmax, width, strength; power=3)
    x_abs = Xmax - width
    W = zeros(length(x_grid))
    for (j, x) in enumerate(x_grid)
        ax = abs(x)
        if ax > x_abs
            W[j] = strength * ((ax - x_abs) / width)^power
        end
    end
    return W
end

# =============================================================================
# SPLIT-STEP INTEGRATOR
# =============================================================================

"""
    splitstep_evolve(ψ0, x, Vx, dt, Nt; save_every=10)

Evolve iψₜ = -ψ'' + V(x)ψ - |ψ|²ψ on an interior grid `x` (Dirichlet at
both endpoints) using symmetric split-step with DST-I.

`ψ0` and `Vx` are vectors on the interior grid (excluding boundary zeros).
`Vx` may be complex: Im(Vx) = -W(x) provides absorbing-layer damping.
Domain is [-Xmax, Xmax]; effective length L = (n+1)*dx where n = length(x).

Returns `(t_saves, ψ_saves)` where `ψ_saves[k]` is the wavefunction at `t_saves[k]`.
"""
function splitstep_evolve(ψ0::Vector{ComplexF64}, x, Vx, dt, Nt; save_every=10)
    n = length(x)
    dx = x[2] - x[1]
    L = (n + 1) * dx   # full domain length = 2*Xmax

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

        # --- Full-step kinetic (sine space via DST-I) ---
        ψ_hat = FFTW.r2r(real.(ψ), FFTW.RODFT00) .+
                 im .* FFTW.r2r(imag.(ψ), FFTW.RODFT00)
        @. ψ_hat = ψ_hat * kinetic_full
        # Inverse DST-I: same transform, scaled by 1/(2(n+1))
        scale = 1.0 / (2 * (n + 1))
        ψ = scale .* (FFTW.r2r(real.(ψ_hat), FFTW.RODFT00) .+
                       im .* FFTW.r2r(imag.(ψ_hat), FFTW.RODFT00))

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
# INITIAL CONDITIONS
# =============================================================================

"""
    build_ic_groundstate(a, b, Nmass, x_grid)

Ground state of -d²/dx² on [a,b]: φ(x) = sin(π(x-a)/(b-a)) for x∈[a,b], 0 outside.
Scaled so that dx·Σ|φ|² = Nmass.
"""
function build_ic_groundstate(a, b, Nmass, x_grid)
    dx = x_grid[2] - x_grid[1]
    φ = [a ≤ x ≤ b ? sin(π * (x - a) / (b - a)) : 0.0 for x in x_grid]
    raw_mass = dx * sum(abs2, φ)
    if raw_mass > 0
        φ .*= sqrt(Nmass / raw_mass)
    end
    return ComplexF64.(φ)
end

"""
    build_ic_perturbed_soliton(a, b, E, Vfun, c, slope_sign, x_grid, Xmax;
                                ε_scale=0.05, ε_pert=0.05, σ_pert=1.0, N_ode=7000)

Build perturbed bound state IC:
    ψ(x,0) = (1+ε_scale)·ψ₀(x) + ε_pert·h(x)

where ψ₀ is the glued soliton and h(x) is a derivative-of-Gaussian centered at
the peak of |ψ₀|, normalized to max(|ψ₀|).
"""
function build_ic_perturbed_soliton(a, b, E, Vfun, c, slope_sign, x_grid, Xmax;
                                     ε_scale=0.05, ε_pert=0.05, σ_pert=1.0,
                                     N_ode=7000)
    # Build the bound state via integrate_support + glue
    x_int, u_int, v_int = integrate_support(a, b, E, Vfun;
                                             N=N_ode, c=c, slope_sign=slope_sign)
    if isempty(x_int)
        @warn "Integration failed for soliton IC at E=$E"
        return nothing
    end

    xfull, psifull, _ = glue_full_solution(a, b, E, x_int, u_int, v_int; Xmax=Xmax)
    if isempty(xfull)
        @warn "Gluing failed for soliton IC at E=$E"
        return nothing
    end

    # Interpolate onto the dynamics grid
    ψ_bound = linear_interp(collect(Float64, real.(xfull)),
                             collect(Float64, real.(psifull)),
                             collect(x_grid))

    # Build perturbation h(x): derivative-of-Gaussian centered at peak of |ψ₀|
    _, i_peak = findmax(abs.(ψ_bound))
    x_c = x_grid[i_peak]
    σ = σ_pert

    # g'(x) = -(x - x_c)/σ² · exp(-(x - x_c)²/(2σ²))
    gprime = [-(xi - x_c) / σ^2 * exp(-(xi - x_c)^2 / (2σ^2)) for xi in x_grid]

    # Normalize h so that ε_pert controls amplitude relative to ψ₀
    h_max = maximum(abs, gprime)
    if h_max > 0
        gprime .*= maximum(abs, ψ_bound) / h_max
    end

    ψ_init = ComplexF64.((1.0 + ε_scale) .* ψ_bound .+ ε_pert .* gprime)
    return ψ_init, ψ_bound
end

"""
    build_ic_gaussian(σ, Nmass, x_grid)

Gaussian IC: φ(x) = exp(-x²/(2σ²)), scaled so dx·Σ|φ|² = Nmass.
"""
function build_ic_gaussian(σ, Nmass, x_grid)
    dx = x_grid[2] - x_grid[1]
    φ = [exp(-x^2 / (2σ^2)) for x in x_grid]
    raw_mass = dx * sum(abs2, φ)
    if raw_mass > 0
        φ .*= sqrt(Nmass / raw_mass)
    end
    return ComplexF64.(φ)
end

# =============================================================================
# HELPERS
# =============================================================================

"""
    get_branch_Emin_mass(br, a, b, Vfun, Ngrid)

Find the solution on `br` with most negative E.
Returns `(Emin, Nmass, sol_at_Emin)`.
"""
function get_branch_Emin_mass(br, a, b, Vfun, Ngrid)
    isempty(br.branch) && return (NaN, NaN, nothing)

    all_Es = [sol.param for sol in br.branch]
    idx_min = argmin(all_Es)
    sol = br.branch[idx_min]

    E = sol.param
    c = sol.c
    ss = sol.slope_sign

    x, u, v = integrate_support(a, b, E, Vfun; N=Ngrid, c=c, slope_sign=ss)
    if isempty(x)
        return (E, NaN, sol)
    end

    Nmass = compute_norm(a, b, E, x, u, v)
    return (E, Nmass, sol)
end

# =============================================================================
# MAIN DRIVER
# =============================================================================

"""
    run_dynamics(branches, a, b, Vfun;
                 ic=:soliton, branch_idx=1,
                 Xmax=50.0, Ngrid=2048, Tmax=50.0, dt=1e-3,
                 ε_scale=0.05, ε_pert=0.05, σ_pert=1.0, σ_gauss=2.0,
                 save_every=100, fps=20, N_ode=7000,
                 absorb_width=0.0, absorb_strength=5.0, absorb_power=3,
                 results_dir="results", label="dynamics")

Run time dynamics for the NLS on [-Xmax, Xmax].

IC options:
- `:soliton`     — perturbed bound state from branch
- `:groundstate` — ground eigenfunction of -∂² on [a,b], scaled to N[Emin]
- `:gaussian`    — Gaussian, scaled to N[Emin]

Absorbing BCs:
- `absorb_width > 0` enables a complex absorbing potential (CAP) in the
  outermost `absorb_width` of each side. Outgoing radiation is damped
  rather than reflected. Mass will decrease as energy radiates away.
- `absorb_strength` controls the damping rate (typical: 1–10).
- `absorb_power` controls the ramp profile (3 = cubic, smooth turn-on).
"""
function run_dynamics(branches, a, b, Vfun;
                      ic=:soliton, branch_idx=1,
                      Xmax=50.0, Ngrid=2048, Tmax=50.0, dt=1e-3,
                      ε_scale=0.05, ε_pert=0.05, σ_pert=1.0, σ_gauss=2.0,
                      save_every=100, fps=20, N_ode=7000,
                      absorb_width=0.0, absorb_strength=5.0, absorb_power=3,
                      results_dir="results", label="dynamics")

    println("\n" * "="^70)
    println("TIME DYNAMICS  (ic = $ic)")
    println("="^70)

    use_absorbing = absorb_width > 0

    # --- Pick branch ---
    br = nothing
    br_idx_used = 0
    if branch_idx ≤ length(branches) && !isempty(branches[branch_idx].branch)
        br = branches[branch_idx]
        br_idx_used = branch_idx
    else
        for (i, b_) in enumerate(branches)
            if !isempty(b_.branch)
                br = b_
                br_idx_used = i
                break
            end
        end
    end

    if br === nothing
        @warn "No valid branch found — skipping dynamics."
        return nothing
    end
    println("  Using Branch $br_idx_used ($(length(br.branch)) points)")

    # --- Build interior grid on [-Xmax, Xmax] ---
    dx = 2Xmax / (Ngrid + 1)
    x_grid = [-Xmax + j * dx for j in 1:Ngrid]

    # Physical potential (real)
    Vx_real = [Vfun(xi) for xi in x_grid]

    # Add absorbing layer as imaginary part: V_eff = V(x) - i·W(x)
    if use_absorbing
        W = build_absorbing_layer(x_grid, Xmax, absorb_width, absorb_strength;
                                   power=absorb_power)
        Vx = ComplexF64.(Vx_real) .- im .* W
        println("  Absorbing layer: width=$absorb_width, strength=$absorb_strength, power=$absorb_power")
    else
        Vx = ComplexF64.(Vx_real)
    end

    # --- Build initial condition ---
    ψ_bound = nothing   # reference profile for plotting (may be nothing)

    if ic == :soliton
        # Use the first branch point (near E_start)
        sol = br.branch[1]
        E0 = sol.param
        c0 = sol.c
        ss0 = sol.slope_sign
        println("  Soliton IC from E = $E0, c = $(round(c0, digits=5)), slope = $ss0")

        result = build_ic_perturbed_soliton(a, b, E0, Vfun, c0, ss0, x_grid, Xmax;
                                             ε_scale=ε_scale, ε_pert=ε_pert,
                                             σ_pert=σ_pert, N_ode=N_ode)
        if result === nothing
            @warn "Failed to build soliton IC."
            return nothing
        end
        ψ_init, ψ_bound = result

    elseif ic == :groundstate
        Emin, Nmass, _ = get_branch_Emin_mass(br, a, b, Vfun, N_ode)
        if !isfinite(Nmass)
            @warn "Could not compute target mass from branch."
            return nothing
        end
        println("  Ground state IC: target mass N = $(round(Nmass, digits=6)) (from E = $(round(Emin, digits=6)))")
        ψ_init = build_ic_groundstate(a, b, Nmass, x_grid)

    elseif ic == :gaussian
        Emin, Nmass, _ = get_branch_Emin_mass(br, a, b, Vfun, N_ode)
        if !isfinite(Nmass)
            @warn "Could not compute target mass from branch."
            return nothing
        end
        println("  Gaussian IC: σ = $σ_gauss, target mass N = $(round(Nmass, digits=6)) (from E = $(round(Emin, digits=6)))")
        ψ_init = build_ic_gaussian(σ_gauss, Nmass, x_grid)

    else
        error("Unknown IC type: $ic. Use :soliton, :groundstate, or :gaussian.")
    end

    N_mass0 = dx * sum(abs2, ψ_init)
    Nt = round(Int, Tmax / dt)
    println("  Grid: $Ngrid interior points, dx = $(round(dx, digits=5))")
    println("  Tmax = $Tmax, dt = $dt, $Nt steps")
    if use_absorbing
        println("  BCs: absorbing (radiation escapes)")
    else
        println("  BCs: hard Dirichlet (reflecting walls)")
    end
    @printf("  Initial mass: %.8f\n", N_mass0)

    # --- Evolve ---
    t_saves, ψ_saves = splitstep_evolve(ψ_init, x_grid, Vx, dt, Nt;
                                         save_every=save_every)

    println("  Evolution complete: $(length(t_saves)) frames saved.")

    # --- Mass diagnostics ---
    N_final = dx * sum(abs2, ψ_saves[end])
    if use_absorbing
        @printf("  Final mass:   %.8f  (%.1f%% absorbed)\n",
                N_final, 100 * (1 - N_final / N_mass0))
    else
        N_max_drift = maximum(abs(dx * sum(abs2, ψk) - N_mass0) for ψk in ψ_saves)
        @printf("  Final mass:   %.8f  (drift = %.2e)\n", N_final, abs(N_final - N_mass0))
        @printf("  Max drift:    %.2e\n", N_max_drift)
    end

    # --- Build GIF ---
    println("  Generating GIF...")
    ymax_plot = 1.3 * maximum(maximum(abs.(ψk)) for ψk in ψ_saves)

    # Determine x-range for plotting: focus on where the action is
    x_plot_max = min(Xmax, max(abs(a), abs(b)) + 15)

    # If absorbing, show the absorbing region boundary
    x_abs = use_absorbing ? Xmax - absorb_width : nothing

    anim = @animate for (k, ψk) in enumerate(ψ_saves)
        t = t_saves[k]
        Nk = dx * sum(abs2, ψk)

        p = plot(x_grid, abs.(ψk);
                 color=:blue, lw=2,
                 xlabel=L"x", ylabel=L"|\psi(x,t)|",
                 title=@sprintf("t = %.2f,  N = %.6f", t, Nk),
                 ylims=(0, ymax_plot),
                 xlims=(-x_plot_max, x_plot_max),
                 legend=false, size=(650, 350))

        # Show reference profile if available
        if ψ_bound !== nothing
            plot!(p, x_grid, abs.(ψ_bound);
                  color=:gray, lw=1.5, ls=:dash, alpha=0.5)
        end

        # Support boundary markers
        vline!(p, [a, b]; color=:gray40, ls=:dash, lw=1, alpha=0.4)

        # Absorbing region boundaries
        if x_abs !== nothing
            vline!(p, [-x_abs, x_abs]; color=:red, ls=:dot, lw=1, alpha=0.3)
        end
    end

    gif_dir = joinpath(results_dir, "dynamics")
    mkpath(gif_dir)
    ic_label = "$(label)_$(ic)"
    gif_path = joinpath(gif_dir, "$(ic_label).gif")
    gif(anim, gif_path; fps=fps)

    println("  GIF saved to: $gif_path")
    return gif_path
end
