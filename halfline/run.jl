###############################################
# run.jl — Half-line NLS bifurcation analysis
###############################################
#
# Solves: -ψ'' + V(x)ψ - ψ³ = Eψ  on x > 0
# BC:     ψ(0) = 0 (Dirichlet)
# Shooting parameter: β = ψ'(0)
#
# Workflow:
#   Stage 1 — Find seeds (zeros of F(β,E)), continue branches via
#             BifurcationKit, compute norms, plot, and save.
#   Stage 2 — (Optional) Track L₊/L₋ spectra along branches,
#             build stability diagram, save.
#   Stage 2b — (Alternative) Load saved JLD2 and re-run spectral only.
#
# Usage:
#   In VS Code Julia REPL, just  include("run.jl")
#   Edit the CONFIGURATION block below, then re-include.
#
###############################################

# ============================================
# CONFIGURATION — edit these parameters
# ============================================

# --- Potential type ---
# Available types:
#   :square       — Constant well V(x) = V0 on (0,b), zero outside.
#                   The simplest case. Good for testing.
#   :square_bump  — Square well V0 on (0,b) plus a Gaussian bump near x = b.
#                   V(x) = V0 + A*exp(-(x - c)^2 / 2w^2) on (0,b).
#                   Controlled by bump_amp_factor, bump_width_frac, bump_center_frac.
#   :step         — Piecewise constant: V = V1 on [0, b/2), V = V0 on [b/2, b).
#                   Set V1 below (default -6.0).
#   :gaussian     — Gaussian well: V(x) = V0 * exp(-(x - b/2)^2 / σ^2) on (0,b).
#                   Width σ = σ_frac * b.
#   :threestep    — Three-step: edge_height on [0,b/4] and [3b/4,b], V0 in middle.
#   :smooth       — Compactly supported bump function: V0 * exp(-1/(1-(x/b)^2)).
#                   C^∞ and vanishes exactly at x = 0 and x = b.
potential_type = :square_bump # :smooth, :threestep, :gaussian, :step, :square, :square_bump
b = 1.0                  # Support boundary: V(x) = 0 for x > b
V0 = -3.3 # Potential depth/height (negative = attractive well)

# Type-specific parameters (ignored if not relevant to chosen type):
bump_amp_factor = 2.3    # :square_bump — bump amplitude = factor * |V0|
bump_width_frac = 0.1    # :square_bump — bump width = fraction * b
bump_center_frac = 1.0   # :square_bump — bump center = fraction * b
V1 = -6.0               # :step — left-half depth
σ_frac = 0.25           # :gaussian — width σ = fraction * b
edge_height = 0.1       # :threestep — height of edge regions

# --- Seed finding ---
# Estart: initial energy guesses where we scan for solutions.
#   Each E in this list gets a full β-scan. Use values where you
#   expect bifurcations (usually just below the linear eigenvalue).
Estart = [-1.2]
β_max = 10.0             # Maximum shooting slope to scan
nβ = 800                 # Number of β gridpoints (higher = finer scan)
N = 3000                 # ODE integration gridpoints on [0, b]
seed_tol = 1e-9          # Tolerance for accepting a zero of F(β, E)

# --- Continuation ---
# BifurcationKit pseudo-arclength continuation parameters.
ds = 0.001               # Initial step size
dsmin = 1e-6             # Minimum step size (smaller = more robust near folds)
dsmax = 0.001            # Maximum step size
Emin = -2.0              # Lower bound on E for continuation (p_min)
max_steps = 3000         # Maximum continuation steps per branch
β_min = 1e-3             # Stop branch when |β| drops below this

# --- Spectral analysis (Stage 2) ---
# Set run_spectral = false to skip entirely (much faster).
run_spectral = false
nev = 2                  # Number of eigenvalues to compute for L₊ and L₋
spectral_skip = 1        # Compute spectrum every `skip` branch points (1 = all)
Ngrid = 1000             # Finite-difference grid size for eigenvalue problem
Xmax_spec = 50.0         # Domain truncation for spectral computation

# --- Output ---
save_data = true         # Save branch data to JLD2
save_plots_flag = true   # Save all plots as PNGs
results_dir = joinpath(@__DIR__, "results")

# --- Time dynamics (Stage 3) ---
# Set run_dynamics_flag = false to skip entirely.
run_dynamics_flag = true
dyn_use_endpoint = true  # false = use seed (Estart), true = use branch endpoint closest to E=0
dyn_Xmax = 100.0          # Domain truncation for dynamics grid
dyn_Ngrid = 2048         # Number of interior grid points (power of 2 recommended)
dyn_Tmax = 50.0          # Total evolution time
dyn_dt = 1e-3            # Time step
dyn_ε_scale = 0.05       # Rescaling of bound state: (1 + ε_scale) * ψ₀
dyn_ε_pert = 0.05        # Amplitude of derivative-of-Gaussian perturbation
dyn_σ_pert = 1.0         # Width of the Gaussian in the perturbation
dyn_save_every = 100     # Save a frame every this many time steps
dyn_fps = 20             # GIF frames per second

# --- Profile plot settings ---
Xmax_profiles = 15.0     # Wide-view x-axis limit for profile plots
n_profiles = 20          # Number of profiles to show per branch

# ============================================
# SETUP — don't edit below unless needed
# ============================================

using Printf

include("core.jl")
include("potentials.jl")
include("resonances.jl")
include("plotting.jl")
include("save.jl")
include("dynamics.jl")

t_start = time()

# ============================================
# STAGE 1: Seeds + Continuation
# ============================================

println("="^70)
println("HALF-LINE NLS BIFURCATION ANALYSIS")
println("="^70)
println("  Equation : -ψ'' + V(x)ψ - ψ³ = Eψ  on x > 0")
println("  BC       : ψ(0) = 0")
println("  Potential: $potential_type  (b = $b, V0 = $V0)")
println("="^70)

# Build potential via dispatcher
Vfun = make_potential(potential_type;
    b=b, V0=V0,
    bump_amp_factor=bump_amp_factor,
    bump_width_frac=bump_width_frac,
    bump_center_frac=bump_center_frac,
    V1=V1, σ_frac=σ_frac, edge_height=edge_height)

label = potential_label(potential_type; b=b, V0=V0)
println("  Label    : $label")
println()

# Plot potential
plt_V = plot_potential(Vfun, b; xmax=max(3*b, 5.0))

# --- Find seeds ---
println("\n" * "="^70)
println("SEED FINDING")
println("="^70)
println("  Scanning β ∈ (0, $β_max] at E = $Estart")
println("  Grid: $nβ points, tol = $seed_tol")
println()

seeds = find_seeds(b, Vfun;
    E_list=Estart,
    β_max=β_max,
    nβ=nβ,
    N=N,
    tol=seed_tol)

if isempty(seeds)
    println("\nNo seeds found! Try adjusting β_max or Estart.")
    error("No seeds found — aborting.")
end

# Print seed table
println("\nSeeds found:")
println("-"^50)
println("  #  |     E      |      β      |   F(β,E)   ")
println("-"^50)
for (i, seed) in enumerate(seeds)
    F_val = F_residual(b, seed.E, Vfun, seed.β; N=N)
    @printf("  %2d | %10.5f | %11.6f | %10.2e\n", i, seed.E, seed.β, F_val)
end
println("-"^50)

# --- Continue branches ---
println("\n" * "="^70)
println("BRANCH CONTINUATION")
println("="^70)
println("  ds = $ds, dsmin = $dsmin, dsmax = $dsmax")
println("  E range: [$Emin, 0), max_steps = $max_steps")
println()

branches = continue_from_seeds(seeds, b, Vfun;
    N=N,
    p_min=Emin,
    p_max=-1e-10,
    ds=ds,
    dsmin=dsmin,
    dsmax=dsmax,
    max_steps=max_steps,
    β_min=β_min,
    verbose=1)

# --- Branch summary ---
println("\n" * "="^70)
println("BRANCH SUMMARY")
println("="^70)

for (i, br) in enumerate(branches)
    if isempty(br.branch)
        println("  Branch $i: EMPTY")
        continue
    end
    Es = [sol.param for sol in br.branch]
    βs = [sol.β for sol in br.branch]
    @printf("  Branch %d: %d points, E ∈ [%.4f, %.4f], β ∈ [%.4f, %.4f]\n",
            i, length(br.branch), minimum(Es), maximum(Es),
            minimum(βs), maximum(βs))
end

# --- Visualization ---
println("\n" * "="^70)
println("VISUALIZATION")
println("="^70)

# Mass-energy: full view + zoomed view (auto-determined)
println("\n  Plotting mass vs energy (full + zoomed)...")
plt_L2, plt_H1, plt_L2_zoom, plt_H1_zoom = plot_mass_energy(
    branches, b, Vfun; N=N, Emin=Emin)

# Solution profiles: wide view + zoomed near the support
println("  Plotting solution profiles (wide + zoomed)...")
plt_profiles = plot_profiles(branches, b, Vfun;
    N=N, Xmax=Xmax_profiles, n_profiles=n_profiles)
plt_profiles_zoom = plot_profiles(branches, b, Vfun;
    N=N, Xmax=max(3*b, 4.0), n_profiles=n_profiles)

# --- Save data (Stage 1) ---
if save_data
    data_dir = joinpath(results_dir, String(potential_type), "data")
    mkpath(data_dir)
    data_path = joinpath(data_dir, "$(label).jld2")
    save_run_data(data_path;
        branches=branches, seeds=seeds,
        b=b, V0=V0, potential_type=potential_type)
end

# --- Save plots (Stage 1) ---
if save_plots_flag
    plot_dir = joinpath(results_dir, String(potential_type))
    save_plots(plot_dir, label;
        potential_plt=plt_V,
        L2_plt=plt_L2,
        H1_plt=plt_H1,
        L2_zoom_plt=plt_L2_zoom,
        H1_zoom_plt=plt_H1_zoom,
        profiles_plt=plt_profiles,
        profiles_zoom_plt=plt_profiles_zoom)
end

t_stage1 = time() - t_start
@printf("\nStage 1 completed in %.1f seconds.\n", t_stage1)

# ============================================
# STAGE 2: Spectral Analysis (optional)
# ============================================

if run_spectral
    println("\n" * "="^70)
    println("SPECTRAL ANALYSIS")
    println("="^70)
    println("  nev = $nev, Ngrid = $Ngrid, Xmax = $Xmax_spec")
    println("  n_grid = 50 (E values for spectral tracking)")
    println()

    spectral_data = []
    spectrum_plts = []

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue
        length(br.branch) < 5 && continue

        println("  Analyzing Branch $i spectrum...")

        spec = track_spectrum_branch(br, b, Vfun;
            nev=nev,
            n_grid=50,
            Ngrid=Ngrid,
            Xmax=Xmax_spec)

        push!(spectral_data, (branch_idx=i, data=spec))

        if !isempty(spec.Es)
            plt_spec = plot_spectrum_evolution(spec;
                n_show=min(nev, 3),
                title="Branch $i Spectrum")
            push!(spectrum_plts, plt_spec)
        end
    end

    # Stability diagram (mass vs E colored by stability + eigenvalue counts)
    println("\n  Computing stability diagram...")
    plt_stability = plot_stability_diagram(branches, b, Vfun;
        nev=nev, skip=max(1, spectral_skip),
        N=N, Xmax=Xmax_spec, Ngrid=Ngrid)

    # Save spectral plots
    if save_plots_flag
        plot_dir = joinpath(results_dir, String(potential_type))
        save_plots(plot_dir, label;
            spectrum_plts=spectrum_plts,
            stability_plt=plt_stability)
    end

    t_total = time() - t_start
    println("\n" * "="^70)
    @printf("ANALYSIS COMPLETE  (%.1f seconds total)\n", t_total)
    println("="^70)
else
    println("\nSpectral analysis skipped (set run_spectral = true to enable).")
    spectral_data = []
end

# ============================================
# STAGE 3: Time Dynamics (optional)
# ============================================

if run_dynamics_flag
    println("\n" * "="^70)
    println("TIME DYNAMICS")
    println("="^70)
    println("  ε_scale = $dyn_ε_scale, ε_pert = $dyn_ε_pert, σ_pert = $dyn_σ_pert")
    println("  Tmax = $dyn_Tmax, dt = $dyn_dt")
    println("  Grid: $dyn_Ngrid points, Xmax = $dyn_Xmax")
    println()

    dyn_label = potential_label(potential_type; b=b, V0=V0)

    run_dynamics_result = run_dynamics(seeds, branches, b, Vfun;
        use_endpoint=dyn_use_endpoint,
        Xmax=dyn_Xmax, Ngrid=dyn_Ngrid, N_ode=N,
        Tmax=dyn_Tmax, dt=dyn_dt,
        ε_scale=dyn_ε_scale, ε_pert=dyn_ε_pert, σ_pert=dyn_σ_pert,
        save_every=dyn_save_every, fps=dyn_fps,
        results_dir=joinpath(results_dir, String(potential_type)),
        label=dyn_label)
else
    println("\nTime dynamics skipped (set run_dynamics_flag = true to enable).")
end

# ============================================
# STAGE 2b: Load from saved data (alternative)
# ============================================
# To re-run spectral analysis on a previous run without re-computing
# the branches, uncomment the block below and comment out Stages 1-2.
#
# data = load_run_data("results/square_bump/data/square_bump_b=1.0_V0=-3.4.jld2")
#
# # Rebuild the potential from saved metadata
# Vfun_loaded = make_potential(data.potential_type; b=data.b, V0=data.V0,
#     bump_amp_factor=bump_amp_factor, bump_width_frac=bump_width_frac,
#     bump_center_frac=bump_center_frac, V1=V1, σ_frac=σ_frac, edge_height=edge_height)
#
# # Loop over saved branches and run spectral
# for (i, bd) in enumerate(data.branch_data)
#     isempty(bd.points) && continue
#     println("\nLoaded Branch $i: $(length(bd.points)) points")
#
#     # Wrap saved points in a branch-like NamedTuple for track_spectrum_branch
#     br_loaded = (; branch = bd.points)
#
#     spec = track_spectrum_branch(br_loaded, data.b, Vfun_loaded;
#         nev=nev, skip=max(1, spectral_skip), Ngrid=Ngrid, Xmax=Xmax_spec)
#
#     if !isempty(spec.Es)
#         plot_spectrum_evolution(spec; n_show=min(nev, 3),
#             title="Loaded Branch $i Spectrum")
#     end
# end
