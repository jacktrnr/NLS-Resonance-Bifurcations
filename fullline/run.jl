#!/usr/bin/env julia
#
# run.jl — NLS bifurcation workflow
#
# Usage:  cd clean/ && julia run.jl
#
# Adjust the parameters below and re-run to explore different potentials.

using Pkg
Pkg.activate(@__DIR__)

include("NLSBifurcation.jl")
using .NLSBifurcation

# ─── Parameters ───────────────────────────────────────────────────────────────

# Potential support
a, b = -1.0, 1.0

# Choose potential (uncomment one or define your own)
V0 = -π^2/4
Vfun = square_well(a, b, V0)
# Vfun = single_well(a, b, V0; skew=0.5, sharpness=4.0)
# Vfun = smooth_double_well(a, b, V0; separation=0.25)
# Vfun = cosine_well(a, b, V0)
# Vfun = sech2_well(a, b, V0; k=2.0)

# Seed-finding parameters
E_list = [-0.4]   # energies to scan for seeds
ζmax   = 8.0                   # max ζ parameter
nscan  = 3400                  # scan resolution

# Continuation parameters
Ngrid     = 7000               # ODE grid points
p_min     = -10.0              # min energy for continuation
max_steps = 500

# Plot parameters
plotEmin = -1.0
l2max    = 10.0
h1max    = 30.0
Xmax     = 13.0
ymax     = 3.0

# ─── Stage 1: Seeds + Continuation + Plots ───────────────────────────────────

# 1. Find seeds
seeds = find_all_seeds(a, b, Vfun;
                       E_list=E_list,
                       ζmax=ζmax,
                       nscan=nscan)

print_seed_table(seeds)
seeds = deduplicate_seeds(seeds)

# 2. Continue branches
branches = continue_from_seeds(seeds, a, b, Vfun;
                               N=Ngrid,
                               p_min=p_min,
                               max_steps=max_steps)

# 3. Plot results
plot_potential(Vfun; xmin=a-0.5, xmax=b+0.5)

plot_mass_energy(branches, a, b, Vfun;
                 Ngrid=Ngrid,
                 plotEmin=plotEmin,
                 l2max=l2max,
                 h1max=h1max)

plot_branch_profiles(branches, :auto;
                     a=a, b=b, Vfun=Vfun,
                     Ngrid=Ngrid,
                     Xmax=Xmax, ymax=ymax)

# ─── Stage 2: Spectral Analysis (optional) ───────────────────────────────────

run_spectral = false
spec_nev     = 4               # number of eigenvalues to track per operator
spec_ngrid   = 50              # sample points along branch
spec_Xmax    = 50.0            # domain half-width for FD grid
spec_FDgrid  = 1500            # FD grid size

if run_spectral
    println("\n" * "="^70)
    println("SPECTRAL ANALYSIS")
    println("="^70)

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue
        println("\n--- Branch $i ---")
        spec = track_spectrum_branch(br, a, b, Vfun;
                   nev=spec_nev, n_grid=spec_ngrid,
                   Xmax=spec_Xmax, Ngrid_FD=spec_FDgrid)
        plot_spectrum_evolution(spec; title="Branch $i spectrum")
    end

    # plot_stability_diagram(branches, a, b, Vfun;
    #                        nev=5, Xmax=spec_Xmax, Ngrid_FD=spec_FDgrid)
end

# ─── Stage 3: Time Dynamics (optional) ────────────────────────────────────
run_dynamics_flag = true
dyn_ic             = :soliton      # :groundstate, :soliton, :gaussian
dyn_Xmax           = 50.0
dyn_Ngrid          = 2048
dyn_Tmax           = 5.0
dyn_dt             = 1e-3
dyn_ε_scale        = 0.05
dyn_ε_pert         = 0.05
dyn_σ_pert         = 1.0
dyn_σ_gauss        = 2.0
dyn_save_every     = 100
dyn_fps            = 20
dyn_absorb_width   = 10.0          # 0 = hard walls, >0 = absorbing layer width
dyn_absorb_strength = 5.0          # CAP damping strength
dyn_absorb_power   = 3             # CAP ramp exponent (3 = cubic)

if run_dynamics_flag
    run_dynamics(branches, a, b, Vfun;
                 ic=dyn_ic, Xmax=dyn_Xmax, Ngrid=dyn_Ngrid,
                 Tmax=dyn_Tmax, dt=dyn_dt,
                 ε_scale=dyn_ε_scale, ε_pert=dyn_ε_pert,
                 σ_pert=dyn_σ_pert, σ_gauss=dyn_σ_gauss,
                 save_every=dyn_save_every, fps=dyn_fps,
                 absorb_width=dyn_absorb_width,
                 absorb_strength=dyn_absorb_strength,
                 absorb_power=dyn_absorb_power,
                 results_dir=joinpath(@__DIR__, "results"))
end

println("\nDone.")
