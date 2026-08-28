"""
    NLSBifurcations

Numerical bifurcation analysis of nonlinear bound states of the 1D NLS/GP equation

    -ψ'' + V(x)ψ - |ψ|²ψ = Eψ

with compactly supported potentials. Computes linear scattering data (bound state
poles, scattering resonances, transmission resonances) and traces nonlinear
bifurcation branches via pseudo-arclength continuation.

See: "Resonance-induced nonlinear bound states" (Turner & Weinstein, Nonlinearity 2026)
"""
module NLSBifurcations

using OrdinaryDiffEq
using OrdinaryDiffEq.SciMLBase: ReturnCode
using LinearAlgebra
using Printf
using FFTW
using Plots
using LaTeXStrings
using Colors
using Accessors
using BifurcationKit
using Random

# Core utilities
include("globals.jl")

# Compactly supported potentials (including paper_potential from eq 5.1)
include("potentials.jl")

# ODE integration and Hamiltonian residual
include("shooting.jl")

# Tail matching and solution gluing
include("glue.jl")

# Seed finding and BifurcationKit continuation
include("continuation.jl")

# Linear scattering theory (QEP for w(k) and s₋(k) zeros)
include("scattering.jl")

# L+/L- spectral analysis
include("spectral.jl")

# Time dynamics (split-step)
include("dynamics.jl")

# Plotting
include("plots.jl")

# ─── Exports ───────────────────────────────────────────────────────────────────

# Utilities
export κ, c_from_ζ, q_from_ζ, clamp1, safe_div, sign_real, bisect_zero

# Potentials
export square_well, two_step_potential, three_step_potential,
       single_well, smooth_double_well, gaussian_bump,
       cosine_well, polynomial_bump, tent_potential,
       skewed_single_well, asymmetric_step_array,
       multi_gaussian_wells, random_bumps, sech2_well, double_well,
       paper_potential, find_threshold_alpha

# Shooting
export integrate_support, H_residual_ζ

# Gluing and norms
export tail_shifts_from_ends, glue_full_solution,
       compute_norm, compute_H1_norm, can_glue

# Continuation
export find_seeds_at_E, find_all_seeds, print_seed_table,
       filter_seeds, deduplicate_seeds,
       continue_from_seeds, continue_single_seed,
       find_branches_at_fixed_E, find_branches_over_Es,
       find_all_branches_from_scattering

# Scattering theory
export compute_w_zeros, compute_s_minus_zeros,
       compute_all_scattering_data, print_scattering_summary,
       find_threshold_alpha_qep

# Spectral analysis
export linear_interp, compute_Lpm_eigenvalues, track_spectrum_branch

# Dynamics
export splitstep_evolve, run_dynamics, build_absorbing_layer,
       build_ic_groundstate, build_ic_perturbed_soliton, build_ic_gaussian,
       get_branch_Emin_mass

# Plotting
export set_plot_style!,
       plot_mass_energy, plot_branch_profiles, plot_potential,
       plot_spectrum_evolution, plot_stability_diagram

end # module
