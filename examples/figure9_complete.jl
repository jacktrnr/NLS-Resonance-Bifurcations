###############################################
# figure9_complete.jl — Complete bifurcation diagram
###############################################
#
# Produces the full N vs E diagram for α = 2.0 with all branches
# connected through folds. The key insight: the resonance branch
# has a fold near E ≈ -1 that the shooting parametrization can't
# traverse in one shot. We trace both legs and join them.
#
# Run:  julia --project=. examples/figure9_complete.jl
#
###############################################

cd(@__DIR__)
include(joinpath("..", "src", "NLSBifurcations.jl"))
using .NLSBifurcations
using Printf
ENV["GKSwstype"] = "100"
using Plots; gr()

println("="^70)
println("COMPLETE BIFURCATION DIAGRAM: α = 2.0 < α★")
println("="^70)

a, b = -1.0, 1.0
Vfun = square_well(a, b, -2.0)

# ═══════════════════════════════════════════════════════════════════
# Scattering data
# ═══════════════════════════════════════════════════════════════════
scat = compute_all_scattering_data(a, b, Vfun; N=800, k_max=4.0, verbose=false)
E_bound = -scat.bound_poles[1].κ^2
E_res = -scat.scat_poles[1].κ^2
N_res = 8 * abs(scat.scat_poles[1].κ)

@printf("\nBound pole:  κ = %.4f, E_bif = %.4f\n", scat.bound_poles[1].κ, E_bound)
@printf("Scat res:    κ = %.4f, E_bif = %.4f, N_thr = %.4f\n",
        scat.scat_poles[1].κ, E_res, N_res)

# ═══════════════════════════════════════════════════════════════════
# Helper: compute N(E) along a branch
# ═══════════════════════════════════════════════════════════════════
function branch_NE(br, slope; Vfun=Vfun, a=a, b=b, Npts=5000, n_sample=100)
    isempty(br.branch) && return Float64[], Float64[]
    Es = Float64[]
    Ns = Float64[]
    step = max(1, div(length(br.branch), n_sample))
    for j in 1:step:length(br.branch)
        sol = br.branch[j]
        c_val = c_from_ζ(sol.ζ, sol.param)
        xs, us, vs = integrate_support(a, b, sol.param, Vfun; N=Npts, c=c_val, slope_sign=slope)
        if !isempty(xs) && can_glue(sol.param, real(us[end]), real(vs[end]))
            N_val = compute_norm(a, b, sol.param, xs, us, vs)
            if isfinite(N_val) && N_val > 0
                push!(Es, sol.param)
                push!(Ns, N_val)
            end
        end
    end
    return Es, Ns
end

# ═══════════════════════════════════════════════════════════════════
# Branch 1: Bound state pole (slope=+1, small ζ)
# ═══════════════════════════════════════════════════════════════════
println("\n[1] Bound state branch...")
seeds_bp = find_all_seeds(a, b, Vfun;
    E_list=[-1.3], ζmax=0.5, nscan=2000, N=6000, tolH=1e-7, slope_set=(+1,))

br_bound = continue_single_seed(seeds_bp[1], a, b, Vfun;
    N=6000, p_min=-4.0, p_max=-0.01, max_steps=2000,
    ds=0.003, dsmin=1e-6, dsmax=0.008, ζ_min=1e-5, verbose=0)
Es_bound, Ns_bound = branch_NE(br_bound, +1)
@printf("  %d points, E ∈ [%.3f, %.4f], N ∈ [%.3f, %.3f]\n",
        length(Es_bound), minimum(Es_bound), maximum(Es_bound),
        minimum(Ns_bound), maximum(Ns_bound))

# ═══════════════════════════════════════════════════════════════════
# Branch 2: Resonance branch LOWER leg (slope=-1, moderate ζ)
# Goes from resonance endpoint (E≈-0.07, N≈2) leftward to fold (E≈-1)
# ═══════════════════════════════════════════════════════════════════
println("\n[2] Resonance branch (lower leg)...")
seeds_res = find_all_seeds(a, b, Vfun;
    E_list=[-0.3], ζmax=5.0, nscan=3000, N=6000, tolH=1e-7, slope_set=(-1,))

br_res_lo = continue_single_seed(seeds_res[1], a, b, Vfun;
    N=6000, p_min=-1.5, p_max=-0.005, max_steps=3000,
    ds=0.002, dsmin=1e-7, dsmax=0.005, ζ_min=1e-4, verbose=0)
Es_res_lo, Ns_res_lo = branch_NE(br_res_lo, -1)
@printf("  %d points, E ∈ [%.3f, %.4f], N ∈ [%.3f, %.3f]\n",
        length(Es_res_lo), minimum(Es_res_lo), maximum(Es_res_lo),
        minimum(Ns_res_lo), maximum(Ns_res_lo))

# ═══════════════════════════════════════════════════════════════════
# Branch 3: Resonance branch UPPER leg (slope=+1, large ζ)
# Continues from fold (E≈-1) leftward at higher N
# ═══════════════════════════════════════════════════════════════════
println("\n[3] Resonance branch (upper leg past fold)...")
seeds_up = find_all_seeds(a, b, Vfun;
    E_list=[-1.3], ζmax=6.0, nscan=4000, N=6000, tolH=1e-7, slope_set=(+1,))
# Take the HIGH-ζ seed (not the bound state one at low ζ)
seeds_up_high = filter(s -> s.ζ > 1.0, seeds_up)

br_res_hi = continue_single_seed(seeds_up_high[1], a, b, Vfun;
    N=6000, p_min=-4.0, p_max=-0.5, max_steps=3000,
    ds=0.003, dsmin=1e-6, dsmax=0.008, ζ_min=1e-4, verbose=0)
Es_res_hi, Ns_res_hi = branch_NE(br_res_hi, +1)
@printf("  %d points, E ∈ [%.3f, %.4f], N ∈ [%.3f, %.3f]\n",
        length(Es_res_hi), minimum(Es_res_hi), maximum(Es_res_hi),
        minimum(Ns_res_hi), maximum(Ns_res_hi))

# ═══════════════════════════════════════════════════════════════════
# Branch 4: Second upper leg (slope=-1, large ζ)
# ═══════════════════════════════════════════════════════════════════
println("\n[4] Second upper family (slope=-1, large ζ)...")
seeds_up2 = find_all_seeds(a, b, Vfun;
    E_list=[-1.3], ζmax=6.0, nscan=4000, N=6000, tolH=1e-7, slope_set=(-1,))
seeds_up2_high = filter(s -> s.ζ > 1.0, seeds_up2)

if !isempty(seeds_up2_high)
    br_up2 = continue_single_seed(seeds_up2_high[1], a, b, Vfun;
        N=6000, p_min=-4.0, p_max=-0.5, max_steps=3000,
        ds=0.003, dsmin=1e-6, dsmax=0.008, ζ_min=1e-4, verbose=0)
    Es_up2, Ns_up2 = branch_NE(br_up2, -1)
    @printf("  %d points, E ∈ [%.3f, %.4f], N ∈ [%.3f, %.3f]\n",
            length(Es_up2), minimum(Es_up2), maximum(Es_up2),
            minimum(Ns_up2), maximum(Ns_up2))
else
    Es_up2, Ns_up2 = Float64[], Float64[]
    println("  (not found)")
end

# ═══════════════════════════════════════════════════════════════════
# Branch 5: Third upper leg (slope=+1, highest ζ)
# ═══════════════════════════════════════════════════════════════════
println("\n[5] Third upper family (slope=+1, highest ζ)...")
seeds_up3 = find_all_seeds(a, b, Vfun;
    E_list=[-1.5], ζmax=8.0, nscan=4000, N=6000, tolH=1e-7, slope_set=(+1,))
seeds_up3_high = filter(s -> s.ζ > 2.5, seeds_up3)

if !isempty(seeds_up3_high)
    br_up3 = continue_single_seed(seeds_up3_high[1], a, b, Vfun;
        N=6000, p_min=-4.0, p_max=-0.5, max_steps=3000,
        ds=0.003, dsmin=1e-6, dsmax=0.008, ζ_min=1e-4, verbose=0)
    Es_up3, Ns_up3 = branch_NE(br_up3, +1)
    @printf("  %d points, E ∈ [%.3f, %.4f], N ∈ [%.3f, %.3f]\n",
            length(Es_up3), minimum(Es_up3), maximum(Es_up3),
            minimum(Ns_up3), maximum(Ns_up3))
else
    Es_up3, Ns_up3 = Float64[], Float64[]
    println("  (not found)")
end

# ═══════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════
println("\n[6] Generating plot...")

plt = plot(; xlabel="E", ylabel="N[ψ_E]",
    title="α = 2.0 < α★  (complete diagram)",
    xlims=(-4.0, 0.1), ylims=(0, 10),
    size=(600, 400), dpi=150, framestyle=:box, legend=:topright)

# Bound state branch
plot!(plt, Es_bound, Ns_bound; lw=2.5, color=:darkgreen, label="bound pole branch")

# Resonance branch (lower + upper = one continuous curve through fold)
plot!(plt, Es_res_lo, Ns_res_lo; lw=2.5, color=:blue, label="resonance branch (lower)")
plot!(plt, Es_res_hi, Ns_res_hi; lw=2.5, color=:red, label="resonance branch (upper, past fold)")

# Additional upper legs
if !isempty(Es_up2)
    plot!(plt, Es_up2, Ns_up2; lw=2, color=:cyan, label="upper family (slope=-1)")
end
if !isempty(Es_up3)
    plot!(plt, Es_up3, Ns_up3; lw=2, color=:purple, label="upper family (highest)")
end

# Mark bifurcation points
scatter!(plt, [E_bound], [0.0]; marker=:circle, ms=8, color=:green, label="")
scatter!(plt, [E_res], [N_res]; marker=:circle, ms=8, color=:green, label="")

# Mark the fold
E_fold = maximum(Es_res_lo[Ns_res_lo .> 5.0]; init=-1.0)
N_fold = isempty(Es_res_lo) ? 6.0 : Ns_res_lo[argmax(Es_res_lo .* (Ns_res_lo .> 5.0))]
scatter!(plt, [E_fold], [N_fold]; marker=:star5, ms=8, color=:orange, label="fold")

savefig(plt, joinpath(@__DIR__, "..", "docs", "figures", "figure9a_complete.png"))
println("  Saved to docs/figures/figure9a_complete.png")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
println("\n" * "="^70)
println("BRANCH TOPOLOGY SUMMARY")
println("="^70)
println("""
  ● Bound state pole branch (dark green):
    From (E=$(round(E_bound,digits=3)), N=0) → curves left to (E=-4, N≈5.4)
    Origin: w(iκ)=0 at κ = $(round(scat.bound_poles[1].κ,digits=4))

  ● Resonance branch LOWER leg (blue):
    From (E=$(round(E_res,digits=3)), N=$(round(N_res,digits=2))) → curves left to fold at E≈-1
    Origin: w(iκ)=0 at κ = $(round(scat.scat_poles[1].κ,digits=4))
    Threshold: N_thr = 8|κ| = $(round(N_res,digits=4))

  ● Resonance branch UPPER leg (red):
    From fold at E≈-1 → continues left at higher N
    This IS the same branch as the blue curve, past the fold.

  ● Additional upper families (cyan, purple):
    Distinct solution families (different internal node count)
    Connect to the resonance family at the fold region.

  The fold at E≈-1 is where the shooting parametrization becomes
  singular — the physical branch is smooth through it.
""")
