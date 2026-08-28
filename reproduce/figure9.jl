###############################################
# figure9.jl — Reproduce Figure 9 (square well)
###############################################
#
# Generates both panels of Figure 9:
#   Left:  Scattering data (poles/zeros in complex k-plane)
#   Right: N[ψ_E] vs E bifurcation diagram
#
# Run:  julia reproduce/figure9.jl
#
###############################################

cd(joinpath(@__DIR__, "..", "fullline"))
include(joinpath(@__DIR__, "..", "fullline", "NLSBifurcation.jl"))
using .NLSBifurcation
using Printf

println("="^70)
println("FIGURE 9 — Symmetric square well V(x) = -α χ_{[-1,1]}(x)")
println("="^70)

a, b = -1.0, 1.0
α_star = π^2 / 4  # ≈ 2.467

cases = [
    (α = 2.0,    label = "(a) α < α★"),
    (α = α_star, label = "(b) α = α★ = π²/4"),
    (α = 3.0,    label = "(c) α > α★"),
]

for case in cases
    α = case.α
    Vfun = square_well(a, b, -α)

    println("\n" * "="^70)
    println("  $(case.label):  α = $(round(α, digits=4))")
    println("="^70)

    # ─── Scattering data (left panel) ─────────────────────────────────────
    println("\n  [1] Scattering data:")
    scat = compute_all_scattering_data(a, b, Vfun; N=500, k_max=4.0, verbose=false)

    println("      Bound state poles (●):")
    for r in scat.bound_poles
        @printf("        k = %.4fi  →  E = %.4f\n", r.κ, r.E)
    end
    println("      Scattering resonance poles (●):")
    for r in scat.scat_poles
        @printf("        k = %.4fi  →  E = %.4f\n", r.κ, r.E)
    end
    println("      Transmission resonances (✕):")
    for r in scat.trans_res
        @printf("        k = %.4f + %.4fi\n", real(r.k), imag(r.k))
    end

    # ─── Nonlinear branches (right panel) ─────────────────────────────────
    println("\n  [2] Nonlinear bifurcation branches:")

    # Scan energies informed by the scattering data
    E_scan = Float64[]
    for r in scat.bound_poles
        push!(E_scan, r.E - 0.1)
        push!(E_scan, r.E - 0.3)
    end
    for r in scat.scat_poles
        push!(E_scan, r.E - 0.1)
        push!(E_scan, r.E - 0.3)
    end
    # Add generic scan points
    append!(E_scan, [-0.1, -0.3, -0.5, -0.8, -1.2, -1.5])
    E_scan = sort(unique(filter(e -> -4.0 ≤ e < -0.01, E_scan)))

    seeds = find_all_seeds(a, b, Vfun;
        E_list=E_scan, ζmax=10.0, nscan=4000, N=5000, tolH=1e-7,
        slope_set=(+1, -1))
    seeds = deduplicate_seeds(seeds; E_tol=0.02, c_tol=0.02)

    if isempty(seeds)
        println("      No seeds found!")
        continue
    end

    branches = continue_from_seeds(seeds, a, b, Vfun;
        N=5000, p_min=-4.0, max_steps=800,
        ds=0.005, dsmin=1e-5, dsmax=0.01,
        verbose=0)

    # Compute norms and print bifurcation data
    println("\n      Branch | Points |   E range       | N range         | slope")
    println("      " * "-"^65)

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue
        length(br.branch) < 5 && continue

        Es = [sol.param for sol in br.branch]
        E_lo, E_hi = extrema(Es)
        slope = br.branch[1].slope_sign

        # Compute N at a few sample points
        n_sample = min(5, length(br.branch))
        indices = round.(Int, range(1, length(br.branch); length=n_sample))
        Ns = Float64[]
        for idx in indices
            sol = br.branch[idx]
            E_val = sol.param
            ζ_val = sol.ζ
            c_val = c_from_ζ(ζ_val, E_val)
            xs, us, vs = integrate_support(a, b, E_val, Vfun;
                N=5000, c=c_val, slope_sign=slope)
            if !isempty(xs)
                ub = real(us[end])
                vb = real(vs[end])
                if can_glue(E_val, ub, vb)
                    N_val = compute_norm(a, b, E_val, xs, us, vs)
                    isfinite(N_val) && N_val > 0 && push!(Ns, N_val)
                end
            end
        end

        if !isempty(Ns)
            N_lo, N_hi = extrema(Ns)
            @printf("      %5d  | %5d  | [%6.3f, %6.3f] | [%5.2f, %5.2f] | %+d\n",
                    i, length(br.branch), E_lo, E_hi, N_lo, N_hi, slope)
        end
    end

    # Print the key bifurcation threshold (N at E closest to predicted)
    println("\n      Key bifurcation features:")
    for r in scat.bound_poles
        @printf("        Bound pole at E = %.4f → branch starts at N → 0\n", r.E)
    end
    for r in scat.scat_poles
        N_threshold = 8 * abs(r.κ)  # 2 × N[S_E] = 2 × 4|κ| = 8|κ|
        @printf("        Resonance at E = %.4f → branch starts at N ≈ %.2f\n", r.E, N_threshold)
    end
end

println("\n" * "="^70)
println("DONE — Figure 9 data generated")
println("="^70)
