###############################################
# verify_table1.jl — Reproduce Table 1 of the paper
###############################################
#
# Paper: "Resonance-induced nonlinear bound states"
# Table 1: Square well V(x) = -π²/4 χ_{[-1,1]}(x)
#
# On the half-line with Dirichlet at x=0:
#   V(x) = -π²/4 on (0, 1), zero outside
#   Odd threshold resonance: U_★(x) = (2/π) sin(πx/2)
#   U_★(1) = 2/π, U_★'(1) = 0
#
# Predictions (Theorem 4.4):
#   E(ε)/ε → -½ U_★(1)² = -2/π² ≈ -0.2026
#   x_R(ε) → 3/4
#
# This script:
#   1. Uses the existing continuation to trace the bifurcation branch
#   2. Extracts (ε, E, x_R) at each branch point
#   3. Compares to predicted asymptotics
#
###############################################

using Printf, OrdinaryDiffEq

# Include dependencies
cd(joinpath(@__DIR__, "..", "halfline"))
include("core.jl")
include("potentials.jl")
include("resonances.jl")

println("="^70)
println("TABLE 1 VERIFICATION")
println("Resonance-induced nonlinear bound states")
println("="^70)
println()
println("Potential: V(x) = -π²/4 on [0,1]")
println("Threshold resonance: U_★(x) = (2/π) sin(πx/2)")
println("Predictions: E(ε)/ε → -2/π² ≈ -0.2026,  x_R(ε) → 3/4")
println()

# ─── Setup ─────────────────────────────────────────────────────────────────────

b = 1.0
V0 = -π^2 / 4
Vfun = square_well(b, V0)

# Analytic threshold mode
U_star_b = 2/π
xR_predicted = 3/4
dEdε_predicted = -2/π^2

println("Analytic values:")
@printf("  U_★(b) = 2/π = %.10f\n", U_star_b)
@printf("  E(ε)/ε → -½ U_★(b)² = -2/π² = %.10f\n", dEdε_predicted)
@printf("  x_R(ε) → 3/4 = 0.75\n")
println()

# ─── Verify via resonance computation ──────────────────────────────────────────

println("Computing resonances via companion EVP (N=600)...")
res = compute_resonances(b, Vfun; N=600, k_max=5.0)

# The threshold resonance should appear near κ=0
println("\nResonances found on imaginary axis:")
for r in res
    @printf("  k = %.6fi,  κ = %.6f,  E = %.6f  [%s]\n",
            imag(r.k), r.κ, r.E, r.type)
end

# Also compute the bifurcation coefficients from the analytic mode
println("\nBifurcation coefficients from analytic U_★:")
# For the square well, U_★(x) = (2/π) sin(πx/2) exactly
# I2 = ∫₀¹ (2/π)² sin²(πx/2) dx = (4/π²)(1/2) = 2/π²
# I4 = ∫₀¹ (2/π)⁴ sin⁴(πx/2) dx = (16/π⁴)(3/8) = 6/π⁴
I2_exact = 2/π^2
I4_exact = 6/π^4
xR_from_formula = b + I2_exact / U_star_b^2 - 2 * I4_exact / U_star_b^2

@printf("  I2 = ∫₀¹ U_★² dx = 2/π² = %.10f\n", I2_exact)
@printf("  I4 = ∫₀¹ U_★⁴ dx = 6/π⁴ = %.10f\n", I4_exact)
@printf("  x_R prediction = b + I2/U_b² - 2I4/U_b² = %.10f\n", xR_from_formula)
println()

# ─── Trace the bifurcation branch ─────────────────────────────────────────────

println("="^70)
println("BRANCH CONTINUATION (close to threshold)")
println("="^70)

# For V₀ = -π²/4, the threshold resonance is at E = 0.
# Near the bifurcation: E ≈ -(2/π²) β² ≈ -0.2026 β²
# So for β = 1: E ≈ -0.2; β = 2: E ≈ -0.81; β = 3: E ≈ -1.82
# We scan at several energies to find the branch.
Estart_list = [-0.4, -0.2, -0.8, -1.5]
seeds = find_seeds(b, Vfun;
    E_list=Estart_list,
    β_max=8.0,
    nβ=3000,
    N=5000,
    tol=1e-9)

if isempty(seeds)
    # Fallback: very fine scan at many energies
    Estart_list = collect(range(-0.05, -2.0; length=10))
    seeds = find_seeds(b, Vfun;
        E_list=Estart_list,
        β_max=10.0,
        nβ=5000,
        N=5000,
        tol=1e-8)
end

if isempty(seeds)
    error("No seeds found! Cannot reproduce Table 1.")
end

println("\nSeeds found: $(length(seeds))")
for (i, s) in enumerate(seeds)
    @printf("  Seed %d: β = %.6f, E = %.6f\n", i, s.β, s.E)
end

# Continue toward E = 0 (the threshold)
branches = continue_from_seeds(seeds, b, Vfun;
    N=5000,
    p_min=-2.0,
    p_max=-1e-6,
    ds=0.0005,
    dsmin=1e-7,
    dsmax=0.001,
    max_steps=5000,
    β_min=1e-5,
    verbose=0)

# ─── Extract Table 1 data ─────────────────────────────────────────────────────

println("\n" * "="^70)
println("TABLE 1 — Numerical verification of Theorem 4.4")
println("="^70)

# Find the branch with points near E = 0
best_branch = nothing
best_Emax = -Inf
for br in branches
    isempty(br.branch) && continue
    Es = [sol.param for sol in br.branch]
    Emax = maximum(Es)
    if Emax > best_Emax
        global best_Emax = Emax
        global best_branch = br
    end
end

if best_branch === nothing || isempty(best_branch.branch)
    error("No valid branch found near threshold.")
end

# Sort by β (ascending = closest to bifurcation first)
sorted_pts = sort(collect(best_branch.branch); by = p -> abs(p.β))

# Select target ε values matching the paper's table
target_εs = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

println()
@printf("  %8s | %14s | %12s | %14s\n", "ε", "E(ε)", "x_R(ε)", "x_R(ε) - 3/4")
println("  " * "-"^60)

for ε_target in target_εs
    # ε = β² in the paper's notation (ψ = √ε U, so ||ψ||² ~ ε)
    # Actually in Theorem 4.4, ε = η² where ψ = η u, so ε relates to the
    # amplitude. The branch parameter β = ψ'(0) = √ε U'(0) = √ε * 1 = √ε
    # So ε = β².

    β_target = sqrt(ε_target)

    # Find nearest branch point
    _, idx = findmin(abs.([p.β for p in sorted_pts] .- β_target))
    pt = sorted_pts[idx]

    E_val = pt.param
    β_val = pt.β
    ε_val = β_val^2

    # Compute x_R by gluing and finding the soliton center
    x_int, u_int, v_int = shoot_from_origin(b, E_val, Vfun, β_val; N=5000)

    if !isempty(x_int)
        κv = sqrt(-E_val)
        A = sqrt(-2E_val)
        ub = u_int[end]
        vb = v_int[end]

        # Match to soliton: ψ(x) = A sech(κ(x - x_R)) for x > b
        # At x = b: |ub| = A sech(κ(b - x_R))
        # => x_R = b - (1/κ) acosh(A/|ub|)  or  b + (1/κ) acosh(A/|ub|)
        # Choose based on derivative sign
        if abs(ub) > 1e-14 && abs(ub) < A
            ratio = A / abs(ub)
            y0 = acosh(ratio)
            # Check derivative to pick correct branch
            s = sign(ub)
            deriv_pos = -s * κv * A * sech(y0) * tanh(y0)
            deriv_neg = -s * κv * A * sech(-y0) * tanh(-y0)

            if abs(deriv_pos - vb) < abs(deriv_neg - vb)
                xR = b - y0/κv
            else
                xR = b + y0/κv
            end
        else
            xR = NaN
        end

        @printf("  %8.3f | %14.3e | %12.7f | %14.3e\n",
                ε_val, E_val, xR, xR - 0.75)
    end
end

println("  " * "-"^60)
println()
println("Predicted: E(ε)/ε → $(@sprintf("%.4f", dEdε_predicted)),  x_R → 0.7500")
println()

# ─── Also print E(ε)/ε convergence ────────────────────────────────────────────

println("\nConvergence of E(ε)/ε:")
@printf("  %8s | %14s | %14s | %10s\n", "ε", "E(ε)", "E(ε)/ε", "predicted")
println("  " * "-"^56)

for ε_target in target_εs
    β_target = sqrt(ε_target)
    _, idx = findmin(abs.([p.β for p in sorted_pts] .- β_target))
    pt = sorted_pts[idx]
    E_val = pt.param
    ε_val = pt.β^2

    @printf("  %8.3f | %14.3e | %14.6f | %10.6f\n",
            ε_val, E_val, E_val/ε_val, dEdε_predicted)
end
println("  " * "-"^56)
println()
println("Done.")
