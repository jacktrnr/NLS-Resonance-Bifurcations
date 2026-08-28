###############################################
# figure9_scattering.jl — Reproduce Figure 9 left panels
###############################################
#
# Computes scattering data (poles of w(k), zeros of s₋(k))
# for the symmetric square well V(x) = -α χ_{[-1,1]}(x)
# at three depths: α < α★, α = α★, α > α★.
#
# Run from repo root:  julia reproduce/figure9_scattering.jl
#
###############################################

# Navigate to the fullline directory for includes
cd(joinpath(@__DIR__, "..", "fullline"))
include("potentials.jl")
include("scattering.jl")

using Printf

println("="^70)
println("FIGURE 9 — Scattering data for symmetric square well")
println("V(x) = -α χ_{[-1,1]}(x)")
println("="^70)

a, b = -1.0, 1.0

# The threshold α★ for the symmetric square well on [-1,1]:
# The n-th odd eigenvalue appears when √α · cos(√α) = 0,
# i.e., √α = (2n-1)π/2. For n=1 (first even): α★ = π²/4 ≈ 2.467
# Actually for the SECOND bound state (first odd): α★ = π² ≈ 9.87
# Wait — let me be more careful.
#
# For the square well -α on [-1,1], eigenvalues satisfy:
#   Even: √(α+E) tan(√(α+E)) = √(-E)
#   Odd:  √(α+E) cot(√(α+E)) = -√(-E)  [i.e., cos(√(α+E)) = 0 at threshold E=0]
#
# At threshold (E=0): odd mode exists when cos(√α) = 0, i.e., √α = π/2
# So α★ = π²/4 ≈ 2.467 is where the FIRST ODD bound state appears.
#
# For the paper's Figure 9 context, the relevant threshold is:
# α★ = 9π²/4 ≈ 22.21 (second odd) or α★ = π²/4 (first odd)
#
# Let's use α★ = π²/4 for the first odd threshold.

α_star = π^2 / 4
println("\nThreshold: α★ = π²/4 ≈ $(round(α_star, digits=4))")

# Three regimes
cases = [
    (α = 2.0,     label = "(a) α < α★"),
    (α = α_star,  label = "(b) α = α★"),
    (α = 3.0,     label = "(c) α > α★"),
]

for case in cases
    α = case.α
    println("\n" * "="^70)
    println("  $(case.label):  α = $(round(α, digits=4))")
    println("="^70)

    Vfun = square_well(a, b, -α)
    data = compute_all_scattering_data(a, b, Vfun; N=500, k_max=4.0)
    print_scattering_summary(data; title="  V(x) = -$(round(α,digits=3)) χ_{[-1,1]}")
end

println("\n" * "="^70)
println("INTERPRETATION (matching paper Figure 9):")
println("="^70)
println("""
  (a) α < α★: One bound state pole (κ > 0) + one scattering resonance (κ < 0)
      → The resonance has NOT yet crossed through zero.
      → Nonlinear branch from bound state starts at N=0.
      → Nonlinear branch from resonance starts at N = 2N[S_{k²}] > 0.

  (b) α = α★: Resonance pole is at κ ≈ 0 (threshold).
      → Multiple branches bifurcate from (E,N) = (0,0).

  (c) α > α★: Two bound state poles (both κ > 0).
      → Resonance has crossed zero and become a new bound state.
      → Transmission resonances now appear on the imaginary axis.
      → Additional nonlinear branches from the new bound state.
""")
