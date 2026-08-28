###############################################
# resonances.jl - Linear resonance computation
###############################################
#
# Finds scattering resonances of H_V = -d²/dx² + V(x) on the half-line
# with Dirichlet BC at x=0 and outgoing radiation BC at x=b:
#
#   u(0) = 0
#   u'(b) = ik u(b)     (outgoing)
#
# Zeros of w(k) with Im(k) > 0 give bound state poles (E = k² < 0).
# Zeros with Im(k) < 0 give scattering resonance poles (anti-bound states).
# On the imaginary axis: k = iκ with κ > 0 (bound) or κ < 0 (resonance).
#
# Method: Ghost-point FD discretization yields a quadratic eigenvalue
# problem in k, linearized to a 2N×2N companion matrix.
#
###############################################

using LinearAlgebra, OrdinaryDiffEq
using OrdinaryDiffEq.SciMLBase: ReturnCode

# =============================================================================
# COMPANION EIGENVALUE PROBLEM
# =============================================================================

"""
    compute_resonances(b, Vfun; N=400, k_max=10.0, tol=1e-6)

Find resonances of H_V on the half-line [0,b] with:
  - Dirichlet BC: u(0) = 0
  - Outgoing radiation BC: u'(b) = ik u(b)

Returns a vector of NamedTuples: (; k, κ, E, type, U_at_b)
where type ∈ {:bound, :resonance, :threshold}.
"""
function compute_resonances(b, Vfun; N::Int=400, k_max::Real=10.0, tol::Real=1e-6)
    h = b / (N + 1)
    # Interior grid: x_j = j*h for j = 1, ..., N
    x = [j * h for j in 1:N]

    # Build FD matrix A for -d²/dx² + V on interior with Dirichlet at x=0
    # and ghost-point handling at x=b.
    #
    # Standard tridiagonal: (-u_{j-1} + 2u_j - u_{j+1}) / h² + V(x_j) u_j
    # At j=N (last interior point, x_N = N*h):
    #   u_{N+1} is the ghost point. The outgoing BC u'(b) = ik u(b) gives:
    #   (u_{N+1} - u_{N-1}) / (2h) = ik u_N
    #   => u_{N+1} = u_{N-1} + 2ih k u_N
    #
    # The row-N equation becomes:
    #   (-u_{N-1} + 2u_N - (u_{N-1} + 2ihk u_N)) / h² + V(x_N) u_N = k² u_N
    #   (-2u_{N-1} + (2 - 2ihk) u_N) / h² + V(x_N) u_N = k² u_N
    #
    # This yields: A u + ik B u = k² u  (quadratic EVP in k)

    # Matrix A: standard FD Laplacian + V, with modified last row
    A = zeros(ComplexF64, N, N)

    # Fill tridiagonal part (rows 1 to N-1)
    for j in 1:N
        A[j, j] = 2.0 / h^2 + Vfun(x[j])
        if j > 1
            A[j, j-1] = -1.0 / h^2
        end
        if j < N
            A[j, j+1] = -1.0 / h^2
        end
    end

    # Modify row N for the ghost-point substitution
    # Original: (-u_{N-1} + 2u_N - u_{N+1})/h² + V u_N
    # With ghost: u_{N+1} = u_{N-1} + 2ihk u_N
    # => (-u_{N-1} + 2u_N - u_{N-1} - 2ihk u_N)/h² + V u_N
    # = (-2u_{N-1} + (2 - 2ihk) u_N)/h² + V u_N
    A[N, N]   = 2.0 / h^2 + Vfun(x[N])   # the k-independent part
    if N > 1
        A[N, N-1] = -2.0 / h^2            # coefficient of u_{N-1} is -2/h²
    end

    # Matrix B: rank-1 correction from the ik term
    # The ik-linear term: -(2ih/h²) = -2i/h contributes to row N, column N
    # In the equation: A u + ik B u = k² u
    # The k-linear part of the row-N diagonal is: -2ih/(h²) * k * u_N = ik * (-2/h) * u_N
    # Wait, let me redo this carefully.
    #
    # Row N equation (after ghost substitution):
    #   (-2 u_{N-1} + 2 u_N) / h² + V(x_N) u_N - (2ik / h) u_N = k² u_N
    #
    # So: [A₀ u]_N - (2ik/h) u_N = k² u_N
    # where A₀ is the k-independent matrix (already built above with the -2/h² off-diagonal).
    #
    # Rewrite: A₀ u + ik * B u = k² u
    # with B[N,N] = -2/h, all other entries zero.

    B = zeros(ComplexF64, N, N)
    B[N, N] = -2.0 / h

    # Linearize the QEP: A u + ik B u = k² u
    # Let w = k u, then:
    #   [0   I ] [u]     [u]
    #   [A  iB ] [w] = k [w]
    #
    # This is a 2N×2N standard eigenvalue problem.

    C = zeros(ComplexF64, 2N, 2N)
    C[1:N, (N+1):(2N)] = I(N)
    C[(N+1):(2N), 1:N] = A
    C[(N+1):(2N), (N+1):(2N)] = im * B

    # Solve
    F = eigen(C)
    k_vals = F.values

    # Filter: keep resonances with |k| < k_max
    results = []
    for (i, k_val) in enumerate(k_vals)
        abs(k_val) > k_max && continue
        abs(real(k_val)) > tol && continue  # only purely imaginary (on iR)

        κ_val = imag(k_val)  # k = iκ => κ = Im(k)
        E_val = k_val^2      # E = k²

        # Classify
        if κ_val > tol
            type = :bound
        elseif κ_val < -tol
            type = :resonance
        else
            type = :threshold
        end

        # Extract eigenfunction (upper N components)
        u_vec = F.vectors[1:N, i]
        # Normalize so u(b) ≈ u[N] (last interior point) = 1
        if abs(u_vec[N]) > 1e-15
            u_vec ./= u_vec[N]
        end

        push!(results, (;
            k = k_val,
            κ = κ_val,
            E = real(E_val),
            type = type,
            u_vec = u_vec,
            U_at_b = u_vec[N]
        ))
    end

    # Sort by |κ| (closest to real axis first)
    sort!(results; by = r -> abs(r.κ))

    return results
end

"""
    compute_resonances_all(b, Vfun; N=400, k_max=10.0)

Compute all resonances (not restricted to imaginary axis).
Returns resonances with Im(k) < 0 (lower half-plane).
"""
function compute_resonances_all(b, Vfun; N::Int=400, k_max::Real=10.0, tol::Real=1e-6)
    h = b / (N + 1)
    x = [j * h for j in 1:N]

    A = zeros(ComplexF64, N, N)
    for j in 1:N
        A[j, j] = 2.0 / h^2 + Vfun(x[j])
        if j > 1
            A[j, j-1] = -1.0 / h^2
        end
        if j < N
            A[j, j+1] = -1.0 / h^2
        end
    end
    A[N, N] = 2.0 / h^2 + Vfun(x[N])
    if N > 1
        A[N, N-1] = -2.0 / h^2
    end

    B = zeros(ComplexF64, N, N)
    B[N, N] = -2.0 / h

    C = zeros(ComplexF64, 2N, 2N)
    C[1:N, (N+1):(2N)] = I(N)
    C[(N+1):(2N), 1:N] = A
    C[(N+1):(2N), (N+1):(2N)] = im * B

    F = eigen(C)
    k_vals = F.values

    results = []
    for (i, k_val) in enumerate(k_vals)
        abs(k_val) > k_max && continue
        imag(k_val) > tol && continue  # keep only lower half-plane + real axis

        push!(results, (;
            k = k_val,
            κ = imag(k_val),
            E = real(k_val^2),
            u_vec = F.vectors[1:N, i]
        ))
    end

    sort!(results; by = r -> abs(imag(r.k)))
    return results
end

# =============================================================================
# NEWTON REFINEMENT OF RESONANCES
# =============================================================================

"""
    refine_resonance_newton(b, Vfun, γ_init; maxiter=20, tol=1e-12)

Newton refinement of a resonance on the imaginary axis.
Given initial γ (where k = iγ, so k² = -γ²), refine by solving:

    f(γ) = U'(b; γ) - γ U(b; γ) = 0

where U solves -U'' + V U = -γ² U with U(0) = 0, U'(0) = 1.

Returns refined γ value.
"""
function refine_resonance_newton(b, Vfun, γ_init; maxiter::Int=20, tol::Real=1e-12)
    γ = γ_init

    for iter in 1:maxiter
        # Solve linear IVP at current γ
        u0 = [0.0, 1.0]  # U(0) = 0, U'(0) = 1
        E_lin = -γ^2

        function f_lin!(du, u, p, x)
            du[1] = u[2]
            du[2] = (Vfun(x) - E_lin) * u[1]
        end

        prob = ODEProblem(f_lin!, u0, (0.0, b))
        sol = solve(prob, Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false, save_start=false)

        if sol.retcode != ReturnCode.Success
            @warn "ODE solve failed at γ = $γ"
            return γ
        end

        Ub  = sol.u[end][1]
        Upb = sol.u[end][2]

        # Residual: f(γ) = U'(b) - γ U(b)
        f_val = Upb - γ * Ub

        # Forward finite difference for f'(γ)
        δ = max(1e-6 * abs(γ), 1e-10)
        γp = γ + δ
        E_lin_p = -γp^2

        function f_linp!(du, u, p, x)
            du[1] = u[2]
            du[2] = (Vfun(x) - E_lin_p) * u[1]
        end

        prob_p = ODEProblem(f_linp!, u0, (0.0, b))
        sol_p = solve(prob_p, Tsit5(); reltol=1e-13, abstol=1e-15, save_everystep=false, save_start=false)

        Ub_p  = sol_p.u[end][1]
        Upb_p = sol_p.u[end][2]
        f_val_p = Upb_p - γp * Ub_p

        fprime = (f_val_p - f_val) / δ

        if abs(fprime) < 1e-20
            @warn "Newton: zero derivative at γ = $γ"
            return γ
        end

        Δγ = -f_val / fprime
        γ += Δγ

        if abs(Δγ) < tol * abs(γ)
            return γ
        end
    end

    @warn "Newton did not converge after $maxiter iterations"
    return γ
end

# =============================================================================
# RESONANCE EIGENFUNCTION AND INTEGRALS
# =============================================================================

"""
    compute_resonance_mode(b, Vfun, γ; Npts=2001)

Compute the resonance eigenfunction U_★(x) with:
  -U'' + V U = -γ² U,   U(0) = 0, U'(0) = 1

Returns (x, U, U') on [0, b].
"""
function compute_resonance_mode(b, Vfun, γ; Npts::Int=2001)
    E_lin = -γ^2
    u0 = [0.0, 1.0]

    function f!(du, u, p, x)
        du[1] = u[2]
        du[2] = (Vfun(x) - E_lin) * u[1]
    end

    prob = ODEProblem(f!, u0, (0.0, b))
    sol = solve(prob, Tsit5(); reltol=1e-13, abstol=1e-15,
                saveat=range(0.0, b; length=Npts))

    x = sol.t
    U = [sol.u[i][1] for i in eachindex(sol.u)]
    Up = [sol.u[i][2] for i in eachindex(sol.u)]

    return x, U, Up
end

"""
    compute_bifurcation_coefficients(b, Vfun, γ; Npts=2001)

Compute the asymptotic bifurcation coefficients from the resonance mode U_★:

  I2 = ∫₀ᵇ U_★² dx
  I4 = ∫₀ᵇ U_★⁴ dx
  U_b = U_★(b)

For the threshold resonance case (γ → 0):
  E(ε) ≈ -½ U_★(b)² ε
  x_R(ε) → b + (1/U_b²) ∫₀ᵇ U² dx - (2/U_b²) ∫₀ᵇ U⁴ dx

Returns a NamedTuple with all coefficients.
"""
function compute_bifurcation_coefficients(b, Vfun, γ; Npts::Int=2001)
    x, U, Up = compute_resonance_mode(b, Vfun, γ; Npts=Npts)

    U_b = U[end]
    Up_b = Up[end]

    # Trapezoidal rule
    dx = x[2] - x[1]
    I2 = dx * (sum(U.^2) - 0.5*(U[1]^2 + U[end]^2))
    I4 = dx * (sum(U.^4) - 0.5*(U[1]^4 + U[end]^4))

    # For threshold resonance (Theorem 4.4):
    # E(ε) = -½ U_★(b)² ε + O(ε²)
    # x_R(ε) → b + (1/U_b²) I2 - (2/U_b²) I4
    dE_dε = -0.5 * U_b^2
    xR_limit = b + I2 / U_b^2 - 2 * I4 / U_b^2

    # For non-zero resonance (Theorem 4.1):
    # Ω = U_b⁴/(2γ) - 2 I4
    # ν₀ = 3Ω/(4γ)
    # A = I2 - U_b²/(2γ)
    if abs(γ) > 1e-10
        Ω = U_b^4 / (2γ) - 2 * I4
        ν₀ = 3Ω / (4γ)
        A_coeff = I2 - U_b^2 / (2γ)
        dNdE_pred = -2/γ + 2 * A_coeff^2 / Ω
    else
        Ω = NaN
        ν₀ = NaN
        A_coeff = NaN
        dNdE_pred = NaN
    end

    return (;
        γ = γ,
        E_bif = -γ^2,
        U_b = U_b,
        Up_b = Up_b,
        I2 = I2,
        I4 = I4,
        dE_dε = dE_dε,
        xR_limit = xR_limit,
        Ω = Ω,
        ν₀ = ν₀,
        A = A_coeff,
        dNdE_pred = dNdE_pred
    )
end

# =============================================================================
# PRINTING
# =============================================================================

"""Print a formatted table of resonance results."""
function print_resonances(results; title="Resonances of H_V")
    println("\n" * "="^70)
    println(title)
    println("="^70)
    println("  #  |   type     |      k        |      κ       |     E = k²    ")
    println("-"^70)
    for (i, r) in enumerate(results)
        type_str = String(r.type)
        @printf("  %2d | %-10s | %12.6f i | %12.6f | %12.6f\n",
                i, type_str, imag(r.k), r.κ, r.E)
    end
    println("-"^70)
end
