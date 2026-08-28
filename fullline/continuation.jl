#####################
# continuation.jl   #
#####################

"""
Seed finding and BifurcationKit continuation for NLS homoclinics.

Workflow:
1. find_all_seeds(a, b, Vfun; E_list=...) — scan for seeds
2. Inspect / filter / deduplicate seeds
3. continue_from_seeds(seeds, a, b, Vfun; ...) — continue branches
"""

using Printf
using Accessors: @optic
using BifurcationKit: AutoSwitch, PALC, Natural, BifurcationProblem, ContinuationPar, NewtonPar, continuation
using OrdinaryDiffEq
using LinearAlgebra

# ============================================================================
# SEED FINDING
# ============================================================================

"""
    find_seeds_at_E(a, b, Vfun; E0=-1.0, N=1000, ζmax=6.0, nscan=3400,
                    tolH=1e-10, slope_set=(+1,-1))

Scan ζ ∈ (0, ζmax] for zeros of the Hamiltonian residual at x=b,
for fixed E = E0. Returns a vector of seed tuples

    (; ζ, c, slope_sign, p=(E=E0,))

suitable for continuation.
"""
function find_seeds_at_E(a, b, Vfun;
                         E0=-1.0, N=1000, ζmax=6.0, nscan=3400,
                         tolH=1e-10, slope_set=(+1, -1))
    seeds = []
    ζgrid = range(1e-3, ζmax; length=nscan)

    for slope_sign in slope_set
        # Evaluate residual on a coarse grid
        R = [H_residual_ζ(a, b, E0, Vfun;
                          ζ=ζ, slope_sign=slope_sign, N=N) for ζ in ζgrid]

        # Look for sign changes or near-zeros
        for i in 1:(length(ζgrid) - 1)
            ζ1, ζ2 = ζgrid[i], ζgrid[i+1]
            r1, r2 = R[i], R[i+1]

            if !isfinite(r1) || !isfinite(r2)
                continue
            end

            take = false
            ζstar = NaN

            if abs(r1) ≤ tolH
                ζstar = ζ1
                take = true
            elseif abs(r2) ≤ tolH
                ζstar = ζ2
                take = true
            elseif r1 * r2 < 0
                # refine by bisection
                fζ = ζ -> H_residual_ζ(a, b, E0, Vfun;
                                       ζ=ζ, slope_sign=slope_sign, N=N)
                ζstar = bisect_zero(fζ, ζ1, ζ2; tol=tolH)
                take = true
            end

            if take
                cstar = c_from_ζ(ζstar, E0)
                # de-duplicate nearby seeds in c
                if all(abs(cstar - c_from_ζ(s.ζ, E0)) > 1e-6 for s in seeds)
                    push!(seeds, (; ζ=ζstar,
                                   c=cstar,
                                   slope_sign=slope_sign,
                                   p=(E=E0,)))
                end
            end
        end
    end

    return seeds
end

"""
    find_all_seeds(a, b, Vfun;
                   E_list::Vector{Float64},
                   N=1000, ζmax=20.0, nscan=3400,
                   tolH=1e-6, slope_set=(+1,-1))

Find all seeds across multiple energy values.

Returns a vector of named tuples, each containing:
- `ζ`: the ζ parameter value
- `c`: amplitude c = √(-2E) tanh(ζ)
- `slope_sign`: ±1
- `p`: parameter tuple (E=E,) for BifurcationKit
"""
function find_all_seeds(a, b, Vfun;
                        E_list::Vector{Float64},
                        N=1000,
                        ζmax=20.0,
                        nscan=3400,
                        tolH=1e-6,
                        slope_set=(+1, -1))

    all_seeds = []

    println("\n" * "="^70)
    println("SEED FINDING")
    println("="^70)

    for E0 in E_list
        println("\nSearching at E = $E0...")

        seeds_at_E = find_seeds_at_E(a, b, Vfun;
                                     E0=E0,
                                     N=N,
                                     ζmax=ζmax,
                                     nscan=nscan,
                                     tolH=tolH,
                                     slope_set=slope_set)

        println("  Found $(length(seeds_at_E)) seed(s)")
        for seed in seeds_at_E
            println("    ζ=$(round(seed.ζ, digits=4)), " *
                   "c=$(round(seed.c, digits=5)), " *
                   "slope=$(seed.slope_sign)")
        end

        append!(all_seeds, seeds_at_E)
    end

    println("\n" * "="^70)
    println("TOTAL: $(length(all_seeds)) seeds found")
    println("="^70)

    return all_seeds
end


"""
    print_seed_table(seeds; sort_by=:E)

Print a nicely formatted table of all seeds.

`sort_by` can be `:E`, `:c`, or `:ζ`
"""
function print_seed_table(seeds; sort_by=:E)
    if isempty(seeds)
        println("No seeds to display")
        return
    end

    # Helper to extract E value (stored in p field)
    get_E(s) = try s.p.E catch; s.p end

    # Sort seeds
    sorted = if sort_by == :E
        sort(seeds, by=s->get_E(s))
    elseif sort_by == :c
        sort(seeds, by=s->s.c)
    elseif sort_by == :ζ
        sort(seeds, by=s->s.ζ)
    else
        seeds
    end

    println("\n" * "="^70)
    println("SEED TABLE (sorted by $sort_by)")
    println("="^70)
    println("Index | Energy (E) |  Amplitude (c) |    ζ      | Slope")
    println("-"^70)

    for (i, seed) in enumerate(sorted)
        E_val = get_E(seed)
        slope_str = seed.slope_sign > 0 ? "+1" : "-1"
        @printf("%5d | %10.5f | %14.6e | %9.5f | %5s\n",
                i, E_val, seed.c, seed.ζ, slope_str)
    end

    println("="^70)
end


"""
    filter_seeds(seeds;
                 E_min=-Inf, E_max=Inf,
                 c_min=0.0, c_max=Inf,
                 ζ_min=0.0, ζ_max=Inf,
                 slope_signs=nothing)

Filter seeds by various criteria.
"""
function filter_seeds(seeds;
                      E_min=-Inf, E_max=Inf,
                      c_min=0.0, c_max=Inf,
                      ζ_min=0.0, ζ_max=Inf,
                      slope_signs=nothing)

    # Helper to extract E value (stored in p field)
    get_E(s) = try s.p.E catch; s.p end

    filtered = filter(seeds) do s
        # Energy filter
        E_val = get_E(s)
        (E_min ≤ E_val ≤ E_max) || return false

        # Amplitude filter
        (c_min ≤ s.c ≤ c_max) || return false

        # ζ filter
        (ζ_min ≤ s.ζ ≤ ζ_max) || return false

        # Slope filter
        if slope_signs !== nothing
            (s.slope_sign in slope_signs) || return false
        end

        return true
    end

    println("Filtered: $(length(seeds)) → $(length(filtered)) seeds")
    return filtered
end


"""
    deduplicate_seeds(seeds; E_tol=1e-3, c_tol=1e-4)

Remove duplicate seeds that are very close in (E, c) space.
"""
function deduplicate_seeds(seeds; E_tol=1e-3, c_tol=1e-4)
    isempty(seeds) && return seeds

    # Helper to extract E value (stored in p field)
    get_E(s) = try s.p.E catch; s.p end

    unique_seeds = [seeds[1]]

    for seed in seeds[2:end]
        is_duplicate = false
        E_seed = get_E(seed)

        for existing in unique_seeds
            E_existing = get_E(existing)

            if abs(E_seed - E_existing) < E_tol &&
               abs(seed.c - existing.c) < c_tol &&
               seed.slope_sign == existing.slope_sign
                is_duplicate = true
                break
            end
        end

        if !is_duplicate
            push!(unique_seeds, seed)
        end
    end

    if length(unique_seeds) < length(seeds)
        println("Removed $(length(seeds) - length(unique_seeds)) duplicate(s)")
    end

    return unique_seeds
end


# ============================================================================
# BRANCH CONTINUATION
# ============================================================================

"""
    continue_from_seeds(seeds, a, b, Vfun;
                       N=7000, p_min=-10.0, p_max=-1e-12,
                       ds=0.01, dsmin=1e-4, dsmax=0.01,
                       max_steps=500, tol=1e-5, ζ_min=1e-3,
                       verbose=1)

Continue branches from a collection of seeds.

Returns vector of branch objects (from BifurcationKit).
"""
function continue_from_seeds(seeds, a, b, Vfun;
                            N=7000,
                            p_min=-10.0,
                            p_max=-1e-12,
                            ds=0.01,
                            dsmin=1e-4,
                            dsmax=0.01,
                            max_steps=500,
                            tol=1e-5,
                            ζ_min=1e-3,
                            verbose=1)

    branches = []

    println("\n" * "="^70)
    println("BRANCH CONTINUATION")
    println("="^70)
    println("Continuing $(length(seeds)) seed(s)...")

    for (idx, sol0) in enumerate(seeds)
        # Extract E value from p field
        E_val = try sol0.p.E catch; sol0.p end

        verbose > 0 && println("\n--- Seed $idx / $(length(seeds)) ---")
        verbose > 0 && println("E=$(E_val), c=$(sol0.c), slope=$(sol0.slope_sign)")

        # Continuation parameter lens
        lensE = @optic _.E

        # Residual function
        F(ζ, p) = begin
            E_val = typeof(p) <: Number ? p : p.E
            return @inbounds [H_residual_ζ(a, b, E_val, Vfun;
                                          ζ=ζ[1],
                                          slope_sign=sol0.slope_sign,
                                          N=N)]
        end

        # Record function
        rec(ζ, p; k...) = begin
            E_val = typeof(p) <: Number ? p : p.E
            return (; ζ=ζ[1],
                     c=c_from_ζ(ζ[1], E_val),
                     slope_sign=sol0.slope_sign)
        end

        # Set up problem
        prob = BifurcationProblem(F, [sol0.ζ], sol0.p, lensE;
                                 record_from_solution=rec)

        # Continuation options
        opts = ContinuationPar(
            ds=ds,
            dsmin=dsmin,
            dsmax=dsmax,
            p_min=p_min,
            p_max=p_max,
            max_steps=max_steps,
            newton_options=NewtonPar(
                tol=tol,
                max_iterations=30,
            ),
            nev=1,
            detect_bifurcation=0,
        )

        # Stopping criterion
        last_br = Ref{Any}(nothing)

        finalise_solution = (z, tau, step, contResult; k...) -> begin
            last_br[] = contResult

            # Detect solver failures
            if any(isnan, z.u) || any(isinf, z.u)
                @warn "Stopping one side early: encountered NaN/Inf in solution."
                return false
            end

            ζ_val = z.u[1]

            # Allow first few steps
            if step ≤ 1
                return true
            end

            # Stop if |ζ| too small
            if abs(ζ_val) ≤ ζ_min
                @info "Reached ζ_min threshold — stopping this side only."
                return false
            end

            return true
        end


        # Run continuation with error handling
        br = try
            continuation(prob, PALC(), opts;
                        bothside=true,
                        verbosity=verbose,
                        finalise_solution=finalise_solution)
        catch e
            @warn "Continuation failed for seed $idx: $e"
            if last_br[] !== nothing
                verbose > 0 && println("Using partial branch")
                last_br[]
            else
                verbose > 0 && println("No partial branch available")
                nothing
            end
        end

        if br !== nothing
            n_pts = length(br.branch)
            E_range = if n_pts > 0
                Es = [sol.param for sol in br.branch]
                (minimum(Es), maximum(Es))
            else
                (NaN, NaN)
            end
            verbose > 0 && println("Branch completed: $n_pts points, E in $E_range")
        else
            verbose > 0 && println("Branch failed")
        end

        push!(branches, br === nothing ? (; branch=[]) : br)
    end

    # Summary
    n_success = sum(!isempty(br.branch) for br in branches)
    println("\n" * "="^70)
    println("CONTINUATION COMPLETE")
    println("Successful: $n_success / $(length(seeds))")
    println("="^70)

    return branches
end


"""
    continue_single_seed(seed, a, b, Vfun; kwargs...)

Continue a single seed (convenience wrapper).
"""
function continue_single_seed(seed, a, b, Vfun; kwargs...)
    branches = continue_from_seeds([seed], a, b, Vfun; kwargs...)
    return branches[1]
end


# ============================================================================
# CONVENIENCE: Combined workflow
# ============================================================================

"""
    find_branches_at_fixed_E(a, b, Vfun; E0=-1.0, ...)

Combines seed finding + continuation at one E.
"""
function find_branches_at_fixed_E(a, b, Vfun;
                                 E0=-1.0,
                                 N=1000,
                                 tolH=1e-10,
                                 ζmax=3.0,
                                 nscan=3400,
                                 p_min=-10.0,
                                 kwargs...)

    # Find seeds at this E
    seeds = find_all_seeds(a, b, Vfun;
                          E_list=[E0],
                          N=N,
                          ζmax=ζmax,
                          nscan=nscan,
                          tolH=tolH)

    # Continue them
    branches = continue_from_seeds(seeds, a, b, Vfun;
                                  N=N,
                                  p_min=p_min,
                                  kwargs...)

    return branches, seeds
end


"""
    find_branches_over_Es(a, b, Vfun; E0_list::Vector{Float64}, ...)

Combines seed finding + continuation over multiple E.
"""
function find_branches_over_Es(a, b, Vfun;
                              E0_list::Vector{Float64},
                              N=1000,
                              tolH=1e-8,
                              ζmax=8.0,
                              nscan=3400,
                              p_min=-10.0,
                              kwargs...)

    # Find all seeds
    seeds = find_all_seeds(a, b, Vfun;
                          E_list=E0_list,
                          N=N,
                          ζmax=ζmax,
                          nscan=nscan,
                          tolH=tolH)

    # Continue them all
    branches = continue_from_seeds(seeds, a, b, Vfun;
                                  N=N,
                                  p_min=p_min,
                                  kwargs...)

    return branches
end


# ============================================================================
# SCATTERING-INFORMED BRANCH DISCOVERY
# ============================================================================

"""
    find_all_branches_from_scattering(a, b, Vfun;
        N_scat=400, N_cont=7000, k_max=5.0,
        nscan=4000, ζmax=12.0, p_min=-10.0, max_steps=500,
        kwargs...)

Use the linear scattering data to systematically find ALL nonlinear branches:

1. Compute scattering data (bound poles, resonance poles, transmission zeros)
2. For each pole/zero at k=iκ, seed the continuation at E near -κ²
3. Scan both slope signs to capture asymmetric branches

This is the recommended workflow for reproducing Figures 7-10 of the paper.
"""
function find_all_branches_from_scattering(a, b, Vfun;
        N_scat::Int=400, N_cont::Int=7000, k_max::Real=5.0,
        nscan::Int=4000, ζmax::Real=12.0,
        p_min::Real=-10.0, max_steps::Int=500,
        verbose::Int=1, kwargs...)

    println("\n" * "="^70)
    println("SCATTERING-INFORMED BRANCH DISCOVERY")
    println("="^70)

    # Step 1: Compute scattering data
    println("\nStep 1: Computing scattering data...")
    scat_data = compute_all_scattering_data(a, b, Vfun; N=N_scat, k_max=k_max, verbose=true)

    # Step 2: Build list of energies to scan
    # Include energies near each pole/zero, plus a denser grid nearby
    E_scan = Float64[]

    for r in scat_data.bound_poles
        E_bif = r.E  # E = -κ² for k = iκ
        # Scan slightly below the bifurcation energy
        push!(E_scan, E_bif - 0.05)
        push!(E_scan, E_bif - 0.2)
        push!(E_scan, E_bif - 0.5)
    end

    for r in scat_data.scat_poles
        E_bif = r.E
        push!(E_scan, E_bif - 0.05)
        push!(E_scan, E_bif - 0.2)
        push!(E_scan, E_bif + 0.05)
    end

    for r in scat_data.trans_res
        E_bif = real(r.k^2)
        if E_bif < 0
            push!(E_scan, E_bif - 0.05)
            push!(E_scan, E_bif - 0.2)
            push!(E_scan, E_bif + 0.05)
        end
    end

    # Add a uniform grid for good measure
    append!(E_scan, range(-0.1, min(p_min, -3.0); length=5))

    # Filter and deduplicate
    E_scan = sort(unique(filter(e -> p_min ≤ e < -1e-4, E_scan)))

    println("\nStep 2: Scanning at $(length(E_scan)) energies")
    println("  E_scan = ", round.(E_scan; digits=3))

    # Step 3: Find seeds at all energies (both slope signs)
    all_seeds = find_all_seeds(a, b, Vfun;
        E_list=E_scan,
        N=N_cont,
        ζmax=ζmax,
        nscan=nscan,
        tolH=1e-7,
        slope_set=(+1, -1))

    all_seeds = deduplicate_seeds(all_seeds; E_tol=0.01, c_tol=0.01)
    println("\nStep 3: Found $(length(all_seeds)) unique seeds")

    # Step 4: Continue all seeds
    println("\nStep 4: Continuing branches...")
    branches = continue_from_seeds(all_seeds, a, b, Vfun;
        N=N_cont,
        p_min=p_min,
        max_steps=max_steps,
        verbose=verbose,
        kwargs...)

    # Step 5: Deduplicate branches (remove branches that overlap in E-N space)
    branches = _deduplicate_branches(branches, a, b, Vfun; N=N_cont)

    println("\n" * "="^70)
    println("DISCOVERY COMPLETE: $(count(!isempty(br.branch) for br in branches)) branches")
    println("="^70)

    return branches, scat_data
end

"""Remove branches that are essentially the same curve (based on endpoint matching)."""
function _deduplicate_branches(branches, a, b, Vfun; N=5000, tol=0.05)
    unique_branches = []

    for br in branches
        isempty(br.branch) && continue
        length(br.branch) < 3 && continue

        # Characterize by (E_min, E_max, slope_sign)
        Es = [sol.param for sol in br.branch]
        E_lo, E_hi = extrema(Es)
        slope = br.branch[1].slope_sign

        is_dup = false
        for ubr in unique_branches
            isempty(ubr.branch) && continue
            Es2 = [sol.param for sol in ubr.branch]
            E_lo2, E_hi2 = extrema(Es2)
            slope2 = ubr.branch[1].slope_sign

            # Same slope sign and overlapping E range
            if slope == slope2 &&
               abs(E_lo - E_lo2) < tol * max(abs(E_lo), 1) &&
               abs(E_hi - E_hi2) < tol * max(abs(E_hi), 1)
                is_dup = true
                break
            end
        end

        if !is_dup
            push!(unique_branches, br)
        end
    end

    if length(unique_branches) < count(!isempty(br.branch) for br in branches)
        n_removed = count(!isempty(br.branch) for br in branches) - length(unique_branches)
        println("  Removed $n_removed duplicate branch(es)")
    end

    return unique_branches
end
