##############
# plots.jl   #
##############

# =============================================================================
# STYLE DEFAULTS
# =============================================================================

"""
    set_plot_style!()

Set global Plots defaults (Computer Modern fonts, box frame, subtle grid).
Call once at the start of a plotting session.
"""
function set_plot_style!()
    default(
        tickfont   = font(10, "Computer Modern"),
        guidefont  = font(13, "Computer Modern"),
        legendfont = font(10, "Computer Modern"),
        titlefont  = font(14, "Computer Modern"),
        framestyle = :box,
        grid       = true,
        gridalpha  = 0.25,
        gridcolor  = :gray85,
        lw         = 2,
        margin     = 5Plots.mm,
    )
end

# =============================================================================
# MASS-ENERGY PLOTS
# =============================================================================

"""
    plot_mass_energy(branches, a, b, Vfun; ...)

Plot L² and H¹ norms vs E for all branches with per-branch colors,
endpoint markers, E=0 reference line, and auto-zoomed views.

Returns `(pltL2, pltH1, pltL2_zoom, pltH1_zoom)`.
"""
function plot_mass_energy(branches, a, b, Vfun;
                          Ngrid=7000,
                          plotEmin=1.0,
                          l2max=nothing,
                          h1max=nothing)

    set_plot_style!()

    # --- Collect norm data per branch ---
    all_Es = Float64[]
    all_Ns = Float64[]
    all_Hs = Float64[]
    branch_data = []

    colors = distinguishable_colors(max(length(branches), 1),
                                    [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue

        Es, Ns, Hs = Float64[], Float64[], Float64[]
        for sol in br.branch
            E, c, slope_sign = sol.param, sol.c, sol.slope_sign
            x, u, v = integrate_support(a, b, E, Vfun;
                                        N=Ngrid, c=c, slope_sign=slope_sign)
            isempty(x) && continue
            N_val = compute_norm(a, b, E, x, u, v)
            H_val = compute_H1_norm(a, b, E, x, u, v)
            if isfinite(N_val) && isfinite(H_val)
                push!(Es, E); push!(Ns, N_val); push!(Hs, H_val)
                push!(all_Es, E); push!(all_Ns, N_val); push!(all_Hs, H_val)
            end
        end
        push!(branch_data, (; Es, Ns, Hs, color=colors[i], idx=i))
    end

    # --- Axis limits ---
    if l2max === nothing && !isempty(all_Ns)
        l2max = 1.15 * maximum(all_Ns)
    end
    if h1max === nothing && !isempty(all_Hs)
        h1max = 1.15 * maximum(all_Hs)
    end
    l2max = something(l2max, 10.0)
    h1max = something(h1max, 30.0)

    # --- Helper to build one mass-energy plot ---
    function _build_plot(ylabel_str, title_str, norm_key, ylim_top, xlims_range)
        plt = plot(;
            xlabel = L"E",
            ylabel = ylabel_str,
            legend = :best,
            xlims  = xlims_range,
            ylims  = (0, ylim_top),
            size   = (550, 350),
            title  = title_str,
        )

        for data in branch_data
            isempty(data.Es) && continue
            ys = getfield(data, norm_key)

            plot!(plt, data.Es, ys;
                  lw=2.5, color=data.color,
                  label="Branch $(data.idx)")

            # Endpoint markers
            scatter!(plt, [data.Es[1], data.Es[end]],
                     [ys[1], ys[end]];
                     marker=:circle, ms=5, color=data.color,
                     markerstrokewidth=1.5, markerstrokecolor=:black,
                     label="")
        end

        # E = 0 reference line
        vline!(plt, [0.0]; color=:black, ls=:dash, lw=1, alpha=0.4, label="")

        return plt
    end

    # --- Full-range plots ---
    Emin_abs = abs(plotEmin)   # robust to either sign convention
    pltL2 = _build_plot(
        L"\mathcal{N}[\psi_E]",
        "Mass vs Energy",
        :Ns, l2max, (-Emin_abs, 0.05))

    pltH1 = _build_plot(
        L"\|\psi_E\|_{H^1}^2",
        L"H^1\textrm{-norm vs Energy}",
        :Hs, h1max, (-Emin_abs, 0.05))

    # --- Zoomed plots (auto-fitted to data extent) ---
    if !isempty(all_Es)
        E_lo = minimum(all_Es)
        E_hi = maximum(all_Es)
        E_pad = max(0.05 * (E_hi - E_lo), 0.02)
        zoom_xlims = (E_lo - E_pad, E_hi + E_pad)
        l2_zoom_max = 1.1 * l2max
        h1_zoom_max = 1.1 * h1max
    else
        zoom_xlims = (-plotEmin, 0.05)
        l2_zoom_max = l2max
        h1_zoom_max = h1max
    end

    pltL2_zoom = _build_plot(
        L"\mathcal{N}[\psi_E]",
        "Mass vs Energy (zoomed)",
        :Ns, l2_zoom_max, zoom_xlims)

    pltH1_zoom = _build_plot(
        L"\|\psi_E\|_{H^1}^2",
        L"H^1\textrm{-norm (zoomed)}",
        :Hs, h1_zoom_max, zoom_xlims)

    display(pltL2)
    display(pltH1)
    return pltL2, pltH1, pltL2_zoom, pltH1_zoom
end


# =============================================================================
# PROFILE PLOTS
# =============================================================================

"""
    plot_branch_profiles(branches, idxs;
                         a=-1, b=1, Vfun=nothing, Ngrid=7000,
                         Xmax=13.0, ymax=nothing, n_profiles=18)

Plot psi_E(x) profiles for each branch in its own subplot.

`idxs` can be:
- `:auto` — plot all non-empty branches (one subplot each)
- `Vector{Int}` — plot only these branch indices

Profile selection (per branch):
1. Mandatory: first point, last point, argmin(E), argmax(E), turning points
2. Remaining slots filled with evenly spaced intermediates
3. Deduplicated by rounded E

Mandatory profiles are drawn bold (lw=3); others are thinner (lw=1.8).
Viridis color gradient encodes progression along the branch.
"""
function plot_branch_profiles(branches, idxs;
                              a=-1, b=1, Vfun=nothing,
                              Ngrid=7000,
                              Xmax=13.0, ymax=nothing,
                              n_profiles=18)

    set_plot_style!()

    # Which branches to plot
    branch_indices = if idxs === :auto
        [i for i in 1:length(branches) if !isempty(branches[i].branch)]
    elseif !isempty(idxs) && idxs[1] isa AbstractVector
        # flatten groups
        reduce(vcat, idxs)
    else
        collect(idxs)
    end

    plots_list = []

    for bi in branch_indices
        br = branches[bi]
        isempty(br.branch) && continue

        npts = length(br.branch)
        Es = [sol.param for sol in br.branch]

        idx_Emin = argmin(Es)
        idx_Emax = argmax(Es)

        # Turning points: where E reverses direction
        turning = Int[]
        if npts >= 3
            dE = diff(Es)
            for t in 2:length(dE)
                if dE[t-1] * dE[t] < 0
                    push!(turning, t)
                end
            end
        end

        # Mandatory: first, last, E-min, E-max, turning points
        mandatory = sort(unique(vcat(1, npts, idx_Emin, idx_Emax, turning)))

        if length(mandatory) >= n_profiles
            indices = mandatory
        else
            remaining = setdiff(1:npts, mandatory)
            n_extra = n_profiles - length(mandatory)
            if !isempty(remaining)
                extra_idx = round.(Int,
                    range(1, length(remaining);
                          length=min(n_extra, length(remaining))))
                extra = remaining[unique(extra_idx)]
                indices = sort(unique(vcat(mandatory, extra)))
            else
                indices = mandatory
            end
        end

        # Deduplicate by rounded E
        seenE = Set{Float64}()
        uniq = Int[]
        for idx in indices
            Er = round(Es[idx]; digits=4)
            if !(Er in seenE)
                push!(uniq, idx)
                push!(seenE, Er)
            end
        end
        indices = uniq

        # Important = first, last, E extremes
        important = Set([1, npts, idx_Emin, idx_Emax])

        # Viridis gradient across selected profiles
        colors = palette(:viridis, max(length(indices), 1))

        p = plot(;
            xlabel = L"x",
            ylabel = L"\psi(x)",
            title  = "Branch $bi  profiles",
            legend = :outerright,
            legendfontsize = 7,
            size   = (650, 300),
        )

        max_psi = 0.0

        # Collect curve data, then draw non-important first, important on top
        curve_data = []

        for (j, idx) in enumerate(indices)
            sol = br.branch[idx]
            E, c, slope_sign = sol.param, sol.c, sol.slope_sign
            xs, us, vs = integrate_support(a, b, E, Vfun;
                                           N=Ngrid, c=c, slope_sign=slope_sign)
            isempty(xs) && continue
            xfull, psifull, _ = glue_full_solution(a, b, E, xs, us, vs; Xmax=Xmax)
            isempty(xfull) && continue

            xfull = real.(xfull)
            psifull = real.(psifull)
            # Sign convention: psi(a) >= 0
            ia = argmin(abs.(xfull .- a))
            if psifull[ia] < 0
                psifull .*= -1
            end

            max_psi = max(max_psi, maximum(abs, psifull))
            is_imp = idx in important

            push!(curve_data, (; xfull, psifull, E, is_imp, color=colors[j]))
        end

        # Non-important curves first (behind)
        for cv in curve_data
            cv.is_imp && continue
            plot!(p, cv.xfull, cv.psifull;
                  color=cv.color, lw=1.8,
                  label=L"E = %$(round(cv.E, digits=4))")
        end

        # Important curves on top (bold)
        for cv in curve_data
            cv.is_imp || continue
            plot!(p, cv.xfull, cv.psifull;
                  color=cv.color, lw=3,
                  label=L"\mathbf{E = %$(round(cv.E, digits=4))}")
        end

        # Y-axis limits
        if ymax !== nothing
            ylims!(p, (-ymax, ymax))
        elseif max_psi > 0
            ylims!(p, (-1.15 * max_psi, 1.15 * max_psi))
        end

        # Support boundary markers
        vline!(p, [a, b]; color=:gray40, ls=:dash, lw=1.5, alpha=0.5,
               label="")

        push!(plots_list, p)
    end

    isempty(plots_list) && return nothing

    fig = plot(plots_list...;
               layout=(length(plots_list), 1),
               size=(650, 300 * length(plots_list)))
    display(fig)
    return fig
end


# =============================================================================
# POTENTIAL PLOT
# =============================================================================

"""
    plot_potential(Vfun; xmin=-2, xmax=2, npts=600)

Simple potential plot for context.
"""
function plot_potential(Vfun; xmin=-2, xmax=2, npts=600)
    set_plot_style!()

    xs = range(xmin, xmax; length=npts)
    Vvals = [Vfun(x) for x in xs]

    plt = plot(xs, Vvals;
               color=:red, lw=2.5,
               label=L"V(x)",
               xlabel=L"x",
               legend=:right,
               margin=0Plots.mm,
               left_margin=-2Plots.mm,
               framestyle=:box,
               size=(350, 200),
               title="")
    display(plt)
    return plt
end


# =============================================================================
# SPECTRAL PLOTS
# =============================================================================

"""
    plot_spectrum_evolution(spec_data; title="")

Plot eigenvalues of L+ and L- as functions of E along a branch.

Single panel showing the first `nev` eigenvalues of both operators,
plus the essential spectrum edge lambda = -E.
"""
function plot_spectrum_evolution(spec_data; title="")
    set_plot_style!()

    Es = spec_data.Es
    evals_plus = spec_data.evals_plus
    evals_minus = spec_data.evals_minus

    if isempty(Es)
        println("No spectral data to plot")
        return nothing
    end

    n_eigs = size(evals_plus, 1)

    plt = plot(;
        xlabel = L"E",
        ylabel = L"\lambda",
        legend = :outerright,
        title  = isempty(title) ? "Stability Spectrum" : title,
        size   = (700, 450),
    )

    # L+ eigenvalues (solid, blue shades)
    colors_plus = [:navy, :blue, :dodgerblue, :deepskyblue, :lightskyblue]
    for j in 1:n_eigs
        lam_j = evals_plus[j, 1:length(Es)]
        col = j <= length(colors_plus) ? colors_plus[j] : :blue
        plot!(plt, Es, lam_j;
              lw=2, color=col,
              label=L"\lambda_{%$j}(L_+)")
    end

    # L- eigenvalues (dashed, red shades)
    n_eigs_m = size(evals_minus, 1)
    colors_minus = [:darkred, :red, :orangered, :orange, :gold]
    for j in 1:n_eigs_m
        lam_j = evals_minus[j, 1:length(Es)]
        col = j <= length(colors_minus) ? colors_minus[j] : :red
        plot!(plt, Es, lam_j;
              lw=2, ls=:dash, color=col,
              label=L"\lambda_{%$j}(L_-)")
    end

    # Zero line
    hline!(plt, [0.0]; color=:black, ls=:dot, lw=1.5, alpha=0.7, label="")

    # Essential spectrum edge: lambda = -E
    Es_line = range(minimum(Es), maximum(Es); length=100)
    plot!(plt, Es_line, -Es_line;
          color=:green, lw=2, ls=:dashdot,
          label=L"\lambda = -E\ \textrm{(ess. spec.)}")

    display(plt)
    return plt
end


# =============================================================================
# STABILITY DIAGRAM
# =============================================================================

"""
    plot_stability_diagram(branches, a, b, Vfun;
                           nev=5, skip=3, Ngrid=7000, Xmax=50.0, Ngrid_FD=6000)

Two-panel stability diagram:
- Top: L² norm vs E colored green (stable) / red (unstable).
  Stable = n(L+) = 1, n(L-) = 0 (Vakhitov-Kolokolov criterion).
- Bottom: counts of negative eigenvalues n(L+) and n(L-) vs E.
"""
function plot_stability_diagram(branches, a, b, Vfun;
                                nev=5, skip=3, Ngrid=7000,
                                Xmax=50.0, Ngrid_FD=6000)
    set_plot_style!()

    plt_mass = plot(;
        xlabel = L"E",
        ylabel = L"\mathcal{N}[\psi]",
        legend = :best,
        title  = "Stability Diagram",
        size   = (600, 320),
    )

    plt_count = plot(;
        xlabel = L"E",
        ylabel = "# negative eigenvalues",
        legend = :best,
        size   = (600, 280),
        yticks = 0:1:10,
    )

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue

        Es = Float64[]
        L2s = Float64[]
        n_neg_plus = Int[]
        n_neg_minus = Int[]

        indices = 1:skip:length(br.branch)
        println("  Analyzing stability for Branch $i ($(length(indices)) points)...")

        for idx in indices
            sol = br.branch[idx]
            E, c, ss = sol.param, sol.c, sol.slope_sign

            x, u, v = integrate_support(a, b, E, Vfun;
                                        N=Ngrid, c=c, slope_sign=ss)
            isempty(x) && continue
            L2 = compute_norm(a, b, E, x, u, v)
            isfinite(L2) || continue

            try
                lp, lm = compute_Lpm_eigenvalues(a, b, E, Vfun, c, ss;
                                                  nev=nev, Ngrid=Ngrid_FD, Xmax=Xmax)
                push!(Es, E)
                push!(L2s, L2)
                push!(n_neg_plus,  count(l -> l < -1e-8, lp))
                push!(n_neg_minus, count(l -> l < -1e-8, lm))
            catch e
                @warn "Eigenvalue failed at E=$E: $e"
            end
        end

        isempty(Es) && continue

        # Sort by E for clean line plots
        perm = sortperm(Es)
        Es = Es[perm]
        L2s = L2s[perm]
        n_neg_plus = n_neg_plus[perm]
        n_neg_minus = n_neg_minus[perm]

        # Mass curve colored by stability
        for j in 1:(length(Es) - 1)
            is_stable = (n_neg_plus[j] == 1 && n_neg_minus[j] == 0)
            color = is_stable ? :green : :red
            style = is_stable ? :solid : :dash

            plot!(plt_mass, [Es[j], Es[j+1]], [L2s[j], L2s[j+1]];
                  lw=3, color=color, ls=style, label="")
        end

        # Eigenvalue count curves
        plot!(plt_count, Es, n_neg_plus;
              lw=2, marker=:circle, ms=3, color=:blue,
              label=(i == 1 ? L"n(L_+)" : ""))
        plot!(plt_count, Es, n_neg_minus;
              lw=2, marker=:square, ms=3, color=:red, ls=:dash,
              label=(i == 1 ? L"n(L_-)" : ""))
    end

    vline!(plt_mass,  [0.0]; color=:black, ls=:dot, lw=1, alpha=0.4, label="")
    vline!(plt_count, [0.0]; color=:black, ls=:dot, lw=1, alpha=0.4, label="")

    fig = plot(plt_mass, plt_count; layout=(2, 1), size=(650, 580))
    display(fig)
    return fig
end
