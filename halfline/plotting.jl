###############################################
# plotting.jl - Visualization for half-line NLS
###############################################
#
# All plotting functions for the bifurcation analysis:
#   plot_potential          — V(x) with support boundary marked
#   plot_mass_energy        — L² and H¹ norms vs E (full + zoomed views)
#   plot_profiles           — Solution profiles ψ(x) along branches
#   plot_spectrum_evolution — Eigenvalues of L₊/L₋ vs E
#   plot_stability_diagram  — Mass vs E colored by stability + n(L±) counts
#
###############################################

using Plots
using LaTeXStrings
using Colors

# =============================================================================
# STYLE DEFAULTS
# =============================================================================

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
# POTENTIAL PLOT
# =============================================================================

"""
    plot_potential(Vfun, b; xmax=5.0, npts=500)

Plot the potential V(x) on [0, xmax] with the support boundary x = b marked.
"""
function plot_potential(Vfun, b; xmax=5.0, npts=500)
    set_plot_style!()

    xs = range(0, xmax; length=npts)
    Vs = [Vfun(x) for x in xs]

    plt = plot(xs, Vs;
               color=:red, lw=2.5,
               xlabel=L"x",
               ylabel=L"V(x)",
               label="",
               title="Potential",
               size=(500, 300))

    # Mark support boundary
    vline!(plt, [b]; color=:gray40, ls=:dash, lw=1.5, alpha=0.6,
           label=L"x = b")

    # Zero line for reference
    hline!(plt, [0.0]; color=:black, ls=:dot, lw=1, alpha=0.3, label="")

    display(plt)
    return plt
end

# Backwards-compatible: allow calling without b
plot_potential(Vfun; xmax=5.0, npts=500) = plot_potential(Vfun, 0.0; xmax=xmax, npts=npts)

# =============================================================================
# MASS-ENERGY PLOTS
# =============================================================================

"""
    plot_mass_energy(branches, b, Vfun; N=2000, Emin=-3.0, ...)

Plot L² and H¹ norms vs E for all branches.

Returns (plt_L2, plt_H1, plt_L2_zoom, plt_H1_zoom) — full-range and
auto-zoomed views that focus on the region with the most data density.
"""
function plot_mass_energy(branches, b, Vfun;
                         N=2000,
                         Emin=-3.0,
                         l2max=nothing,
                         h1max=nothing)
    set_plot_style!()

    # --- Collect norm data for all branches ---
    all_Es = Float64[]
    all_L2 = Float64[]
    all_H1 = Float64[]
    branch_data = []

    colors = distinguishable_colors(max(length(branches), 1),
                                   [RGB(1,1,1), RGB(0,0,0)]; dropseed=true)

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue

        Es = Float64[]
        L2s = Float64[]
        H1s = Float64[]

        for sol in br.branch
            E = sol.param
            β = sol.β

            x, u, v = shoot_from_origin(b, E, Vfun, abs(β); N=N)
            if isempty(x)
                continue
            end

            L2 = compute_L2_norm(b, E, x, u, v)
            H1 = compute_H1_norm(b, E, x, u, v)

            if isfinite(L2) && isfinite(H1)
                push!(Es, E)
                push!(L2s, L2)
                push!(H1s, H1)
                push!(all_Es, E)
                push!(all_L2, L2)
                push!(all_H1, H1)
            end
        end

        push!(branch_data, (; Es=Es, L2s=L2s, H1s=H1s, color=colors[i], idx=i))
    end

    # --- Axis limits ---
    if l2max === nothing && !isempty(all_L2)
        l2max = 1.15 * maximum(all_L2)
    end
    if h1max === nothing && !isempty(all_H1)
        h1max = 1.15 * maximum(all_H1)
    end
    l2max = something(l2max, 10.0)
    h1max = something(h1max, 30.0)

    # --- Helper: build one mass-energy plot ---
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

        # Reference lines
        vline!(plt, [0.0]; color=:black, ls=:dash, lw=1, alpha=0.4, label="")

        return plt
    end

    # --- Full-range plots ---
    plt_L2 = _build_plot(
        L"\mathcal{N}[\psi] = \int |\psi|^2\,dx",
        "Mass vs Energy",
        :L2s, l2max, (Emin, 0.05))

    plt_H1 = _build_plot(
        L"\|\psi\|_{H^1}^2",
        L"H^1\textrm{-norm vs Energy}",
        :H1s, h1max, (Emin, 0.05))

    # --- Zoomed plots (focus on data extent with padding) ---
    if !isempty(all_Es)
        E_lo = minimum(all_Es)
        E_hi = maximum(all_Es)
        E_pad = max(0.05 * (E_hi - E_lo), 0.02)
        zoom_xlims = (E_lo - E_pad, E_hi + E_pad)

        # Tighter y-limits: based on actual data in the zoomed E range
        l2_zoom_max = 1.1 * l2max
        h1_zoom_max = 1.1 * h1max
    else
        zoom_xlims = (Emin, 0.05)
        l2_zoom_max = l2max
        h1_zoom_max = h1max
    end

    plt_L2_zoom = _build_plot(
        L"\mathcal{N}[\psi]",
        "Mass vs Energy (zoomed)",
        :L2s, l2_zoom_max, zoom_xlims)

    plt_H1_zoom = _build_plot(
        L"\|\psi\|_{H^1}^2",
        L"H^1\textrm{-norm (zoomed)}",
        :H1s, h1_zoom_max, zoom_xlims)

    display(plt_L2)
    display(plt_H1)

    return plt_L2, plt_H1, plt_L2_zoom, plt_H1_zoom
end

# =============================================================================
# PROFILE PLOTS
# =============================================================================

"""
    plot_profiles(branches, b, Vfun; N=2000, Xmax=20.0, n_profiles=18, ymax=nothing)

Plot solution profiles ψ(x) for selected energies along each branch.
Profiles are chosen to include endpoints, turning points, and evenly
spaced intermediates. A viridis color gradient encodes progression along
the branch.
"""
function plot_profiles(branches, b, Vfun;
                      N=2000,
                      Xmax=20.0,
                      n_profiles=18,
                      ymax=nothing)

    set_plot_style!()
    plots_list = []

    for (i, br) in enumerate(branches)
        isempty(br.branch) && continue

        p = plot(;
            xlabel = L"x",
            ylabel = L"\psi(x)",
            title = "Branch $i  profiles",
            legend = :outerright,
            legendfontsize = 7,
            size = (650, 300),
        )

        Es = [sol.param for sol in br.branch]
        n_total = length(Es)

        # Identify turning points (where E reverses direction)
        turning = Int[]
        if n_total >= 3
            dE = diff(Es)
            for k in 2:length(dE)
                if dE[k-1] * dE[k] < 0
                    push!(turning, k)
                end
            end
        end

        # Mandatory: endpoints + turning points
        mandatory = sort(unique(vcat(1, turning, n_total)))

        if length(mandatory) >= n_profiles
            indices = mandatory
        else
            remaining = setdiff(1:n_total, mandatory)
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
        uniq_indices = Int[]
        for idx in indices
            Er = round(Es[idx]; digits=4)
            if !(Er in seenE)
                push!(uniq_indices, idx)
                push!(seenE, Er)
            end
        end
        indices = uniq_indices

        colors = palette(:viridis, max(length(indices), 1))
        max_psi = 0.0

        for (j, idx) in enumerate(indices)
            sol = br.branch[idx]
            E = sol.param
            β = sol.β

            x, u, v = shoot_from_origin(b, E, Vfun, β; N=N)
            isempty(x) && continue

            xfull, ψfull = glue_solution(b, E, x, u, v; Xmax=Xmax)
            isempty(xfull) && continue

            max_psi = max(max_psi, maximum(abs, ψfull))

            plot!(p, xfull, real.(ψfull);
                  color=colors[j], lw=1.8,
                  label=L"E = %$(round(E, digits=4))")
        end

        # Y-axis limits
        if ymax !== nothing
            ylims!(p, (-ymax, ymax))
        elseif max_psi > 0
            ylims!(p, (-1.15*max_psi, 1.15*max_psi))
        end

        # Mark support boundary
        vline!(p, [b]; color=:gray40, ls=:dash, lw=1.5, alpha=0.5,
               label=L"x = b")

        push!(plots_list, p)
    end

    isempty(plots_list) && return nothing

    fig = plot(plots_list...;
               layout=(length(plots_list), 1),
               size=(650, 300*length(plots_list)))

    display(fig)
    return fig
end

# =============================================================================
# SPECTRAL PLOTS
# =============================================================================

"""
    plot_spectrum_evolution(spec_data; n_show=2, show_minus=false, title="")

Plot eigenvalues of L₊ (and optionally L₋) as functions of E along a branch.
Top panel: all tracked eigenvalues + the λ = -E reference line.
Bottom panel: second eigenvalue λ₂(L₊) alone (key for stability transitions).
"""
function plot_spectrum_evolution(spec_data; n_show=2, show_minus=false, title="")
    set_plot_style!()

    Es = spec_data.Es
    evals_plus = spec_data.evals_plus
    evals_minus = spec_data.evals_minus

    if isempty(Es)
        println("No spectral data to plot")
        return nothing
    end

    n_eigs = min(n_show, size(evals_plus, 1))

    # --- Top panel: all eigenvalues ---
    plt1 = plot(;
        xlabel = L"E",
        ylabel = L"\lambda",
        legend = :right,
        title = isempty(title) ? "Stability Spectrum" : title,
        size = (600, 400),
    )

    colors_plus = [:blue, :dodgerblue, :deepskyblue, :lightskyblue, :lightblue]

    for j in 1:n_eigs
        λ_j = evals_plus[j, 1:length(Es)]
        color = j <= length(colors_plus) ? colors_plus[j] : :blue
        plot!(plt1, Es, λ_j;
              lw=2, color=color,
              label=j == 1 ? L"\lambda_j(L_+)" : "",
              alpha=1.0 - 0.15*(j-1))
    end

    if show_minus
        colors_minus = [:red, :orangered, :orange, :gold, :yellow]
        for j in 1:n_eigs
            λ_j = evals_minus[j, 1:length(Es)]
            color = j <= length(colors_minus) ? colors_minus[j] : :red
            plot!(plt1, Es, λ_j;
                  lw=2, ls=:dash, color=color,
                  label=j == 1 ? L"\lambda_j(L_-)" : "",
                  alpha=1.0 - 0.15*(j-1))
        end
    end

    hline!(plt1, [0.0]; color=:black, ls=:dot, lw=1.5, alpha=0.7, label="")

    # Essential spectrum edge: λ = -E
    Es_line = range(minimum(Es), maximum(Es); length=100)
    plot!(plt1, Es_line, -Es_line;
          color=:green, lw=2, ls=:dashdot,
          label=L"\lambda = -E\ \textrm{(ess. spec. edge)}")

    # --- Bottom panel: second eigenvalue only ---
    plt2 = plot(;
        xlabel = L"E",
        ylabel = L"\lambda_2(L_+)",
        legend = :best,
        title = L"\textrm{Second eigenvalue of}\ L_+",
        size = (600, 300),
    )

    if size(evals_plus, 1) >= 2
        λ2 = evals_plus[2, 1:length(Es)]
        plot!(plt2, Es, λ2;
              lw=2.5, color=:blue,
              label=L"\lambda_2(L_+)")
    end

    hline!(plt2, [0.0]; color=:black, ls=:dot, lw=1.5, alpha=0.7, label="")

    fig = plot(plt1, plt2; layout=(2,1), size=(650, 750))
    display(fig)
    return fig
end

# =============================================================================
# STABILITY DIAGRAM
# =============================================================================

"""
    plot_stability_diagram(branches, b, Vfun; nev=5, skip=3, N=2000, Xmax=50.0, Ngrid=800)

Create a two-panel stability diagram:
- Top: L² norm vs E with green (stable) / red (unstable) coloring.
  Stable = n(L₊) = 1, n(L₋) = 0 (Vakhitov-Kolokolov criterion).
- Bottom: counts of negative eigenvalues n(L₊) and n(L₋) vs E.
"""
function plot_stability_diagram(branches, b, Vfun;
                               nev=5, skip=3, N=2000, Xmax=50.0, Ngrid=800)
    set_plot_style!()

    plt_mass = plot(;
        xlabel = L"E",
        ylabel = L"\mathcal{N}[\psi]",
        legend = :best,
        title = "Stability Diagram",
        size = (600, 320),
    )

    plt_count = plot(;
        xlabel = L"E",
        ylabel = "# negative eigenvalues",
        legend = :best,
        size = (600, 280),
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
            E = sol.param
            β = sol.β

            x, u, v = shoot_from_origin(b, E, Vfun, β; N=N)
            if isempty(x)
                continue
            end
            L2 = compute_L2_norm(b, E, x, u, v)
            if !isfinite(L2)
                continue
            end

            try
                λp, λm = compute_Lpm_eigenvalues(b, E, Vfun, β;
                                                nev=nev, Ngrid=Ngrid, Xmax=Xmax)

                push!(Es, E)
                push!(L2s, L2)
                push!(n_neg_plus, count(λ -> λ < -1e-8, λp))
                push!(n_neg_minus, count(λ -> λ < -1e-8, λm))
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

        # Plot mass curve colored by stability
        for j in 1:(length(Es)-1)
            E1, E2 = Es[j], Es[j+1]
            L2_1, L2_2 = L2s[j], L2s[j+1]

            is_stable = (n_neg_plus[j] == 1 && n_neg_minus[j] == 0)
            color = is_stable ? :green : :red
            style = is_stable ? :solid : :dash

            plot!(plt_mass, [E1, E2], [L2_1, L2_2];
                  lw=3, color=color, ls=style, label="")
        end

        # Eigenvalue count curves
        plot!(plt_count, Es, n_neg_plus;
              lw=2, marker=:circle, ms=3, color=:blue,
              label=(i==1 ? L"n(L_+)" : ""))

        plot!(plt_count, Es, n_neg_minus;
              lw=2, marker=:square, ms=3, color=:red, ls=:dash,
              label=(i==1 ? L"n(L_-)" : ""))
    end

    vline!(plt_mass, [0.0]; color=:black, ls=:dot, lw=1, alpha=0.4, label="")
    vline!(plt_count, [0.0]; color=:black, ls=:dot, lw=1, alpha=0.4, label="")

    fig = plot(plt_mass, plt_count; layout=(2,1), size=(650, 580))
    display(fig)

    return fig
end
