###############################################
# save.jl - Data saving and loading utilities
###############################################
#
# Provides:
#   save_run_data  — serialize branches + seeds + params to JLD2
#   load_run_data  — load a saved run, returns a NamedTuple
#   save_plots     — save plot objects to organized PNG subdirectories
#
# Output directory structure (created automatically):
#   results/<potential_type>/
#     data/            — JLD2 files
#     potential/       — V(x) plots
#     mass_energy/     — L2 and H1 norm vs E plots (full + zoomed)
#     profiles/        — Solution profile plots (wide + zoomed)
#     spectrum/        — Eigenvalue evolution plots
#     stability/       — Stability diagram plots
#
###############################################

using JLD2

# =============================================================================
# SAVE / LOAD RUN DATA
# =============================================================================

"""
    save_run_data(filepath; branches, seeds, b, V0, potential_type, kwargs...)

Save bifurcation run data to a JLD2 file.
Branches are serialized as vectors of named tuples `(β, E, param)`,
which allows reloading without BifurcationKit.
"""
function save_run_data(filepath; branches, seeds, b, V0, potential_type, kwargs...)
    mkpath(dirname(filepath))

    # Serialize branches: extract raw data from BifurcationKit objects
    branch_data = []
    for br in branches
        if isempty(br.branch)
            push!(branch_data, (; points=NamedTuple[]))
        else
            points = [(; β=sol.β, E=sol.param, param=sol.param) for sol in br.branch]
            push!(branch_data, (; points=points))
        end
    end

    # Serialize seeds (strip the `p` field which is redundant)
    seed_data = [(; β=s.β, E=s.E) for s in seeds]

    jldsave(filepath;
        branch_data    = branch_data,
        seed_data      = seed_data,
        b              = b,
        V0             = V0,
        potential_type  = String(potential_type),
        extra          = Dict(String(k) => v for (k,v) in kwargs),
    )

    println("  Data saved to: $filepath")
    return filepath
end

"""
    load_run_data(filepath) -> NamedTuple

Load a previously saved JLD2 run file.

Returns `(; branch_data, seed_data, b, V0, potential_type, extra)`.
Use `branch_data[i].points` to access the i-th branch's solution points,
each of which is a NamedTuple `(β, E, param)`.
"""
function load_run_data(filepath)
    data = load(filepath)

    return (;
        branch_data    = data["branch_data"],
        seed_data      = data["seed_data"],
        b              = data["b"],
        V0             = data["V0"],
        potential_type  = Symbol(data["potential_type"]),
        extra          = get(data, "extra", Dict()),
    )
end

# =============================================================================
# SAVE PLOTS
# =============================================================================

"""
    save_plots(results_dir, label; <keyword plots>...)

Save whichever plot keyword arguments are provided (non-nothing) into
organized subdirectories under `results_dir`.

Supported keyword arguments:
  potential_plt, L2_plt, H1_plt, L2_zoom_plt, H1_zoom_plt,
  profiles_plt, profiles_zoom_plt, spectrum_plts, stability_plt
"""
function save_plots(results_dir, label;
                    potential_plt=nothing,
                    L2_plt=nothing,
                    H1_plt=nothing,
                    L2_zoom_plt=nothing,
                    H1_zoom_plt=nothing,
                    profiles_plt=nothing,
                    profiles_zoom_plt=nothing,
                    spectrum_plts=nothing,
                    stability_plt=nothing)

    saved = String[]

    function _save(subdir, filename, plt)
        plt === nothing && return
        dir = joinpath(results_dir, subdir)
        mkpath(dir)
        path = joinpath(dir, filename)
        savefig(plt, path)
        push!(saved, path)
    end

    _save("potential",    "$(label).png",           potential_plt)
    _save("mass_energy",  "$(label)_L2.png",        L2_plt)
    _save("mass_energy",  "$(label)_H1.png",        H1_plt)
    _save("mass_energy",  "$(label)_L2_zoomed.png", L2_zoom_plt)
    _save("mass_energy",  "$(label)_H1_zoomed.png", H1_zoom_plt)
    _save("profiles",     "$(label).png",           profiles_plt)
    _save("profiles",     "$(label)_zoomed.png",    profiles_zoom_plt)
    _save("stability",    "$(label).png",           stability_plt)

    # Spectrum plots can be a vector (one per branch)
    if spectrum_plts !== nothing
        dir = joinpath(results_dir, "spectrum")
        mkpath(dir)
        if spectrum_plts isa AbstractVector
            for (i, plt) in enumerate(spectrum_plts)
                plt === nothing && continue
                path = joinpath(dir, "$(label)_branch$(i).png")
                savefig(plt, path)
                push!(saved, path)
            end
        else
            path = joinpath(dir, "$(label).png")
            savefig(spectrum_plts, path)
            push!(saved, path)
        end
    end

    println("  Saved $(length(saved)) plot(s):")
    for p in saved
        println("    $p")
    end

    return saved
end
