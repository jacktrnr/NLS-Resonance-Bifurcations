###############################################
# potentials.jl - Potential constructors
###############################################
#
# Each potential V(x) has compact support on [0, b] (V(x) = 0 for x > b).
# The NLS equation is:  -ψ'' + V(x)ψ - ψ³ = Eψ  on x > 0.
#
# Available potentials:
#   square_well           — constant V0 on (0, b)
#   square_well_with_bump — V0 + Gaussian bump near boundary
#   step_potential        — piecewise constant (V1 left, V2 right)
#   gaussian_well         — Gaussian centered at b/2
#   three_step_potential  — edge shelves + deep center
#   smooth_well           — C^∞ bump function, vanishes exactly at endpoints
#
# Use make_potential(:type; b, V0, ...) to build any of these from a symbol.
#
###############################################

# =============================================================================
# POTENTIAL CONSTRUCTORS
# =============================================================================

"""
    square_well(b, V0) -> V(x)

Constant potential:
    V(x) = V0   for 0 < x < b
    V(x) = 0    otherwise
"""
square_well(b, V0) = x -> (0 < x < b ? V0 : 0.0)

"""
    square_well_with_bump(b, V0; bump_amp, bump_width, bump_center) -> V(x)

Square well with a Gaussian bump superimposed:
    V(x) = V0 + A * exp(-(x - c)² / 2w²)   for 0 < x < b
    V(x) = 0                                  otherwise

Default bump is centered at 0.9b with amplitude 0.5|V0| and width b/20.
Useful for breaking symmetry or creating resonance-like features.
"""
square_well_with_bump(b, V0;
                      bump_amp = 0.5*abs(V0),
                      bump_width = b/20,
                      bump_center = 0.9b) = x -> begin
    if 0 < x < b
        V0 + bump_amp * exp(-((x - bump_center)^2) / (2*bump_width^2))
    else
        0.0
    end
end

"""
    step_potential(b, V1, V2) -> V(x)

Two-step piecewise constant:
    V(x) = V1   for 0 < x < b/2
    V(x) = V2   for b/2 <= x < b
    V(x) = 0    otherwise

Models an asymmetric well (e.g. V1 = -6, V2 = -1).
"""
function step_potential(b, V1, V2)
    return x -> begin
        if 0 < x < b/2
            V1
        elseif b/2 <= x < b
            V2
        else
            0.0
        end
    end
end

"""
    gaussian_well(b, V0; σ=0.3) -> V(x)

Gaussian well centered at b/2:
    V(x) = V0 * exp(-(x - b/2)² / σ²)   for 0 < x < b
    V(x) = 0                               otherwise

The width parameter σ controls how localized the well is.
Note: this is NOT truly compactly supported — it's truncated at x = 0 and x = b.
"""
function gaussian_well(b, V0; σ=0.3)
    return x -> begin
        if 0 < x < b
            V0 * exp(-((x - b/2)/σ)^2)
        else
            0.0
        end
    end
end

"""
    three_step_potential(b, V0; edge_height=0.1) -> V(x)

Three-step piecewise constant:
    V(x) = edge_height   for 0 < x <= b/4       (left shelf)
    V(x) = V0            for b/4 < x < 3b/4     (deep center)
    V(x) = edge_height   for 3b/4 <= x < b      (right shelf)
    V(x) = 0             otherwise

Models a well with raised edges — can trap multiple modes.
"""
function three_step_potential(b, V0; edge_height=0.1)
    return x -> begin
        if 0 < x <= b/4
            edge_height
        elseif b/4 < x < 3*b/4
            V0
        elseif 3*b/4 <= x < b
            edge_height
        else
            0.0
        end
    end
end

"""
    smooth_well(b, V0) -> V(x)

C^∞ compactly supported bump function:
    V(x) = V0 * exp(-1 / (1 - (x/b)²))   for 0 < x < b
    V(x) = 0                                otherwise

Vanishes exactly (with all derivatives) at x = 0 and x = b.
Useful when you need a smooth potential for theoretical reasons.
"""
function smooth_well(b, V0)
    return x -> begin
        if 0 < x < b
            t = (x / b)^2
            t >= 1.0 ? 0.0 : V0 * exp(-1.0 / (1.0 - t))
        else
            0.0
        end
    end
end

# =============================================================================
# DISPATCHER — build any potential from a symbol
# =============================================================================

"""
    make_potential(type::Symbol; b, V0, kwargs...) -> V(x)

Build a potential function from a type symbol and keyword arguments.

# Supported types
| Symbol         | Constructor              | Extra kwargs used                               |
|:---------------|:-------------------------|:------------------------------------------------|
| `:square`      | `square_well`            | —                                               |
| `:square_bump` | `square_well_with_bump`  | `bump_amp_factor`, `bump_width_frac`, `bump_center_frac` |
| `:step`        | `step_potential`         | `V1`                                            |
| `:gaussian`    | `gaussian_well`          | `σ_frac`                                        |
| `:threestep`   | `three_step_potential`   | `edge_height`                                   |
| `:smooth`      | `smooth_well`            | —                                               |

# Example
```julia
Vfun = make_potential(:square_bump; b=1.0, V0=-3.4,
                      bump_amp_factor=2.3, bump_width_frac=0.1, bump_center_frac=1.0)
```
"""
function make_potential(type::Symbol; b, V0,
                        bump_amp_factor=2.3,
                        bump_width_frac=0.1,
                        bump_center_frac=1.0,
                        V1=-6.0,
                        σ_frac=0.25,
                        edge_height=0.1)
    if type == :square
        return square_well(b, V0)
    elseif type == :square_bump
        return square_well_with_bump(b, V0;
            bump_amp    = bump_amp_factor * abs(V0),
            bump_width  = bump_width_frac * b,
            bump_center = bump_center_frac * b)
    elseif type == :step
        return step_potential(b, V1, V0)
    elseif type == :gaussian
        return gaussian_well(b, V0; σ = σ_frac * b)
    elseif type == :threestep
        return three_step_potential(b, V0; edge_height=edge_height)
    elseif type == :smooth
        return smooth_well(b, V0)
    else
        error("Unknown potential type: $type. " *
              "Choose from: :square, :square_bump, :step, :gaussian, :threestep, :smooth")
    end
end

"""
    potential_label(type::Symbol; b, V0, kwargs...) -> String

Generate a clean filename-safe label, e.g. `"square_bump_b=1.0_V0=-3.4"`.
"""
function potential_label(type::Symbol; b, V0, kwargs...)
    return "$(type)_b=$(b)_V0=$(V0)"
end
