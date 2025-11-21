# spde.jl

abstract type SPDEModel <: INLALatentModel end

"""
    SimpleSPDEModel(dim, range, sigma, nu)

Scaffold for a Matérn SPDE model.

- `dim`   : spatial dimension (e.g. 1, 2, or 3)
- `range` : correlation range parameter
- `sigma` : marginal std dev
- `nu`    : smoothness parameter

This does not yet build a precision matrix; it is a placeholder
for future SPDE / GMRF construction.
"""
struct SimpleSPDEModel <: SPDEModel
    dim::Int
    range::Float64
    sigma::Float64
    nu::Float64
end

"""
    precision(m::SimpleSPDEModel, θ)

Placeholder: precision matrix for SPDE model.

Right now this throws an error. Later, you can implement the SPDE
GMRF precision (e.g. Lindgren–Rue–Lindström construction) here.
"""
function precision(m::SimpleSPDEModel, θ)
    error("SPDE precision not implemented yet. This is a scaffold.")
end
