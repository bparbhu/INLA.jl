# models.jl

abstract type INLALatentModel end

"""
    AR1Model(n; sigma=sigma_true, beta=beta_true)

Simple AR(1) latent Gaussian model with n points, variance σ², and
regression coefficient β for the Bernoulli-logit likelihood.
"""
struct AR1Model <: INLALatentModel
    n::Int
    sigma::Float64
    beta::Float64
end

AR1Model(n::Int; sigma::Float64 = sigma_true, beta::Float64 = beta_true) =
    AR1Model(n, sigma, beta)

"""
    precision(m::AR1Model, α)

Precision matrix Q(α) for the AR(1) model.
Currently just defers to global `calc_Q(α)` which uses global `n` and `sigma_true`.
"""
function precision(m::AR1Model, α::Float64)
    return calc_Q(α)
end
