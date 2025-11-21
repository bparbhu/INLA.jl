# priors.jl

"""
    calc_lprior_rinla(α; μ=0.0, τ=0.5)

R-INLA-style prior for α via Normal prior on θ₂ = log((1+α)/(1-α)).
θ₂ ~ Normal(μ, 1/τ). Default matches theta2_param=c(0,0.5) in Siegert’s blog.

Returns the log prior density for α, including the Jacobian of the
transformation α ↦ θ₂.
"""
function calc_lprior_rinla(α::Float64; μ::Float64=0.0, τ::Float64=0.5)
    if abs(α) >= 1.0
        return -Inf
    end
    σ2 = 1.0 / τ
    θ2 = log((1 + α) / (1 - α))
    # Normal log-density for θ2
    logdens = -0.5 * log(2π * σ2) - 0.5 * (θ2 - μ)^2 / σ2
    # Jacobian dθ2/dα = 2 / (1 - α^2)
    logjac = log(2.0) - log(1.0 - α^2)
    return logdens + logjac
end
