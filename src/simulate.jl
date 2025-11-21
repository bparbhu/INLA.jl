# simulate.jl

# Toy example constants (as in Siegert’s blog)
const n          = 100
const alpha_true = 0.6
const sigma_true = 0.1
const beta_true  = 10.0

"""
    simulate_ar1(n, α, σ)

Simulate an AR(1) latent state sequence x₁,…,xₙ with
stationary initial distribution: x₁ ~ N(0, σ² / (1 - α²)).
"""
function simulate_ar1(n::Int, α::Float64, σ::Float64)
    x = zeros(Float64, n)
    x[1] = rand(Normal(0, σ / sqrt(1 - α^2)))
    for t in 2:n
        x[t] = α * x[t-1] + σ * randn()
    end
    return x
end

# Default toy dataset used if you just `using INLA` and start playing
x_true = simulate_ar1(n, alpha_true, sigma_true)
p_true = 1.0 ./ (1.0 .+ exp.(-beta_true .* x_true))

# y is 0/1 (Float64) so it fits your existing calc_* functions
y = Float64.(rand.(Bernoulli.(p_true)))
