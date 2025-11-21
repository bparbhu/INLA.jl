# likelihood.jl

"""
    calc_ljoint(y, x, α)

Joint log-density log p(y, x, α) up to an additive constant, for the
Bernoulli-logit likelihood and AR(1) latent x with precision Q(α),
plus the R-INLA-style prior on α.
"""
function calc_ljoint(y::AbstractVector{<:Real},
                     x::AbstractVector{<:Real},
                     α::Float64)

    # For n=100 it's fine to work dense
    Qα = Matrix(calc_Q(α))

    # standard dense Cholesky: Q = L * L'
    F = cholesky(Symmetric(Qα); check = false)
    L = F.L

    # log |Q|^{1/2} = ∑ log diag(L)
    logdet_Q_half = sum(log.(diag(L)))

    # quadratic form: x' Q x = ||L * x||²
    Lx        = L * x
    quad_form = sum(abs2, Lx)

    # Bernoulli log-likelihood under logit link: η = β x
    xβ = beta_true .* x
    ll = sum(xβ .* y .- log1p.(exp.(xβ)))

    return ll + logdet_Q_half - 0.5 * quad_form + calc_lprior_rinla(α)
end

"""
    calc_ff(x, α)

f(x; α) = log p(y, x | α) up to constant, used for Laplace approximation.
"""
function calc_ff(x::AbstractVector{<:Real}, α::Float64)
    Qα = calc_Q(α)
    xβ = beta_true .* x
    ll = sum(xβ .* y .- log1p.(exp.(xβ)))
    quad = 0.5 * (x' * Qα * x)
    return ll - quad
end

"""
    calc_grad_ff(x, α)

Gradient ∂f/∂x.
"""
function calc_grad_ff(x::AbstractVector{<:Real}, α::Float64)
    Qα = calc_Q(α)
    xβ = beta_true .* x
    g_like = beta_true .* y .- beta_true .* exp.(xβ) ./ (1 .+ exp.(xβ))
    return g_like .- Qα * x
end

"""
    calc_neg_hess_ff(x, α)

Negative Hessian of f(x; α):

Q(α) + diag(β² * exp(βx)/(1+exp(βx))²).
"""
function calc_neg_hess_ff(x::AbstractVector{<:Real}, α::Float64)
    Qα = calc_Q(α)
    xβ = beta_true .* x
    w  = beta_true^2 .* exp.(xβ) ./ (1 .+ exp.(xβ)).^2
    return Qα + spdiagm(0 => w)
end

# g(x) = log p(y | x, α) and its derivatives for Newton-like updates

calc_g0(x) = sum(beta_true .* x .* y .- log1p.(exp.(beta_true .* x)))

function calc_g1(x)
    xβ = beta_true .* x
    return beta_true .* y .- beta_true .* exp.(xβ) ./ (1 .+ exp.(xβ))
end

function calc_g2(x)
    xβ = beta_true .* x
    return -beta_true^2 .* exp.(xβ) ./ (1 .+ exp.(xβ)).^2
end
