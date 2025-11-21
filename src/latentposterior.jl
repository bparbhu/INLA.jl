# latentposterior.jl

"""
    PostXRecord

Stores, for each α on the grid:

- α
- x0          : mode of x | y, α
- diag_sigma2 : diagonal of covariance of x | y, α (Laplace)
- unn_log_post: unnormalized log p(α | y)
"""
struct PostXRecord
    alpha        :: Float64
    x0           :: Vector{Float64}
    diag_sigma2  :: Vector{Float64}
    unn_log_post :: Float64
end

"""
    compute_post_x(alpha_vec)

Approximate p(x | y, α) via Laplace for each α in `alpha_vec`.
Returns a vector of `PostXRecord`.
"""
function compute_post_x(alpha_vec::Vector{Float64})
    posts = Vector{PostXRecord}(undef, length(alpha_vec))
    for (k, α) in enumerate(alpha_vec)
        x0 = calc_x0(α)
        H  = calc_neg_hess_ff(x0, α)
        Σ  = inv(Matrix(H))
        diag_Σ = diag(Σ)
        F  = cholesky(Symmetric(Matrix(H)))
        logdet_H_half = sum(log.(diag(F.U)))
        unn_log_post = calc_ljoint(y, x0, α) - logdet_H_half
        posts[k] = PostXRecord(α, x0, diag_Σ, unn_log_post)
    end
    return posts
end

"""
    latent_posterior_at_mode(α_hat)

"R-INLA-like" latent posterior at the hyperparameter mode α̂:
returns (μ_tight, σ_tight) where μ_tight is the mode x₀(α̂) and
σ_tight is the marginal sd from the Laplace covariance at α̂.
"""
function latent_posterior_at_mode(α_hat::Float64)
    x_hat = calc_x0(α_hat)
    H_hat = calc_neg_hess_ff(x_hat, α_hat)
    Σ_hat = inv(Matrix(H_hat))
    mu_tight    = x_hat
    sigma_tight = sqrt.(diag(Σ_hat))
    return mu_tight, sigma_tight
end
