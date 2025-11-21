# api.jl

using Statistics

"""
    INLAResult

Holds the main outputs of an INLA run for 1D hyperparameter α
and latent Gaussian field x.
"""
struct INLAResult
    alpha_grid :: Vector{Float64}  # grid of α values
    post_alpha :: Vector{Float64}  # p(α | y) on that grid (normalized)
    alpha_mode :: Float64          # argmax of log p(α | y)
    mu_latent  :: Vector{Float64}  # posterior mean of x_t (mixture)
    sd_latent  :: Vector{Float64}  # posterior sd of x_t (mixture)
    mu_tight   :: Vector{Float64}  # mode-based "R-INLA-like" mean
    sd_tight   :: Vector{Float64}  # mode-based sd
end

"""
    inla(y; model=:ar1, likelihood=:bernoulli_logit,
             alpha_min=-0.95, alpha_max=0.95, n_alpha=31)

High-level INLA wrapper for the AR(1)-Bernoulli toy model.

Returns an `INLAResult` with:
- posterior over α on a grid
- mixture posterior for x_t (integrated over α)
- mode-based ("R-INLA-like") posterior for x_t at α̂

Currently supports only `model = :ar1` and `likelihood = :bernoulli_logit`,
and uses your existing calc_* stack under the hood.
"""
function inla(y_in::AbstractVector{<:Real};
              model::Symbol      = :ar1,
              likelihood::Symbol = :bernoulli_logit,
              alpha_min::Float64 = -0.95,
              alpha_max::Float64 =  0.95,
              n_alpha::Int       = 31)

    # 1. Latent model
    if model != :ar1
        error("Only model=:ar1 is currently implemented.")
    end

    m = AR1Model(length(y_in))

    # 2. Likelihood
    lik = make_likelihood(likelihood)
    check_supported(lik)  # error if not BernoulliLogit (for now)

    # 3. Put y into the global that calc_* expects
    global y = Float64.(y_in)

    # 4. α grid and log posterior on grid
    alpha_vec = collect(range(alpha_min, alpha_max, length=n_alpha))
    lpost     = [calc_lpost(α) for α in alpha_vec]
    lpost    .-= mean(lpost)
    Z_alpha   = calc_Z(alpha_vec, lpost)
    post_alpha = exp.(lpost) ./ Z_alpha

    # Posterior mode α̂
    idx_max   = argmax(lpost)
    alpha_hat = alpha_vec[idx_max]

    # 5. Latent mixture via compute_post_x
    post_x = compute_post_x(alpha_vec)

    mu_mat       = hcat([p.x0          for p in post_x]...)  # n × K
    sigma2_mat   = hcat([p.diag_sigma2 for p in post_x]...)
    lpost_latent = [p.unn_log_post     for p in post_x]

    lpost_latent .-= mean(lpost_latent)
    Z_latent = calc_Z(alpha_vec, lpost_latent)
    w_alpha  = exp.(lpost_latent) ./ Z_latent

    n_local = size(mu_mat, 1)
    mu_post    = [sum(mu_mat[i, :] .* w_alpha) for i in 1:n_local]
    sigma_post = [sqrt(sum(((mu_mat[i, :] .- mu_post[i]).^2 .+ sigma2_mat[i, :]) .* w_alpha))
                  for i in 1:n_local]

    # 6. Mode-based ("R-INLA-like") posterior at α̂
    mu_tight, sd_tight = latent_posterior_at_mode(alpha_hat)

    return INLAResult(
        alpha_vec,
        post_alpha,
        alpha_hat,
        mu_post,
        sigma_post,
        mu_tight,
        sd_tight
    )
end
