############################################################
# INLA-from-scratch AR(1)-Bernoulli example in pure Julia
# with R-INLA–style prior and "R-INLA-like" latent posterior
############################################################

using LinearAlgebra
using SparseArrays
using Random
using Distributions
using DataFrames
using Gadfly
using Statistics

############################################################
# 1. Simulate data (same model as Siegert's blog)
############################################################

Random.seed!(1234)

const n           = 100
const alpha_true  = 0.6
const sigma_true  = 0.1
const beta_true   = 10.0

# simulate AR(1) latent x_t with stationary initial distribution
function simulate_ar1(n::Int, α::Float64, σ::Float64)
    x = zeros(Float64, n)
    x[1] = rand(Normal(0, σ / sqrt(1 - α^2)))
    for t in 2:n
        x[t] = α * x[t-1] + σ * randn()
    end
    return x
end

x_true = simulate_ar1(n, alpha_true, sigma_true)
p_true = 1.0 ./ (1.0 .+ exp.(-beta_true .* x_true))

y = rand.(Bernoulli.(p_true))
y = Float64.(y)  # 0/1

############################################################
# 2. Plot simulated data (x_t, p_t, y_t)
############################################################

df_sim = DataFrame(
    i = vcat(1:n, 1:n, 1:n),
    variable = vcat(fill("x", n), fill("p", n), fill("y", n)),
    value    = vcat(x_true,       p_true,       y)
)

plot_sim = plot(
    df_sim,
    x = :i,
    y = :value,
    color = :variable,
    Geom.line,
    Scale.color_discrete,
    Guide.xlabel("t"),
    Guide.ylabel(""),
    Guide.colorkey(title=""),
    Coord.Cartesian(ymin = minimum(vcat(x_true, p_true, y)) - 0.1,
                    ymax = maximum(vcat(x_true, p_true, y)) + 0.1),
    Theme(key_position = :right)
)

draw(SVG("simulated_data.svg", 6inch, 4inch), plot_sim)

############################################################
# 3. Core INLA building blocks
############################################################

# Sparse AR(1) precision matrix Q(α)
function calc_Q(α::Float64)
    main = vcat([1.0], fill(1.0 + α^2, n - 2), [1.0])
    off  = fill(-α, n - 1)
    Q = spdiagm(0 => main, 1 => off, -1 => off)
    return Q / sigma_true^2
end

# R-INLA-style prior on α via θ2 = log((1+α)/(1-α)), θ2 ~ N(0, 1/0.5)
"""
    calc_lprior_rinla(α; μ=0.0, τ=0.5)

R-INLA-style prior for α via Normal prior on θ2 = log((1+α)/(1-α)).
θ2 ~ Normal(μ, 1/τ). Default matches theta2_param=c(0,0.5) in Siegert's blog.
Includes Jacobian of transformation.
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

# Joint log-density log p(y, x, α) up to a constant
function calc_ljoint(y::AbstractVector{<:Real},
                     x::AbstractVector{<:Real},
                     α::Float64)

    # For n=100 it's totally fine to work dense
    Qα = Matrix(calc_Q(α))

    # standard dense Cholesky: Q = L * L'
    F = cholesky(Symmetric(Qα); check = false)
    L = F.L

    # log |Q|^{1/2} = sum(log diag(L))
    logdet_Q_half = sum(log.(diag(L)))

    # quadratic form: x' Q x = ||L * x||^2
    Lx        = L * x
    quad_form = sum(abs2, Lx)

    # Bernoulli log-likelihood
    xβ = beta_true .* x
    ll = sum(xβ .* y .- log1p.(exp.(xβ)))

    # add R-INLA-style prior on α
    return ll + logdet_Q_half - 0.5 * quad_form + calc_lprior_rinla(α)
end



# f(x; α) = log p(y, x | α) up to constant
function calc_ff(x::AbstractVector{<:Real}, α::Float64)
    Qα = calc_Q(α)
    xβ = beta_true .* x
    ll = sum(xβ .* y .- log1p.(exp.(xβ)))
    # -0.5 x'Qx
    quad = 0.5 * (x' * Qα * x)
    return ll - quad
end

# gradient ∂f/∂x
function calc_grad_ff(x::AbstractVector{<:Real}, α::Float64)
    Qα = calc_Q(α)
    xβ = beta_true .* x
    # derivative of loglik wrt x
    g_like = beta_true .* y .- beta_true .* exp.(xβ) ./ (1 .+ exp.(xβ))
    # subtract Qx
    return g_like .- Qα * x
end

# Negative Hessian of f(x; α): Q + diag(β^2 * exp(βx)/(1+exp(βx))^2)
function calc_neg_hess_ff(x::AbstractVector{<:Real}, α::Float64)
    Qα = calc_Q(α)
    xβ = beta_true .* x
    w  = beta_true^2 .* exp.(xβ) ./ (1 .+ exp.(xβ)).^2
    return Qα + spdiagm(0 => w)
end

# g(x) = log p(y|x, α) and its derivatives for Newton-like updates
calc_g0(x) = sum(beta_true .* x .* y .- log1p.(exp.(beta_true .* x)))

function calc_g1(x)
    xβ = beta_true .* x
    return beta_true .* y .- beta_true .* exp.(xβ) ./ (1 .+ exp.(xβ))
end

function calc_g2(x)
    xβ = beta_true .* x
    return -beta_true^2 .* exp.(xβ) ./ (1 .+ exp.(xβ)).^2
end

# Iterative mode finder for f(x; α) (same scheme as Siegert)
function calc_x0(α::Float64; tol::Float64=1e-12, maxiter::Int=1000)
    Q = calc_Q(α)
    x  = zeros(Float64, n)
    x0 = copy(x)
    for iter in 1:maxiter
        g1 = calc_g1(x0)
        g2 = calc_g2(x0)
        A  = Q - spdiagm(0 => g2)
        b  = g1 .- x0 .* g2
        x  = A \ b
        if mean((x .- x0).^2) < tol
            return x
        end
        x0 .= x
    end
    @warn "calc_x0 did not converge in $maxiter iterations"
    return x
end

############################################################
# 4. Posterior over α (log p(α|y) up to constant)
############################################################

function calc_lpost(α::Float64)
    x0 = calc_x0(α)
    H  = calc_neg_hess_ff(x0, α)
    F  = cholesky(Symmetric(Matrix(H)))
    # log |H|^{1/2} = sum(log diag(U)) when H = U'U
    logdet_H_half = sum(log.(diag(F.U)))
    return calc_ljoint(y, x0, α) - logdet_H_half
end

# Composite Simpson rule for normalization constant Z
function calc_Z(alpha_vec::AbstractVector{<:Real},
                lpost_vec::AbstractVector{<:Real})
    nn = length(alpha_vec)
    hh = alpha_vec[2] - alpha_vec[1]
    ww = vcat(1.0,
              collect(Iterators.flatten(fill.([4.0, 2.0], (nn - 3) ÷ 2))),
              4.0, 1.0)
    return sum(ww .* exp.(lpost_vec)) * hh / 3.0
end

alpha_vec = collect(range(-0.95, 0.95, length=31))
lpost     = [calc_lpost(α) for α in alpha_vec]




# stabilize, normalize
lpost .-= mean(lpost)
Z_alpha    = calc_Z(alpha_vec, lpost)
post_alpha = exp.(lpost) ./ Z_alpha

# Posterior mode α̂
idx_max = argmax(lpost)
alpha_hat = alpha_vec[idx_max]
println("Posterior mode of α (R-INLA-like) = ", alpha_hat)

############################################################
# 5. Alpha posterior plot
############################################################

df_alpha = DataFrame(alpha = alpha_vec, posterior = post_alpha)

max_post = maximum(df_alpha.posterior)

plot_alpha = plot(
    layer(df_alpha, x = :alpha, y = :posterior, Geom.line, Geom.point),
    # vertical line at true alpha
    layer(x = [alpha_true, alpha_true],
          y = [0.0, max_post],
          Geom.line,
          Theme(default_color = "black")),
    Guide.xlabel("α"),
    Guide.ylabel("posterior density"),
    Guide.title("Posterior of α"),
    Theme(key_position = :none)
)

draw(SVG("alpha_posterior.svg", 6inch, 4inch), plot_alpha)

############################################################
# 6. R-INLA prior visualisation (logit α and α)
############################################################

# Default and "new" priors as in Siegert's blog
n_prior = 10_000
logit_default = rand(Normal(0, sqrt(1/0.15)), n_prior)
logit_new     = rand(Normal(0, sqrt(1/0.5)),  n_prior)

α_default = (exp.(logit_default) .- 1.0) ./ (exp.(logit_default) .+ 1.0)
α_new     = (exp.(logit_new)     .- 1.0) ./ (exp.(logit_new)     .+ 1.0)

df_prior = vcat(
    DataFrame(
        type           = fill("default prior N(0, 1/0.15)", n_prior),
        transformation = fill("logit_alpha", n_prior),
        value          = logit_default
    ),
    DataFrame(
        type           = fill("default prior N(0, 1/0.15)", n_prior),
        transformation = fill("alpha", n_prior),
        value          = α_default
    ),
    DataFrame(
        type           = fill("new prior N(0, 1/0.5)", n_prior),
        transformation = fill("logit_alpha", n_prior),
        value          = logit_new
    ),
    DataFrame(
        type           = fill("new prior N(0, 1/0.5)", n_prior),
        transformation = fill("alpha", n_prior),
        value          = α_new
    )
)

# Split into 4 panels manually (Gadfly has no facet_grid)
df_default_logit = filter(row -> row.type == "default prior N(0, 1/0.15)" &&
                                 row.transformation == "logit_alpha",
                          df_prior)

df_default_alpha = filter(row -> row.type == "default prior N(0, 1/0.15)" &&
                                 row.transformation == "alpha",
                          df_prior)

df_new_logit = filter(row -> row.type == "new prior N(0, 1/0.5)" &&
                              row.transformation == "logit_alpha",
                      df_prior)

df_new_alpha = filter(row -> row.type == "new prior N(0, 1/0.5)" &&
                              row.transformation == "alpha",
                      df_prior)

p1 = plot(df_default_logit,
          x = :value, Geom.histogram(bincount=30),
          Guide.title("Default prior: logit(α)"),
          Guide.xlabel(""), Guide.ylabel(""))

p2 = plot(df_default_alpha,
          x = :value, Geom.histogram(bincount=30),
          Guide.title("Default prior: α"),
          Guide.xlabel(""), Guide.ylabel(""))

p3 = plot(df_new_logit,
          x = :value, Geom.histogram(bincount=30),
          Guide.title("New prior: logit(α)"),
          Guide.xlabel(""), Guide.ylabel(""))

p4 = plot(df_new_alpha,
          x = :value, Geom.histogram(bincount=30),
          Guide.title("New prior: α"),
          Guide.xlabel(""), Guide.ylabel(""))

prior_grid = vstack(hstack(p1, p2), hstack(p3, p4))
draw(SVG("prior_histograms.svg", 10inch, 6inch), prior_grid)

############################################################
# 7. Latent posterior p(x|y): mixture and R-INLA-like mode
############################################################

# Approximate p(x|y, α) for many values of α and build mixture over α
struct PostXRecord
    alpha       :: Float64
    x0          :: Vector{Float64}
    diag_sigma2 :: Vector{Float64}
    unn_log_post:: Float64
end

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

post_x = compute_post_x(alpha_vec)

# Extract matrices for mixture
mu_mat    = hcat([p.x0          for p in post_x]...)         # n × K
sigma2_mat= hcat([p.diag_sigma2 for p in post_x]...)         # n × K
lpost_latent = [p.unn_log_post  for p in post_x]

# normalize α posterior for mixture
lpost_latent .-= mean(lpost_latent)
Z_latent   = calc_Z(alpha_vec, lpost_latent)
w_alpha    = exp.(lpost_latent) ./ Z_latent   # mixture weights

# Mixture posterior mean and sd for each x_t
mu_post    = [sum(mu_mat[i, :]    .* w_alpha) for i in 1:n]
sigma_post = [sqrt(sum(((mu_mat[i, :] .- mu_post[i]).^2 .+ sigma2_mat[i, :]) .* w_alpha))
              for i in 1:n]

# "R-INLA-like": use α̂ at posterior mode and Laplace at that α̂
function latent_posterior_at_mode(α_hat::Float64)
    x_hat = calc_x0(α_hat)
    H_hat = calc_neg_hess_ff(x_hat, α_hat)
    Σ_hat = inv(Matrix(H_hat))
    mu_tight    = x_hat
    sigma_tight = sqrt.(diag(Σ_hat))
    return mu_tight, sigma_tight
end

mu_tight, sigma_tight = latent_posterior_at_mode(alpha_hat)

# Coverage diagnostics
inside_mix = (x_true .>= mu_post   .- 2 .* sigma_post) .&
             (x_true .<= mu_post   .+ 2 .* sigma_post)
coverage_mix = mean(inside_mix)

inside_tight = (x_true .>= mu_tight .- 2 .* sigma_tight) .&
               (x_true .<= mu_tight .+ 2 .* sigma_tight)
coverage_tight = mean(inside_tight)

println("Mixture INLA-from-scratch coverage: ", coverage_mix)
println("R-INLA-like mode-based coverage:   ", coverage_tight)

############################################################
# 8. Latent posterior plot (mixture + R-INLA-like + truth)
############################################################

df_latent = vcat(
    DataFrame(
        type = "mixture_inla_from_scratch",
        i    = 1:n,
        mu   = mu_post,
        lwr  = mu_post .- 2 .* sigma_post,
        upr  = mu_post .+ 2 .* sigma_post
    ),
    DataFrame(
        type = "rinla_like_mode",
        i    = 1:n,
        mu   = mu_tight,
        lwr  = mu_tight .- 2 .* sigma_tight,
        upr  = mu_tight .+ 2 .* sigma_tight
    ),
    DataFrame(
        type = "truth",
        i    = 1:n,
        mu   = x_true,
        lwr  = x_true,
        upr  = x_true
    )
)

latent_plot = plot(
    df_latent,
    x = :i,
    y = :mu,
    ymin = :lwr,
    ymax = :upr,
    color = :type,
    linetype = :type,
    group = :type,
    Geom.ribbon,
    Geom.line,
    Guide.xlabel("t"),
    Guide.ylabel("x"),
    Guide.colorkey(title=""),
    Theme(
        alphas = [0.20],
        line_width = 1.5pt
    )
)

draw(SVG("latent_posterior_tight.svg", 10inch, 4inch), latent_plot)

println("\nDone. Wrote:")
println("  simulated_data.svg")
println("  alpha_posterior.svg")
println("  prior_histograms.svg")
println("  latent_posterior_tight.svg")
println("\nPosterior mode α̂ = ", alpha_hat)
println("Mixture coverage   = ", coverage_mix)
println("R-INLA-like cov.   = ", coverage_tight)
