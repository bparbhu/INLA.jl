using DataFrames, DataFramesMeta, Gadfly, Random, Distributions, StatsBase, SparseArrays, Optim, RCall, Compose

Random.seed!(1234)
n = 100
alpha_true = 0.6
sigma_true = 0.1
beta_true = 10

# simulate x_t as AR1, p_t as logit(x_t), and y_t as Bernoulli(p_t)
function simulate_ar1(n, alpha, sigma)
    x = Vector{Float64}(undef, n)
    x[1] = randn() * sigma
    for t in 2:n
        x[t] = alpha * x[t-1] + randn() * sigma
    end
    return x
end

x_true = simulate_ar1(n, alpha_true, sigma_true)
p_true = 1 ./ (1 .+ exp.(-beta_true * x_true))
y = rand.(Bernoulli.(p_true))

# create data frame for plotting
df = DataFrame(i = 1:n, x = x_true, p = p_true, y = y)
df_long = stack(df, [:x, :p, :y], :i, variable_name=:variable, value_name=:value)

# plot the time series x_t, p_t, and y_t in 3 panels
# Separate data for each variable
df_x = filter(row -> row[:variable] == "x", df_long)
df_p = filter(row -> row[:variable] == "p", df_long)
df_y = filter(row -> row[:variable] == "y", df_long)

# Create plots for each variable
plot_x = plot(df_x, x=:i, y=:value, color=:variable, Geom.line, Coord.cartesian(xmin=0, xmax=100), Guide.ylabel("x"))
plot_p = plot(df_p, x=:i, y=:value, color=:variable, Geom.line, Coord.cartesian(xmin=0, xmax=100), Guide.ylabel("p"))
plot_y = plot(df_y, x=:i, y=:value, color=:variable, Geom.line, Coord.cartesian(xmin=0, xmax=100), Guide.ylabel("y"))

# Combine the plots vertically
final_plot = vstack(plot_x, plot_p, plot_y)


# The sparse precision matrix Q
function calc_Q(alpha)
    diagonal_values = vcat(1, repeat([1 + alpha^2], n - 2), 1)
    off_diagonal_values = repeat([-alpha], n - 1)
    Q = spdiagm(0 => diagonal_values, 1 => off_diagonal_values, -1 => off_diagonal_values)
    return Q / sigma_true^2
end


# get the log prior for alpha
function calc_lprior(alpha, a=1, b=1)
    return (a - 1) * log((alpha + 1) / 2) + (b - 1) * log(1 - (alpha + 1) / 2)
end


# calculate the log joint distribution
function calc_ljoint(y, x, alpha, a=1, b=1)
    chol_Q = cholesky(calc_Q(alpha))
    logdet_Q_half = sum(log.(diag(chol_Q.L)))
    quad_form = dot(chol_Q.L * x, x)
    
    res = sum(beta_true .* x .* y .- log1p.(exp.(beta_true .* x))) +
          logdet_Q_half - 0.5 * quad_form +
          calc_lprior(alpha, a, b)
    return res
end

# the function f(x) and its negative hessian
function calc_ff(x, alpha)
    return sum(beta_true .* x .* y .- log1p.(exp.(beta_true .* x))) -
           0.5 * dot(x, calc_Q(alpha) * x)
end

function calc_grad_ff(x, alpha)
    return beta_true .* y .-
           beta_true .* exp.(beta_true .* x) ./ (1 .+ exp.(beta_true .* x)) .-
           calc_Q(alpha) * x
end

function calc_neg_hess_ff(x, alpha)
    return calc_Q(alpha) .+
           Diagonal(beta_true^2 .* exp.(beta_true .* x) ./ (1 .+ exp.(beta_true .* x)).^2)
end

# calculate the mode of f(x)
function calc_g0(x)
    return sum(beta_true .* x .* y .- log1p.(exp.(beta_true .* x)))
end

function calc_g1(x)
    return beta_true .* y .- beta_true .* exp.(beta_true .* x) ./ (1 .+ exp.(beta_true .* x))
end

function calc_g2(x)
    return -beta_true^2 .* exp.(beta_true .* x) ./ (1 .+ exp.(beta_true .* x)).^2
end

function calc_x0(alpha, tol=1e-12)
    Q = calc_Q(alpha)
    x = x0 = zeros(n)
    
    while true
        g1 = calc_g1(x)
        g2 = calc_g2(x)
        
        x = (Q - Diagonal(g2)) \ (g1 .- x0 .* g2)
        
        if mean((x .- x0).^2 .< tol) > 0.5
            break
        else
            x0 = x
        end
    end
    
    return x
end


function calc_x0_brute(alpha)
    function obj_func(x)
        return -calc_ff(x, alpha), -calc_grad_ff(x, alpha)
    end
    
    initial_x = zeros(n)
    result = optimize(obj_func, initial_x, BFGS())
    return result.minimizer
end


# Approximating the log posterior up to an additive constant
function calc_lpost(alpha)
    x0 = calc_x0(alpha)
    chol_h = cholesky(calc_neg_hess_ff(x0, alpha))
    return calc_ljoint(y, x0, alpha) - sum(log.(diag(chol_h.L)))
end

alpha_vec = range(-0.95, 0.95, length=31)
lpost = map(calc_lpost, alpha_vec)


# Normalisation of the posterior
function calc_Z(alpha_vec, lpost_vec)
    nn = length(alpha_vec)
    hh = alpha_vec[2] - alpha_vec[1]
    ww = vcat(1, repeat([4, 2], Int((nn - 3) / 2)), [4, 1])
    return sum(ww .* exp.(lpost_vec)) * hh / 3
end


lpost = lpost - mean(lpost) # to avoid numerical overflow
Z = calc_Z(alpha_vec, lpost)

# plot the unnormalised log-posterior and the normalised posterior:
# Data frame for plotting
Z = calc_Z(alpha_vec, lpost)
df_posterior = vcat(
    @linq DataFrame(alpha=alpha_vec, posterior=lpost,
                    type="unnormalised_log_posterior"),
    @linq DataFrame(alpha=alpha_vec, posterior=exp.(lpost) ./ Z,
                    type="normalised_posterior")
)

# Plot unnormalised log posterior and normalised posterior in 2 panels
plot(df_posterior,
    x=:alpha, y=:posterior, color=:type,
    Geom.line, Geom.point,
    Guide.xlabel("alpha"), Guide.ylabel("posterior"),
    layer(Geom.vline(xintercept=[alpha_true], linestyle=:dash)),
    Coord.cartesian(xmin=-0.95, xmax=0.95),
    Scale.color_discrete_manual("blue", "red"),
    Guide.title("Unnormalised Log Posterior and Normalised Posterior"),
    Guide.colorkey(title=""),
    Theme(key_position=:none),
    Facet.grid(:type .=> "unnormalised_log_posterior", :type .=> "normalised_posterior")
)


# comparison with R-INLA

using DataFrames, DataFramesMeta, Gadfly, StatsBase

# Data frame for plotting
df_prior = vcat(
    @linq DataFrame(logit_alpha=randn(10^4) .* sqrt(1/0.15),
                    type="default prior N(0,0.15)"),
    @linq DataFrame(logit_alpha=randn(10^4) .* sqrt(1/0.5),
                    type="new prior N(0,0.5)")
)

@transform!(df_prior, alpha = (exp.(:logit_alpha) .- 1) ./ (exp.(:logit_alpha) .+ 1))
df_prior_long = stack(df_prior, [:logit_alpha, :alpha], :type)

# Plot distributions of logit(alpha) and alpha for the two priors in 4 panels
plot(df_prior_long,
    x=:value, color=:variable, group=:type,
    Geom.histogram(bincount=30),
    Guide.xlabel("value"), Guide.ylabel(""),
    Coord.cartesian(xmin=-3, xmax=3),
    Scale.color_discrete_manual("blue", "red"),
    Guide.colorkey(title=""),
    Theme(key_position=:none),
    Facet.grid(:type .=> "default prior N(0,0.15)", :type .=> "new prior N(0,0.5)", :variable)
)



R"""
library(INLA)
"""

theta1_true = log((1 - alpha_true^2) / sigma_true^2)
theta2_param = [0, 0.5]

R"""
A_mat <- inla.as.sparse(diag($beta_true, $n, $n))
"""

inla_formula = "y ~ -1 + f(i, model='ar1', hyper=list(theta1 = list(fixed=TRUE, initial=$theta1_true), theta2 = list(param=$theta2_param)))"
inla_data = DataFrame(i=1:n, y=y)

R"""
res <- inla(
    formula=$inla_formula,
    data=$inla_data,
    family='binomial',
    Ntrials=rep(1, $n),
    control.predictor=list(A = $A_mat)
)
"""

res = R"res"

using DataFrames, DataFramesMeta, Gadfly, RCall

# Get the R-INLA marginals
marginals_hyperpar = rcopy(R"res$marginals.hyperpar$`Rho for i`")

df_compare = vcat(
    @linq DataFrame(alpha=marginals_hyperpar[:x], posterior=marginals_hyperpar[:y], type="R-INLA"),
    @linq DataFrame(alpha=alpha_vec, posterior=exp.(lpost) ./ Z, type="inla_from_scratch")
)

# Plot posteriors for inla-from-scratch and R-INLA
plot(df_compare,
    x=:alpha, y=:posterior, color=:type,
    Geom.line, Geom.point,
    Geom.vline(xintercept=alpha_true),
    Guide.xlabel("alpha"), Guide.ylabel("posterior"),
    Guide.colorkey(title=""),
    Scale.color_discrete_manual("blue", "red"),
    Theme(key_position=:top)
)


# Define a function to calculate posterior x given alpha
function calc_post_x(alpha_)
    mode_ = calc_x0(alpha_)
    chol_H_ = cholesky(calc_neg_hess_ff(mode_, alpha_))
    # Save alpha, the mode, the covariance matrix, and p(theta|y) unnormalized
    return Dict(
        :alpha => alpha_,
        :x0 => mode_,
        :diag_sigma => vec(inv(chol_H_)),
        :unn_log_post => calc_ljoint(y, mode_, alpha_) - sum(log.(diag(chol_H_)))
    )
end


# Approximate p(x | y, alpha) for many values of alpha
alpha_vec = range(-0.95, 0.95, length=31)
post_x = [calc_post_x(alpha_) for alpha_ in alpha_vec]

# Extract marginal means and variances for each p(x|y,theta)
mu = hcat([post["x0"] for post in post_x]...)
sigma2 = hcat([post["diag_sigma"] for post in post_x]...)

# Normalize the posterior
lpost = [post["unn_log_post"] for post in post_x]
lpost .-= mean(lpost)  # to avoid numerical overflow
post_alpha = exp.(lpost) ./ calc_Z(alpha_vec, lpost)

mu_post = [sum(mu[i, :] .* post_alpha) for i in 1:n]
sigma_post = [sqrt(sum(((mu[i, :] .- mu_post[i]).^2 .+ sigma2[i, :]) .* post_alpha)) for i in 1:n]


# Create a DataFrame for plotting
df_latent = vcat(
    DataFrame(
        type = "inla_from_scratch",
        i = 1:n,
        mu = mu_post,
        lwr = mu_post .- 2 .* sigma_post,
        upr = mu_post .+ 2 .* sigma_post
    ),
    DataFrame(
        type = "R-INLA",
        i = 1:n,
        mu = res[:summary][:random][:i][:mean],
        lwr = res[:summary][:random][:i][:mean] .- 2 .* res[:summary][:random][:i][:sd],
        upr = res[:summary][:random][:i][:mean] .+ 2 .* res[:summary][:random][:i][:sd]
    ),
    DataFrame(
        type = "truth",
        i = 1:n,
        mu = x_true,
        lwr = x_true,
        upr = x_true
    )
)

# Create ribbon plots for p(x|y) for inla-from-scratch and R-INLA, and true values
plot(df_latent,
    x=:i,
    Geom.ribbon(ymin=:lwr, ymax=:upr, alpha=:type, linetype=:type, color=:type),
    Geom.line(y=:mu, color=:type),
    Guide.xlabel(nothing),
    Guide.ylabel(nothing),
    Guide.colorkey(title=nothing),
    Guide.title(nothing),
    Guide.linetypekey(title=nothing),
    Guide.alpha_key(title=nothing)
)
