using DataFrames, DataFramesMeta, Gadfly, Random, Distributions, StatsBase, SparseArrays, Optim, RCall, Compose, LinearAlgebra, CategoricalArrays


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
    offset = 1e-10
    alpha = clamp(alpha, offset, 1-offset) # avoid exact 0 or 1

    z = x * alpha
    px = 1 ./ (1 .+ exp.(-z))
    logp = log.(clamp.(px, 1e-10, 1-1e-10))
    log1_p = log.(clamp.(1 .- px, 1e-10, 1-1e-10))

    ll_y = y .* logp + (1 .- y) .* log1_p
    ll_alpha = log(alpha) * (a - 1) + log(1 - alpha) * (b - 1)
    lp_alpha = ll_alpha + sum(ll_y)

    return lp_alpha
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
        g1 = calc_g1(x0)
        g2 = calc_g2(x0)
        
        x = (Q - Diagonal(g2)) \ (g1 .- x0 .* g2)
        if mean((x .- x0).^2 .< tol) > 0.5
            break
        end
        x0 = x
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
    log_det = 2 * sum(log.(diag(chol_h)))
    ljoint = calc_ljoint(y, x0, alpha)
    println("alpha: ", alpha, ", x0: ", x0, ", chol_h: ", chol_h, ", calc_neg_hess_ff(x0, alpha): ", calc_neg_hess_ff(x0, alpha))
    return ljoint - log_det
end


alpha_vec = range(-0.95, 0.95, length=31)
lpost = map(calc_lpost, alpha_vec)

#lpost .= lpost .- mean(lpost) # to avoid numerical overflow


lpost = [calc_lpost(alpha) for alpha in alpha_vec]
println(lpost)


# Normalization of the posterior
function calc_Z(alpha_vec, lpost_vec)
    nn = length(alpha_vec)
    hh = alpha_vec[2] - alpha_vec[1]
    ww = vcat(1, repeat([4, 2], Int((nn - 3) / 2)), [4, 1])

    # replace -Inf with very large negative number before exponentiation
    lpost_vec = replace(lpost_vec, -Inf => -1e300)
    lpost_max = maximum(lpost_vec)

    # use log-sum-exp trick to prevent overflow
    exp_lpost_sum = log(sum(ww .* exp.(lpost_vec .- lpost_max)))

    return exp_lpost_sum + lpost_max
end


Z = calc_Z(alpha_vec, lpost)

# plot the unnormalised log-posterior and the normalised posterior:
# Data frame for plotting
df_posterior = vcat(
    @linq DataFrame(alpha=alpha_vec, posterior=lpost,
                    type="unnormalised_log_posterior"),
    @linq DataFrame(alpha=alpha_vec, posterior=exp.(lpost .- Z),
                    type="normalised_posterior")
)


# Create a combined DataFrame for both unnormalized_log_posterior and normalized_posterior
combined_df = vcat(
    DataFrame(alpha=alpha_vec, posterior=lpost, type="unnormalized_log_posterior"),
    DataFrame(alpha=alpha_vec, posterior=exp.(lpost) ./ Z, type="normalized_posterior")
)

# Create vline_layer
vline_layer = layer(xintercept=[alpha_true], Geom.vline, Theme(line_style=[:dash], line_width=1mm, default_color=colorant"black"))



# Define a function to create custom ytick labels
# Define the y-ticks for the two types of plots
yticks_normalized_log_posterior = [-5000, -4000, -3000]
yticks_other = [0.0, 0.5, 1.0]

# Create two separate dataframes, one for each type of plot
df_posterior_normalized = DataFrame(alpha=alpha_vec, posterior=exp.(lpost) ./ Z, type="normalized_posterior")
df_posterior_other = DataFrame(alpha=alpha_vec, posterior=lpost, type="unnormalized_log_posterior")


println("df_posterior_unnormalized: ", size(df_posterior_normalized))
println("df_posterior_other: ", size(df_posterior_other))
println("Any missing in 'posterior' column of df_posterior_unnormalized: ", any(ismissing, df_posterior_normalized.posterior))
println("Any missing in 'posterior' column of df_posterior_other: ", any(ismissing, df_posterior_other.posterior))


# Create the two plots
plot_normalized = plot(df_posterior_normalized,
    x=:alpha, y=:posterior,
    Geom.line, Geom.point,
    layer(xintercept=[alpha_true], Geom.vline, Theme(line_style=[:dash], line_width=1mm, default_color=colorant"black")),
    Guide.xlabel("alpha_normalized"),
    Guide.yticks(ticks=yticks_normalized_log_posterior),
    Theme(panel_fill=colorant"transparent")
)
plot_other = plot(df_posterior_other,
    x=:alpha, y=:posterior,
    Geom.line, Geom.point,
    layer(xintercept=[alpha_true], Geom.vline, Theme(line_style=[:dash], line_width=1mm, default_color=colorant"black")),
    Guide.xlabel("alpha_unnormalized"),
    Guide.yticks(ticks=yticks_other),
    Theme(panel_fill=colorant"transparent")
)

# Combine the two plots
plot_combined = hstack(plot_normalized, plot_other)



# comparison with R-INLA
# Data frame for plotting
df_prior = vcat(
    DataFrame(logit_alpha=randn(10^4) .* sqrt(1/0.15), 
              type="default prior N(0,0.15)"),
    DataFrame(logit_alpha=randn(10^4) .* sqrt(1/0.5), 
              type="new prior N(0,0.5)")
)

@transform!(df_prior, alpha = (exp.(:logit_alpha) .- 1) ./ (exp.(:logit_alpha) .+ 1))

df_prior_long = stack(df_prior, [:logit_alpha, :alpha], :type)

# Renaming columns to match the R version
rename!(df_prior_long, :variable => :transformation, :value => :value)

# Reordering levels to match the R version
df_prior_long.transformation = categorical(df_prior_long.transformation, 
                                           ordered=true, 
                                           levels=["logit_alpha", "alpha"])




# Plot distributions of logit(alpha) and alpha for the two priors in 4 panels
df1 = df_prior_long[df_prior_long.type .== "default prior N(0,0.15)", :]
df2 = df_prior_long[df_prior_long.type .== "new prior N(0,0.5)", :]

p1 = plot(df1, x=:value, color=:transformation, Geom.histogram(bincount=30),
          Guide.xlabel("value"), Guide.ylabel(""),
          Coord.cartesian(xmin=-3, xmax=3),
          Scale.color_discrete_manual("blue", "red"),
          Guide.colorkey(title=""),
          Theme(key_position=:none))

p2 = plot(df2, x=:value, color=:transformation, Geom.histogram(bincount=30),
          Guide.xlabel("value"), Guide.ylabel(""),
          Coord.cartesian(xmin=-3, xmax=3),
          Scale.color_discrete_manual("blue", "red"),
          Guide.colorkey(title=""),
          Theme(key_position=:none))

# Stacking plots vertically
vstack(p1, p2)



R"""
library(INLA)
library(Matrix)
library(tidyverse)

set.seed(1234)
n = 100
alpha_true = .6
sigma_true = 0.1
beta_true = 10

# simulate x_t as AR1, p_t as logit(x_t), and y_t as Bernoulli(p_t)
x_true = arima.sim(n=n, model=list(ar=alpha_true), sd=sigma_true) %>% as.numeric
p_true = 1 / (1 + exp(-beta_true * x_true))
y = rbinom(n, 1, p_true)

# create data frame for plotting
df = data_frame(i = 1:n, x = x_true, p = p_true, y = y) %>% 
  gather(key='variable', value='value', -i) %>%
  mutate(variable = factor(variable, levels=c('x','p','y'))) 

# plot the time series x_t, p_t, and y_t in 3 panels
ggplot(df, aes(x=i, y=value, color=variable)) + 
  geom_line(na.rm=TRUE, show.legend=FALSE) + 
  facet_wrap(~variable, ncol=1, scales='free_y') + 
  xlim(0,100) + labs(x=NULL, y=NULL)


calc_Q = function(alpha) {
  1 / sigma_true^2 * Matrix::bandSparse(n, k=0:1, 
                                        diagonals = list(c(1, 1 + alpha^2, 1) %>% rep(c(1, n-2, 1)), 
                                                         -alpha %>% rep(n-1)), 
                                        symmetric=TRUE)
}

calc_lprior = function(alpha, a=1, b=1) {
  (a-1) * log((alpha + 1) / 2) + (b-1) * log(1 - (alpha+1)/2)
}

calc_ljoint = function(y, x, alpha, a=1, b=1) {
  chol_Q = calc_Q(alpha) %>% chol
  logdet_Q_half = chol_Q %>% diag %>% log %>% sum
  quad_form = crossprod(chol_Q %*% x) %>% drop
  res = 
    sum(beta_true * x * y - log1p(exp(beta_true * x))) + 
    logdet_Q_half - 0.5 * quad_form + 
    calc_lprior(alpha, a, b)
  return(res)
}

calc_ff = function(x, alpha) {
  sum(beta_true * x * y - log1p(exp(beta_true * x))) - 
    0.5 * drop(as.matrix(x %*% calc_Q(alpha) %*% x))
}

calc_grad_ff = function(x, alpha) {
  beta_true * y - 
    beta_true * exp(beta_true * x) / (1 + exp(beta_true * x)) - 
    drop(as.matrix(calc_Q(alpha) %*% x))
}

calc_neg_hess_ff = function(x, alpha) {
  calc_Q(alpha) + 
    diag(beta_true^2 * exp(beta_true * x) / (1 + exp(beta_true * x))^2)
}



calc_g0 = function(x) {
  sum(beta_true * x * y - log1p(exp(beta_true * x)))
}
calc_g1 = function(x) {
  beta_true * y - beta_true * exp(beta_true * x) / (1 + exp(beta_true * x)) 
}
calc_g2 = function(x) {
  (-1) * beta_true^2 * exp(beta_true * x) / (1 + exp(beta_true * x))^2
}

calc_x0 = function(alpha, tol=1e-12) {
  Q = calc_Q(alpha)
  x = x0 = rep(0, n)
  while(1) {
    g1 = calc_g1(x)
    g2 = calc_g2(x)
    x = drop(solve(Q - bandSparse(n=n, k=0, diagonals=list(g2))) %*% 
               (g1 - x0 * g2))
    if (mean((x-x0)^2 < tol)) {
      break
    } else {
      x0 = x
    }
  }
  return(x)
}


calc_x0_brute = function(alpha) {
  optim(par=rep(0, n), fn=calc_ff, gr=calc_grad_ff, alpha=alpha, 
        control=list(fnscale=-1), method='BFGS')$par
}

calc_lpost = function(alpha) {
  x0 = calc_x0(alpha)
  chol_h = chol(calc_neg_hess_ff(x0, alpha))
  calc_ljoint(y, x0, alpha) - sum(log(diag(chol_h)))
}
alpha_vec = seq(-.95, .95, len=31)
lpost = sapply(alpha_vec, calc_lpost)

calc_Z = function(alpha_vec, lpost_vec) {
  nn = length(alpha_vec)
  hh = alpha_vec[2] - alpha_vec[1]
  ww = c(1, rep(c(4,2), (nn-3)/2), c(4,1))
  return(sum(ww * exp(lpost_vec)) * hh / 3)
}

lpost = lpost - mean(lpost) # to avoid numerical overflow
Z = calc_Z(alpha_vec, lpost)

df_posterior = 
  bind_rows(
    data_frame(alpha=alpha_vec, posterior=lpost, 
               type='unnormalised_log_posterior'),
    data_frame(alpha=alpha_vec, posterior=exp(lpost)/Z, 
               type='normalised_posterior')) %>% 
  mutate(type = factor(type, levels=c('unnormalised_log_posterior', 
                                      'normalised_posterior')))

# plot unnormalised log posterior and normalised posterior in 2 panels
ggplot(df_posterior, aes(x=alpha, y=posterior)) + 
  geom_line() + geom_point() +
  geom_vline(aes(xintercept=alpha_true), linetype='dashed') + 
  facet_wrap(~type, scale='free_y', ncol=1) +
  theme(legend.position='none')

# data frame for plotting
df_prior = 
  bind_rows(
    data_frame(logit_alpha = rnorm(1e4,0,sqrt(1/0.15)), 
               type='default prior N(0,0.15)'), 
    data_frame(logit_alpha = rnorm(1e4,0,sqrt(1/0.5)), 
               type='new prior N(0,0.5)')) %>% 
  group_by(type) %>% 
  mutate(alpha = (exp(logit_alpha)-1)/(exp(logit_alpha) + 1)) %>%
  ungroup %>%
  gather(key='transformation', value='value', -type) %>%
  mutate(transformation = factor(transformation, levels=c('logit_alpha', 'alpha'))) 

# plot distributions of logit(alpha) and alpha for the two priors in 4 panels
ggplot(df_prior, aes(x=value)) + 
  geom_histogram(bins=30) + 
  facet_grid(type ~ transformation, scale='free') +
  labs(x=NULL, y=NULL)

theta1_true = log((1-alpha_true^2)/sigma_true^2) 
theta2_param = c(0, 0.5)
A_mat = diag(beta_true, n, n) %>% inla.as.sparse

inla_formula = 
  y ~ -1 + 
  f(i, model='ar1', hyper=list(
    theta1 = list(fixed=TRUE, initial=theta1_true),
    theta2 = list(param=theta2_param)))
inla_data = data_frame(i=1:n, y=y)

res = inla(
  formula=inla_formula, 
  data=inla_data, 
  family='binomial', 
  Ntrials=rep(1,n), 
  control.predictor=list(A = A_mat)
)
"""

res = R"res"

# Get the R-INLA marginals
marginals_hyperpar_df = DataFrame(marginals_hyperpar, [:x, :y])

df_compare = vcat(
    @linq DataFrame(alpha=marginals_hyperpar_df.x, posterior=marginals_hyperpar_df.y, type="R-INLA"),
    @linq DataFrame(alpha=alpha_vec, posterior=exp.(lpost) ./ Z, type="inla_from_scratch")
)

scratch_df = DataFrame(alpha=alpha_vec, posterior=exp.(lpost) ./ Z, type="inla_from_scratch")

# Plot posteriors for inla-from-scratch and R-INLA
df_concatenated = vcat([df[1] for df in df_compare]...) 

# plot
p1 = plot(df_concatenated, x=:alpha, y=:posterior, color=:type, Geom.line, Geom.point,
    Guide.xlabel("alpha"), Guide.ylabel("posterior"),
    Guide.colorkey(title=""),
    Scale.color_discrete_manual("blue", "red"))

p2 =  plot(scratch_df, x=:alpha, y=:posterior, color=:type, Geom.line, Geom.point,
    Guide.xlabel("alpha"), Guide.ylabel("posterior"),
    Guide.colorkey(title=""),
    Scale.color_discrete_manual("blue", "red"))


draw(SVG(6inch, 3inch), hstack(p1, p2))



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
