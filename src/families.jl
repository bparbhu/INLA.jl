# families.jl

abstract type INLALikelihood end

struct BernoulliLogit    <: INLALikelihood end
struct PoissonLog        <: INLALikelihood end
struct GaussianIdentity  <: INLALikelihood end

"""
    default_likelihood_symbol(::INLALatentModel)

Default likelihood for a given latent model.
"""
default_likelihood_symbol(::INLALatentModel) = :bernoulli_logit

"""
    make_likelihood(sym::Symbol)

Construct a likelihood type from a Symbol.
"""
make_likelihood(::Val{:bernoulli_logit}) = BernoulliLogit()
make_likelihood(::Val{:poisson_log})     = PoissonLog()
make_likelihood(::Val{:gaussian})        = GaussianIdentity()

function make_likelihood(sym::Symbol)
    if sym == :bernoulli_logit
        return make_likelihood(Val(:bernoulli_logit))
    elseif sym == :poisson_log
        return make_likelihood(Val(:poisson_log))
    elseif sym == :gaussian
        return make_likelihood(Val(:gaussian))
    else
        error("Unknown likelihood symbol: $sym")
    end
end

"""
    check_supported(lik::INLALikelihood)

For now, only BernoulliLogit is fully wired into your calc_* stack.
"""
function check_supported(lik::INLALikelihood)
    if lik isa BernoulliLogit
        return
    else
        error("Likelihood $(typeof(lik)) is not yet wired into the core INLA stack.")
    end
end
