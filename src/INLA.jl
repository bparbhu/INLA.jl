module INLA

using LinearAlgebra
using SparseArrays
using Random
using Distributions
using Statistics

export simulate_ar1, calc_Q, calc_lprior_rinla, calc_ljoint,
       calc_ff, calc_grad_ff, calc_neg_hess_ff,
       calc_x0, calc_lpost, calc_Z,
       compute_post_x, latent_posterior_at_mode,
       # new exports:
       INLALatentModel, AR1Model,
       INLALikelihood, BernoulliLogit, PoissonLog, GaussianIdentity,
       INLAResult, inla,
       SimpleSPDEModel

include("simulate.jl")
include("precision.jl")
include("priors.jl")
include("likelihood.jl")
include("mode_finding.jl")
include("hyperposterior.jl")
include("latentposterior.jl")

# new layers
include("models.jl")       # latent model registry (AR1, SPDE scaffold)
include("families.jl")     # likelihood families
include("quadrature.jl")   # adaptive quadrature
include("api.jl")          # high-level inla(...) function
include("spde.jl")         # SPDE / Matérn scaffold

end # module
