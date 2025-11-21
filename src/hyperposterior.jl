# hyperposterior.jl

"""
    calc_lpost(α)

Approximate log p(α | y) up to an additive constant, using
Laplace approximation for p(x | y, α).
"""
function calc_lpost(α::Float64)
    x0 = calc_x0(α)
    H  = calc_neg_hess_ff(x0, α)
    F  = cholesky(Symmetric(Matrix(H)))
    # log |H|^{1/2} = ∑ log diag(U) when H = U'U
    logdet_H_half = sum(log.(diag(F.U)))
    return calc_ljoint(y, x0, α) - logdet_H_half
end

"""
    calc_Z(alpha_vec, lpost_vec)

Composite Simpson rule for the normalization constant:

Z = ∫ exp(log p(α | y)) dα ≈ Σ w_i exp(lpost_i).
"""
function calc_Z(alpha_vec::AbstractVector{<:Real},
                lpost_vec::AbstractVector{<:Real})
    nn = length(alpha_vec)
    hh = alpha_vec[2] - alpha_vec[1]
    ww = vcat(1.0,
              collect(Iterators.flatten(fill.([4.0, 2.0], (nn - 3) ÷ 2))),
              4.0, 1.0)
    return sum(ww .* exp.(lpost_vec)) * hh / 3.0
end
