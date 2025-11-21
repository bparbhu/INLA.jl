# precision.jl

"""
    calc_Q(α)

Sparse AR(1) precision matrix Q(α) for the latent x, using
global constants `n` and `sigma_true`.

This matches Siegert’s derivation: main diagonal [1, 1+α², …, 1+α², 1],
off-diagonals -α, scaled by 1/σ².
"""
function calc_Q(α::Float64)
    main = vcat([1.0], fill(1.0 + α^2, n - 2), [1.0])
    off  = fill(-α, n - 1)
    Q = spdiagm(0 => main, 1 => off, -1 => off)
    return Q / sigma_true^2
end
