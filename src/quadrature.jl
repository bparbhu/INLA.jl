# quadrature.jl

"""
    adaptive_simpson(f, a, b; tol=1e-5, maxdepth=10)

Adaptive Simpson integration of f over [a,b].
Returns an approximation to ∫_a^b f(x) dx.
"""
function adaptive_simpson(f, a, b; tol=1e-5, maxdepth=10)
    fa = f(a)
    fb = f(b)
    m  = (a + b) / 2
    fm = f(m)
    S  = (b - a) * (fa + 4fm + fb) / 6
    return _adaptive_simpson(f, a, b, fa, fb, fm, S, tol, maxdepth)
end

function _adaptive_simpson(f, a, b, fa, fb, fm, S, tol, depth)
    m  = (a + b) / 2
    lm = (a + m) / 2
    rm = (m + b) / 2

    flm = f(lm)
    frm = f(rm)

    Sleft  = (m - a) * (fa + 4flm + fm) / 6
    Sright = (b - m) * (fm + 4frm + fb) / 6
    S2     = Sleft + Sright

    if depth <= 0 || abs(S2 - S) < 15 * tol
        return S2 + (S2 - S) / 15
    else
        return _adaptive_simpson(f, a, m, fa, fm, flm, Sleft,  tol/2, depth-1) +
               _adaptive_simpson(f, m, b, fm, fb, frm, Sright, tol/2, depth-1)
    end
end

"""
    calc_Z_adaptive(alpha_min, alpha_max, calc_lpost; tol=1e-5)

Compute Z = ∫ exp(log p(α | y)) dα with adaptive Simpson.
`calc_lpost` is your existing function log p(α | y) up to constant.
"""
function calc_Z_adaptive(alpha_min::Float64,
                         alpha_max::Float64,
                         calc_lpost::Function;
                         tol=1e-5)
    f(α) = exp(calc_lpost(α))
    return adaptive_simpson(f, alpha_min, alpha_max; tol=tol)
end
