# mode_finding.jl

"""
    calc_x0(α; tol=1e-12, maxiter=1000)

Iterative mode finder for f(x; α) using Siegert’s Newton-like scheme.
Returns the mode x₀(α).
"""
function calc_x0(α::Float64; tol::Float64=1e-12, maxiter::Int=1000)
    Q  = calc_Q(α)
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
