using LinearAlgebra

function get_alpha_t0(weight_Qx::Array{Float64,2}, rho_eq::Array{Float64,2})
    return transpose(weight_Qx) * rho_eq
end

function get_beta_T(Nv::Int64, weight_Qx::Array{Float64,2})
    beta_T = zeros(Nv,1)
    for idx=1:Nv
        beta_T[idx] = sum(weight_Qx[:, idx])
    end
    return beta_T
end

function get_alpha_t0_x_square_norm(alpha_t0::Array{Float64,2}, Qx::Array{Float64,2}, w0::Array{Float64,2})
    alpha_t0_x = Qx * alpha_t0
    alpha_t0_x_square = alpha_t0_x.^2
    alpha_t0_norm = sqrt(sum(w0 .* alpha_t0_x_square))
    return alpha_t0_x, alpha_t0_x_square, alpha_t0_norm
end

function get_beta_t_tau(w0, beta_x, Qx, Nv)
    beta = ones(Nv,1)
    temp = w0 .* beta_x
    for idx_eigv in 1:Nv
        beta[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
    end
    return beta
end

function gaussian(x::Array{Float64,1}, μ::Float64, σ::Float64)
    prefactor = (σ * √(2π))^(-1)
    inner_array = ((x .- μ) ./ σ) .^ 2
    inner_array = -0.5 * inner_array
    exp_term = exp.(inner_array)
    return prefactor * exp_term
end

function gaussian_kde(xref::Array{Float64,2}, y_record::Array{Float64,2}, σ::Float64, w0::Array{Float64,2})
    N = size(xref)[1]
    peq_kde_estimate = zeros(N, 1)
    for μ in y_record
        peq_kde_estimate[:, 1] = peq_kde_estimate[:, 1] .+ gaussian(xref[:, 1], μ, σ)
    end
    peq_kde_estimate[:, 1] = peq_kde_estimate[:, 1] ./ sum(w0 .* peq_kde_estimate[:, 1])
    peq_kde_estimate[:, 1] = max.(peq_kde_estimate[:, 1], 1e-10) 
    return peq_kde_estimate
end

function get_D_by_Stokes_Einstein_relation(a::Float64)
    """
    a: Radius of brownian particle, input unit: Å
    """
    kBT = 4.11e-21 # unit: J,   T=298K
    a = a * 1e-10  # Convert from Å to m
    η = 9e-4 # water viscosity, unit: Pa⋅s
    D = kBT / (6π * (η * a))
    return D
end