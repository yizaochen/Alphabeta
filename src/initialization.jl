using LinearAlgebra

function get_alpha_t0(weight_Qx::Array{Float64,2}, rho_eq::Array{Float64,2})
    return transpose(weight_Qx) * rho_eq
end

function get_alpha_t0_x_square_norm(alpha_t0::Array{Float64,2}, Qx::Array{Float64,2}, w0::Array{Float64,2})
    alpha_t0_x = Qx * alpha_t0
    alpha_t0_x_square = alpha_t0_x.^2
    alpha_t0_norm = sqrt(sum(w0 .* alpha_t0_x_square))
    return alpha_t0_x, alpha_t0_x_square, alpha_t0_norm
end

function get_beta_t_tau(w0, beta_x, Qx, Nv)
    beta = ones(Nv)
    temp = w0 .* beta_x
    for idx_eigv in 1:Nv
        beta[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
    end
    return beta
end