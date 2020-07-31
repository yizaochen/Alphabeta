function get_alpha_t0(w0, alpha_x, Qx, Nv)
    alpha = ones(Nv)
    temp = w0 .* alpha_x
    for idx_eigv in 1:Nv
        alpha[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
    end
    return alpha
end


function get_beta_t_tau(w0, beta_x, Qx, Nv)
    beta = ones(Nv)
    temp = w0 .* beta_x
    for idx_eigv in 1:Nv
        beta[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
    end
    return beta
end