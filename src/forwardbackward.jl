using Fretem, LinearAlgebra, SparseArrays, PhotonOperator
include("initialization.jl")


function initialize(Nh, Np, xratio, xavg)
    x, w, Ldx, L = getLagrange(Np, xratio/Nh)
    e_norm = x[end] - x[1]
    interpo_xs = x .+ x[end]
    N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
    return e_norm, interpo_xs, xref, w0
end


function get_mat_vec(Nv, tau)
    alpha_mat = zeros(Nv,tau+1)
    beta_mat = zeros(Nv,tau+1)
    Anorm_vec = ones(1,tau+2)
    return alpha_mat, beta_mat, Anorm_vec
end


function forward(alpha_mat, atemp, tau, x_record, LQ, Qx, dt, xref, e_norm, interpo_xs, Np, w0, Anorm_vec)
    expLQDT = exp.(-LQ .* dt)
    alpha_mat[:, 1] = atemp
    for alpha_idx in 1:tau
        y = x_record[alpha_idx+1]
        photon_mat = get_photon_matrix(y, xref, e_norm, interpo_xs, Np, w0)

        # < alpha | exp(-H dt)
        prev_ahat_edt = expLQDT .* atemp

        # < alpha | exp(-H dt) y
        psi_photon_psi = Qx' * photon_mat * Qx
        alpha_new =  psi_photon_psi * prev_ahat_edt

        # Normalization
        Anorm_vec[alpha_idx+1] = norm(alpha_new)
        atemp = alpha_new / Anorm_vec[alpha_idx+1]

        alpha_mat[:, alpha_idx+1] = atemp
    end
    return alpha_mat, Anorm_vec, atemp
end


function get_LQ_diff_ij(Nv, LQ)
    LQ_diff_ij = zeros(Nv,Nv)
    for i in 1:Nv
        LQ_diff_ij[i,:] = 1 ./ (LQ .- LQ[i])
        LQ_diff_ij[i,i] = 0
    end
    return LQ_diff_ij
end


function backward(LQ, dt, Nv, beta_mat, btemp, tau, x_record, alpha_mat, xref, e_norm, interpo_xs, Np, w0, Qx, Anorm_vec)
    LQ_diff_ij = get_LQ_diff_ij(Nv, LQ) # Eq. (63) in JPCB 2013

    expLQDT = exp.(-LQ .* dt)
    someones = ones(1,Nv)
    eLQDT = expLQDT * someones

    #btemp = btemp ./ Anorm_vec[tau+2] ./ Anorm_vec[tau+1]
    beta_mat[:, end] = btemp
    exp_ab_mat = zeros(Nv,Nv)
    for beta_idx in tau:-1:1
        y = x_record[beta_idx+1]
        photon_mat = get_photon_matrix(y, xref, e_norm, interpo_xs, Np, w0)

        psi_photon_psi = Qx' * photon_mat * Qx
        btemp = psi_photon_psi * btemp
        
        # Eq. (64) and Eq. (63)
        outer= alpha_mat[:, beta_idx] * btemp'
        exp_ab_mat = exp_ab_mat .+ outer .* ( diagm(expLQDT * dt) + LQ_diff_ij .* (eLQDT-eLQDT'))
        
        btemp = expLQDT .* btemp
        btemp = btemp / Anorm_vec[beta_idx]
        beta_mat[:, beta_idx] = btemp
    end
    return exp_ab_mat
end


function forward_backward(Nh, Np, xratio, xavg, peq, D, Nv, tau, x_record, dt)
    e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)
    LQ, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D, Nv)

    alpha_t0 = get_alpha_t0(w0, rho, Qx, Nv)
    beta_t_tau = get_beta_t_tau(w0, rho, Qx, Nv)
    atemp = alpha_t0
    btemp = beta_t_tau

    alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau)

    Anorm_vec[1] = norm(alpha_t0)

    alpha_mat, Anorm_vec, atemp = forward(alpha_mat, atemp, tau, x_record, LQ, Qx, dt, xref, e_norm, interpo_xs, Np, w0, Anorm_vec)

    Anorm_vec[tau+2] = sum(btemp .* atemp)
    atemp = atemp ./ Anorm_vec[end]
    #log_likelihood = -sum(log.(Anorm_vec)) # Eq. (41)

    exp_ab_mat = backward(LQ, dt, Nv, beta_mat, btemp, tau, x_record, alpha_mat, xref, e_norm, interpo_xs, Np, w0, Qx, Anorm_vec)

    # Eq. (72) and Eq. (78)
    peq_new = diag(Qx * exp_ab_mat * Qx')
    peq_new_normalize = peq_new ./ sum(w0 .* peq_new)
    return peq_new_normalize
end