using Fretem, LinearAlgebra, SparseArrays, PhotonOperator
include("initialization.jl")


function initialize(Nh::Int64, Np::Int64, xratio::Int64, xavg::Int64)
    x, w, Ldx, L = getLagrange(Np, xratio/Nh)
    e_norm = x[end] - x[1]
    interpo_xs = x .+ x[end]
    N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
    return e_norm, interpo_xs, xref, w0
end

function get_mat_vec(Nv::Int64, tau::Int64)
    alpha_mat = zeros(Nv,tau+1)
    beta_mat = zeros(Nv,tau+1)
    Anorm_vec = ones(1,tau)
    return alpha_mat, beta_mat, Anorm_vec
end

function get_weight_Qx(N::Int64, Nv::Int64, w0::Array{Float64,2}, Qx::Array{Float64,2})
    weight_Qx = zeros(N, Nv)
    for i = 1:Nv
        weight_Qx[:, i] = w0 .* Qx[:, i]
    end
    return weight_Qx
end

function get_alpha_hat_e_delta_t(Lambdas::Array{Float64,1}, delta_t::Float64, alpha_hat::Array{Float64,2})
    expLQDT = exp.(-Lambdas .* delta_t)
    alpha_hat_e_delta_t = expLQDT .* alpha_hat
    sum_c2_c72_square = sum(alpha_hat_e_delta_t[2:end].^2)
    alpha_hat_e_delta_t[1] = sqrt(1 - sum_c2_c72_square)
    return alpha_hat_e_delta_t
end

function get_alpha_hat_e_delta_t_v1(expLQDT::Array{Float64,1}, alpha_hat::Array{Float64,2})
    alpha_hat_e_delta_t = expLQDT .* alpha_hat
    sum_c2_c72_square = sum(alpha_hat_e_delta_t[2:end].^2)
    alpha_hat_e_delta_t[1] = sqrt(1 - sum_c2_c72_square)
    return alpha_hat_e_delta_t
end

function get_e_delta_t_y_beta(Lambdas::Array{Float64,1}, delta_t::Float64, y_beta::Array{Float64,2})
    y_beta_norm = norm(y_beta)
    expLQDT = exp.(-Lambdas .* delta_t)
    e_delt_t_y_beta = expLQDT .* y_beta
    sum_c2_c72_square = sum(e_delt_t_y_beta[2:end].^2)
    e_delt_t_y_beta[1] = sqrt(y_beta_norm^2 - sum_c2_c72_square)
    return e_delt_t_y_beta
end

function get_normalized_beta(e_delta_t_y_beta_hat::Array{Float64,2}, photon_id::Int64, scale_factor_array::Array{Float64,1}, Qx::Array{Float64,2}, alpha_hat::Array{Float64,2}, w0::Array{Float64,2})
    beta_hat_first = e_delta_t_y_beta_hat ./ scale_factor_array[photon_id]
    beta_hat_first_square = (Qx * beta_hat_first) .^ 2
    alpha_hat_x_square = (Qx * alpha_hat) .^ 2
    return beta_hat_first ./ sqrt(sum(w0 .* alpha_hat_x_square .* beta_hat_first_square))
end

function get_posterior(alpha_hat::Array{Float64,2}, beta_hat::Array{Float64,2}, Qx::Array{Float64,2})
    alpha_hat_x = Qx * alpha_hat
    alpha_hat_x_square = alpha_hat_x .^ 2
    beta_hat_x = Qx * beta_hat
    beta_hat_x_square = beta_hat_x .^ 2
    return alpha_hat_x_square .* beta_hat_x_square
end

function forward(alpha_mat::Array{Float64,2}, atemp::Array{Float64,1}, tau::Int64, x_record::Array{Float64,2}, 
    LQ::Array{Float64,1}, Qx::Array{Float64,2}, dt::Float64, xref::Array{Float64,2}, e_norm::Float64, 
    interpo_xs::Array{Float64,1}, Np::Int64,  w0::Array{Float64,2}, Anorm_vec::Array{Float64,2})
    k_photon = 3 # unit: kcal/mol/angstrom^2
    expLQDT = exp.(-LQ .* dt)
    alpha_mat[:, 1] = atemp
    for alpha_idx in 1:tau
        y = x_record[alpha_idx+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_photon)

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

function forward_v1(alpha_mat::Array{Float64,2}, atemp::Array{Float64,1}, tau::Int64, x_record::Array{Float64,2}, 
    LQ::Array{Float64,1}, Qx::Array{Float64,2}, dt::Float64, xref::Array{Float64,2}, e_norm::Float64, 
    interpo_xs::Array{Float64,1}, Np::Int64,  w0::Array{Float64,2}, Anorm_vec::Array{Float64,2})
    k_photon = 3 # unit: kcal/mol/angstrom^2
    expLQDT = exp.(-LQ .* dt)
    alpha_mat[:, 1] = atemp # Record <alpha_0
    prev_ahat_edt = get_alpha_hat_e_delta_t_v1(expLQDT, atemp) # < alpha_0 | exp(-H dt)
    for alpha_idx in 1:tau
        y = x_record[alpha_idx+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_photon)
        psi_photon_psi = Qx' * photon_mat * Qx
        alpha_new =  psi_photon_psi * prev_ahat_edt # < alpha_{t-1} | exp(-H dt) y

        # Normalization
        Anorm_vec[alpha_idx] = norm(alpha_new)
        atemp = alpha_new ./ Anorm_vec[alpha_idx]
        alpha_mat[:, alpha_idx+1] = atemp

        # < alpha | exp(-H dt)
        prev_ahat_edt = get_alpha_hat_e_delta_t_v1(expLQDT, atemp)
    end
    return alpha_mat, Anorm_vec, atemp
end

function get_LQ_diff_ij(Nv::Int64, LQ::Array{Float64,1})
    LQ_diff_ij = zeros(Nv,Nv)
    for i in 1:Nv
        LQ_diff_ij[i,:] = 1 ./ (LQ .- LQ[i])
        LQ_diff_ij[i,i] = 0
    end
    return LQ_diff_ij
end

function backward(LQ::Array{Float64,1}, dt::Float64, Nv::Int64, beta_mat::Array{Float64,2}, btemp::Array{Float64,1},
    tau::Int64, x_record::Array{Float64,2}, alpha_mat::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64,
    interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, Qx::Array{Float64,2}, Anorm_vec::Array{Float64,2})
    k_photon = 3 # unit: kcal/mol/angstrom^2
    LQ_diff_ij = get_LQ_diff_ij(Nv, LQ) # Eq. (63) in JPCB 2013

    expLQDT = exp.(-LQ .* dt)
    someones = ones(1,Nv)
    eLQDT = expLQDT * someones

    #btemp = btemp ./ Anorm_vec[tau+2] ./ Anorm_vec[tau+1]
    beta_mat[:, end] = btemp
    exp_ab_mat = zeros(Nv,Nv)
    for beta_idx in tau:-1:1
        y = x_record[beta_idx+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_photon)

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

function backward_v1(LQ::Array{Float64,1}, dt::Float64, Nv::Int64, beta_mat::Array{Float64,2}, btemp::Array{Float64,1},
    tau::Int64, x_record::Array{Float64,2}, alpha_mat::Array{Float64,2}, xref::Array{Float64,2}, e_norm::Float64,
    interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, Qx::Array{Float64,2}, Anorm_vec::Array{Float64,2})
    k_photon = 3 # unit: kcal/mol/angstrom^2
    LQ_diff_ij = get_LQ_diff_ij(Nv, LQ) # Eq. (63) in JPCB 2013

    expLQDT = exp.(-LQ .* dt)
    someones = ones(1,Nv)
    eLQDT = expLQDT * someones

    #btemp = btemp ./ Anorm_vec[tau+2] ./ Anorm_vec[tau+1]
    beta_mat[:, end] = btemp
    exp_ab_mat = zeros(Nv,Nv)
    for beta_idx in tau:-1:1
        y = x_record[beta_idx+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_photon)
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


function forward_backward(Nh::Int64, Np::Int64, xratio::Int64, xavg::Int64, peq::Array{Float64,2}, D::Float64, 
    Nv::Int64, tau::Int64, x_record::Array{Float64,2}, dt::Float64)
    e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)
    LQ, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D, Nv)
    N  = Nh*Np - Nh + 1 # Total number of nodes
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx)

    alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau)

    alpha_t0 = get_alpha_t0(weight_Qx, rho)
    alpha_mat, Anorm_vec, atemp = forward_v1(alpha_mat, atemp, tau, x_record, LQ, Qx, dt, xref, e_norm, interpo_xs, Np, w0, Anorm_vec)
    log_likelihood = sum(log.(Anorm_vec)) # Eq. (41)

    btemp = get_beta_T(Nv, weight_Qx);
    #exp_ab_mat = backward(LQ, dt, Nv, beta_mat, btemp, tau, x_record, alpha_mat, xref, e_norm, interpo_xs, Np, w0, Qx, Anorm_vec)

    # Eq. (72) and Eq. (78)
    peq_new = diag(Qx * exp_ab_mat * Qx')
    peq_new_normalize = peq_new ./ sum(w0 .* peq_new)
    return peq_new_normalize, log_likelihood
end


function backward_with_betamatrix(LQ::Array{Float64,1}, dt::Float64, Nv::Int64, beta_mat::Array{Float64,2}, 
    btemp::Array{Float64,1}, tau::Int64, x_record::Array{Float64,2}, alpha_mat::Array{Float64,2}, 
    xref::Array{Float64,2},e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, w0::Array{Float64,2}, 
    Qx::Array{Float64,2}, Anorm_vec::Array{Float64,2})
    k_photon = 3 # unit: kcal/mol/angstrom^2
    LQ_diff_ij = get_LQ_diff_ij(Nv, LQ) # Eq. (63) in JPCB 2013

    expLQDT = exp.(-LQ .* dt)
    someones = ones(1,Nv)
    eLQDT = expLQDT * someones

    #btemp = btemp ./ Anorm_vec[tau+2] ./ Anorm_vec[tau+1]
    beta_mat[:, end] = btemp
    exp_ab_mat = zeros(Nv,Nv)
    for beta_idx in tau:-1:1
        y = x_record[beta_idx+1]
        photon_mat = get_photon_matrix_gaussian(y, xref, e_norm, interpo_xs, Np, w0, k_photon)

        psi_photon_psi = Qx' * photon_mat * Qx
        btemp = psi_photon_psi * btemp
        
        # Eq. (64) and Eq. (63)
        outer= alpha_mat[:, beta_idx] * btemp'
        exp_ab_mat = exp_ab_mat .+ outer .* ( diagm(expLQDT * dt) + LQ_diff_ij .* (eLQDT-eLQDT'))
        
        btemp = expLQDT .* btemp
        btemp = btemp / Anorm_vec[beta_idx]
        beta_mat[:, beta_idx] = btemp
    end
    return exp_ab_mat, beta_mat
end


function get_all_likelihood(Nh::Int64, Np::Int64, xratio::Int64, xavg::Int64, peq::Array{Float64,2}, D::Float64, 
    Nv::Int64, tau::Int64, x_record::Array{Float64,2}, dt::Float64)
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
    
    exp_ab_mat, beta_mat = backward_with_betamatrix(LQ, dt, Nv, beta_mat, btemp, tau, x_record, alpha_mat, xref, e_norm, interpo_xs, Np, w0, Qx, Anorm_vec)

    #log_likelihood = -sum(log.(Anorm_vec)) # Eq. (41)
    #likelihood = exp(log_likelihood) # Positive overflow....
    #Q = get_Q(LQ, dt, alpha_mat, beta_mat, tau)
    #S = log_likelihood - Q
    return Anorm_vec
end


function get_Q(LQ::Array{Float64,1}, dt::Float64, alpha_mat::Array{Float64,2}, beta_mat::Array{Float64,2}, tau::Int64)
    container = zeros(1, tau)
    expLQDT = exp.(-LQ .* dt)
    for alpha_idx in 1:tau
        atemp = alpha_mat[:, alpha_idx]
        btemp = beta_mat[:, alpha_idx+1]
        prev_ahat_edt = expLQDT .* atemp
        container[alpha_idx] = dot(prev_ahat_edt, btemp)
    end
    #return sum(container) / likelihood
    return sum(container)
end