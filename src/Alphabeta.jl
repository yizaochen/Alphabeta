module Alphabeta

    export get_alpha_t0_x_by_V0_Veq, get_alpha_by_proj_alphax_to_Qx, proj_vector_from_eigenspace_to_xspace, em_iteration

    include("forwardbackward.jl")
    export get_weight_Qx, get_alpha_t0, get_alpha_t0_x_square_norm, get_beta_T, get_alpha_hat_e_delta_t, get_e_delta_t_y_beta,get_beta_t_tau, get_alpha_t0_x_square_norm, forward_backward_v0, forward_backward_v1, forward_backward_v2,get_normalized_beta, get_posterior, get_loglikelihood, optimize_D

    include("abruptdetect.jl")
    export detect_abrupt

    include("smooth.jl")
    export smooth_psi, smooth_peq

    include("evaluation.jl")
    export iteration_evaluation

    include("plot_util.jl")
    export plot_alpha_t0, plot_alpha_t0_e_dt, plot_photon_mat, plot_alpha_hat, plot_alpha_hat_e_dt, plot_beta_T, plot_beta_T_minus_1, plot_beta_hat_posterior

    using Printf, JLD

    """
    proj_alpha_from_eigenspace_to_xspace(Qx, alpha)

    Qx: N_x times N_v matrix
    where N_x is the number of collocation points 
         and N_v is the number of eigenvectors 
    vector is a column vector
    """
    function proj_vector_from_eigenspace_to_xspace(Qx, vector)
        return Qx * vector
    end
    

    function get_alpha_t0_x_by_V0_Veq(w0, V0, Veq)
        C0 = 1 / sum(w0 .* exp.(-V0))
        Ceq = 1 / sum(w0 .* exp.(-Veq))
        factor = C0 / sqrt(Ceq)
        exp_term = exp.(-V0 .+ (Veq ./ 2))
        exp_term = max.(exp_term, 1e-10)
        return factor .* exp_term
    end


    function get_alpha_by_proj_alphax_to_Qx(w0, alpha_x, Qx, Nv)
        alpha = ones(Nv)
        temp = w0 .* alpha_x
        for idx_eigv in 1:Nv
            alpha[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
        end
        return alpha
    end

    function em_iteration(n_iteration::Int64, N::Int64, p0::Array{Float64,2}, 
        Nh::Int64, Np::Int64, xratio::Int64, xavg::Int64, D::Float64, Nv::Int64, tau::Int64,
        y_record::Array{Float64,2}, save_freq::Float64, xref::Array{Float64,2}, e_norm::Float64, w0::Array{Float64,2}, f_out_pcontain::String, f_out_l_record::String)

        # Initailize container
        p_container = zeros(Float64, n_iteration+1, N)
        log_likelihood_records = zeros(n_iteration+1)
        
        # Iteration of EM
        p_prev = p0  # initial guess
        p_container[1, :] = p0 # The first row in container is p0
        for iter_id = 1:n_iteration
            println(@sprintf "Iteration-ID: %d" iter_id)
            # Every 5 iterations, check abrupt change and do smooth
            if iter_id % 5 == 0
                abrupt_boolean, idx_larger_than_1 = detect_abrupt(xref[:,1], p_prev[:,1], N, e_norm)
                p_prev[:,1] = smooth_peq(N, p_prev[:,1], abrupt_boolean, w0, xref)
            end
            p_em, log_likelihood = forward_backward_v2(Nh, Np, xratio, xavg, p_prev, D, Nv, tau, y_record, save_freq)
            p_em = max.(p_em, 1e-10)   
            p_container[iter_id+1, :] = p_em    
            p_prev[:,1] = p_em
            log_likelihood_records[iter_id] = log_likelihood
        end
        
        # Output
        save(f_out_pcontain, "p_container", p_container)
        println(@sprintf "Write p_container to %s" f_out_pcontain)

        save(f_out_l_record, "log_likelihood_records", log_likelihood_records)
        println(@sprintf "Write log_likelihood_records to %s" f_out_l_record)

        return p_container, log_likelihood_records
    end
    
end
