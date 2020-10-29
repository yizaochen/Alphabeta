module Alphabeta

    export get_alpha_t0_x_by_V0_Veq, get_alpha_by_proj_alphax_to_Qx, proj_vector_from_eigenspace_to_xspace, em_iteration

    include("forwardbackward.jl")
    export get_alpha_t0, get_beta_t_tau, forward_backward

    include("abruptdetect.jl")
    export detect_abrupt

    include("smooth.jl")
    export smooth_psi

    include("evaluation.jl")
    export iteration_evaluation

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
        y_record::Array{Float64,2}, save_freq::Float64, xref::Array{Float64,2}, e_norm::Float64, f_out::String)
        # Initailize container
        p_container = zeros(Float64, n_iteration+1, N)
        
        # Iteration of EM
        p_prev = p0  # initial guess
        p_container[1, :] = p0 # The first row in container is p0
        for iter_id = 1:n_iteration
            println(iter_id)
            # Every 5 iterations, check abrupt change and do smooth
            if iter_id % 5 == 0
                abrupt_boolean, idx_larger_than_1 = detect_abrupt(xref[:,1], p_prev[:,1], N, e_norm)
                p_prev[:,1] = smooth_psi(N, p_prev[:,1], abrupt_boolean, idx_larger_than_1, xref[:,1])
            end
            p_em = forward_backward(Nh, Np, xratio, xavg, p_prev, D, Nv, tau, y_record, save_freq)
            p_em = max.(p_em, 1e-10)   
            p_container[iter_id+1, :] = p_em    
            p_prev[:,1] = p_em
        end
        
        # Output
        save(f_out, "p_container", p_container)
        println(@sprintf "Write p_container to %s" f_out)
        return p_container
    end
    
end
