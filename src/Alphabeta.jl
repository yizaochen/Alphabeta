module Alphabeta

    export get_alpha_t0_x_by_V0_Veq, get_alpha_by_proj_alphax_to_Qx, proj_vector_from_eigenspace_to_xspace

    include("forwardbackward.jl")
    export get_alpha_t0, get_beta_t_tau, forward_backward

    include("abruptdetect.jl")
    export detect_abrupt

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

    
end
