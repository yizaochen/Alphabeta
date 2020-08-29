module Fretdraw
using PyPlot, Printf, SparseArrays

    export plot_alpha_cycle

    function get_peq_c(Vref, w0)
        peq = exp.(-Vref)
        peq = max.(peq,1e-10)
        sum_factor = get_integral(w0, peq)
        c = 1 / sum_factor
        peq = peq ./ sum_factor
        return peq, c
    end

    function get_rhoeq(Vref, w0)
        peq, c = get_peq_c(Vref, w0)
        return sqrt.(peq)
    end

    function get_pref(Vref, w0)
        pref = exp.(-Vref)
        pref = max.(pref,1e-10)
        sum_factor = get_integral(w0, pref)
        pref = pref ./ sum_factor
        return pref
    end

    function get_coefficients_by_proj(w0, alpha, Qx, Nv)
        c_array = ones(Nv)
        temp = w0 .* alpha
        for idx_eigv in 1:Nv
            c_array[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
        end
        return c_array
    end

    function get_rho_t(c_array, dt, LQ, Qx)
        exp_lambda_mat = exp.(-LQ .* dt)
        c_array_after_exp_lambda = exp_lambda_mat .* c_array
        return Qx * c_array_after_exp_lambda
    end

    function get_coefficients_delta_function(x0, xref, e_norm, interpo_xs, Np, Qx, rho_eq, Nv)
        idx_xpos = find_nearest_point(x0, xref, e_norm, interpo_xs, Np)
        c_array = ones(Nv)
        for idx_eigv in 1:Nv
            c_array[idx_eigv] = Qx[idx_xpos, idx_eigv] / rho_eq[idx_xpos]
        end
        return c_array
    end


    function rebuild_p0_by_Qx_carray(Nv, c_array, Qx, rho_eq)
        N = size(Qx)[1]
        temp = zeros(N)
        for idx_eigv in 1:Nv
            temp = temp .+ (c_array[idx_eigv] .* Qx[:, idx_eigv])
        end
        return temp .* rho_eq
    end


    function find_nearest_point(x, xref, e_norm, interpo_xs, Np)
        x_left = xref[1] # The most left point
        diff = x - x_left
        n_element = floor(Int, diff / e_norm)
        node_left = x_left + n_element * e_norm
        points = node_left .+ interpo_xs
        min_idx = argmin(abs.(points .- x))
        idx = n_element * (Np - 1) + min_idx
        return idx
    end

    function get_photon_matrix(x, xref, e_norm, interpo_xs, Np, w0)
        idx = find_nearest_point(x, xref, e_norm, interpo_xs, Np)
        temp_vec = zeros(size(xref))
        temp_vec[idx] = 1
        temp_vec = w0 .* temp_vec
        photon_mat = spdiagm(0 => vec(temp_vec))
        return photon_mat
    end

    """
    get_integral(w, f)

    w: weight
    f: f(x)
    """
    function get_integral(w, f)
       return sum(w .* f) 
    end


    function get_alphat0(w0, V0, Veq)
        C0 = 1 / get_integral(w0, exp.(-V0))
        Ceq = 1 / get_integral(w0, exp.(-Veq))
        factor = C0 / sqrt(Ceq)
        exp_term = exp.(-V0 .+ (Veq ./ 2))
        exp_term = max.(exp_term, 1e-10)
        return factor .* exp_term
    end


    function get_p0_p0appr(w0, V0, Vref, Qx, Nv, rho_eq)
        alpha_t0 = get_alphat0(w0, V0, Vref)
        p0 = get_pref(V0, w0)
        c_array = get_coefficients_by_proj(w0, alpha_t0, Qx, Nv)
        p0_appr = rebuild_p0_by_Qx_carray(Nv, c_array, Qx, rho_eq)
        return p0, p0_appr
    end


    function plot_Vref(ax, xref, Vref, lbfz, ylims)
        ylabel = "\$ V(x) \$"
        ax.plot(xref, Vref)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        ax.set_ylim(ylims[1], ylims[2])
        return ax
    end


    """
    plot_V0(ax, xref, V0, k, lbfz, ylims, k_pos)

    """
    function plot_V0(ax, xref, V0, k, lbfz, ylims, k_pos)
        ylabel = "\$ V_0(x) \$"
        ax.plot(xref, V0)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        ax.set_ylim(ylims[1], ylims[2])
        txt = @sprintf "\$ V = %.3f (x-1)^2 \$" k
        ax.text(k_pos[1], k_pos[2], txt, fontsize=lbfz)
        return ax
    end   


    function plot_peq(ax, xref, Vref, w0, c1pos, intpos, lbfz)
        ylabel = "\$ p_{eq}(x) \$"
        peq, c = get_peq_c(Vref, w0)
        ax.plot(xref, peq)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        txt = @sprintf "\$ C = %.3f \$" c
        ax.text(c1pos[1], c1pos[2], txt, fontsize=lbfz)
        txt = @sprintf "\$ \\int w(x) p_{eq}(x) dx = %.3f \$" get_integral(w0, peq)
        ax.text(intpos[1], intpos[2], txt, fontsize=lbfz)
        return ax
    end


    function plot_Qx_by_idx(xref, Qx, idx, figsize)
        ylabel = @sprintf "\$ \\psi_{%d}(x) \$" idx
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xref, Qx[:, idx])
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        return ax
    end


    function plot_all_Qx(xref, Qx, figsize)
        fig, axes = plt.subplots(ncols=4, nrows=3, figsize=figsize)
        idx = 1
        for row_id in 1:3
            for col_id in 1:4
                ax = axes[row_id, col_id]
                ylabel = @sprintf "\$ \\psi_{%d}(x) \$" idx
                ax.plot(xref, Qx[:, idx])
                ax.set_xlabel("x")
                ax.set_ylabel(ylabel)
                idx += 1
            end
        end
        return axes
    end


    function plot_p0(ax, xref, Vref, w0, lbfz)
        ylabel = "\$ p(x,0) \$"
        peq, c = get_peq_c(Vref, w0)
        ax.plot(xref, peq)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        return ax
    end


    function plot_rhoeq(ax, xref, Vref, w0, lbfz)
        ylabel = "\$ \\rho_{eq}(x) \$"
        rho_eq = get_rhoeq(Vref, w0)
        ax.plot(xref, rho_eq)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        return ax
    end


    function plot_Vref_peq_exp_negativeV(xref, Vref, w0, figsize; c1pos=(1.3, 0.8), intpos=(1.25, 6), lbfz=10, vref_ylims=(0,8))
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=figsize)

        ax1 = plot_Vref(axes[1], xref, Vref, lbfz, vref_ylims)
        ax2 = plot_peq(axes[2], xref, Vref, w0, c1pos, intpos, lbfz)
        ax3 = plot_rhoeq(axes[3], xref, Vref, w0, lbfz)
        return fig, axes
    end


    function plot_V0_p0_alphat0(xref, V0, k, w0, Veq, figsize; lbfz=10, vref_ylims=(0,8), p_ylims=(0, 12), k_pos=(0.5, 7))
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=figsize)

        ax1 = plot_V0(axes[1], xref, V0, k, lbfz, vref_ylims, k_pos)
        ax2 = plot_p0(axes[2], xref, V0, w0, lbfz)
        ax3 = plot_alpha_t0(axes[3], xref, w0, V0, Veq, lbfz, p_ylims)
        return fig, axes
    end

    function plot_alpha_t0(ax, xref, w0, V0, Veq, lbfz, ylims)
        alpha_t0 = get_alphat0(w0, V0, Veq)
        ylabel = "\$ \\left< \\alpha_{t0}| x \\right> \$"
        ax.plot(xref, alpha_t0)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel(ylabel, fontsize=lbfz)
        ax.set_ylim(ylims[1], ylims[2])
        return ax
    end

    function plot_p0_pappr(xref, p0, p_appr, Nv, k0, figsize, lbfz)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
        ax.plot(xref, p0, label="\$ y = p(x,0)\$", color="red")
        ax.plot(xref, p_appr, label="\$ y = p_{appr}(x,0)\$", color="blue")
        ax.legend(frameon=false)
        ax.set_xlabel("x", fontsize=lbfz)
        ax.set_ylabel("y", fontsize=lbfz)
        title = @sprintf "\$ N_v=%d~~V_0=%.2f(x-1)^2  \$" Nv k0
        ax.set_title(title)
        return fig, ax
    end

    function plot_p0_pappr_by_ax(ax, xref, p0, p_appr, Nv, k0, lbfz, lgfz, row_id, col_id)
        ax.plot(xref, p0, label="\$ y = p(x,0)\$", color="red")
        ax.plot(xref, p_appr, label="\$ y = p_{appr}(x,0)\$", color="blue")
        ax.legend(frameon=false, fontsize=lgfz)
        title = @sprintf "\$ N_v=%d~~V_0=%.2f(x-1)^2  \$" Nv k0
        ax.set_title(title, fontsize=lbfz)

        if row_id == 5
        ax.set_xlabel("x", fontsize=lbfz)
        end
        if col_id == 1
        ax.set_ylabel("y", fontsize=lbfz)
        end
    end

    function plot_delta_appr_by_ax(ax, x0, xref, e_norm, interpo_xs, Np, Qx, rho_eq, Nv, lbfz, row_id, col_id)
        c_array = get_coefficients_delta_function(x0, xref, e_norm, interpo_xs, Np, Qx, rho_eq, Nv)
        p_appr_delta = rebuild_p0_by_Qx_carray(Nv, c_array, Qx, rho_eq)
        ax.plot(xref, p_appr_delta, color="blue")
        title = @sprintf "\$ N_v=%d~~\\delta(x- %.2f)  \$" Nv x0 
        ax.set_title(title, fontsize=lbfz)
        if row_id == 5
            ax.set_xlabel("x", fontsize=lbfz)
        end
        if col_id == 1
            ax.set_ylabel("y", fontsize=lbfz)
        end   
    end

    function plot_alpha_cycle(xref, prev_ahat, prev_ahat_edt, ahat, ylims, alpha_idx, observe_y, Qx)
        prev_idx = alpha_idx - 1
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(12,4))

        ylabel = @sprintf "\$ \\left< \\hat{\\alpha}_{t_{%d}} | x \\right> \$" prev_idx
        y_array = proj_eig_to_x(Qx, prev_ahat)
        plot_prev_alpha_hat(axes[1], xref, y_array, ylabel, ylims)

        ylabel = @sprintf "\$ \\left< \\hat{\\alpha}_{t_{%d}} | e^{-H \\Delta t} | x \\right> \$" prev_idx
        y_array = proj_eig_to_x(Qx, prev_ahat_edt)
        plot_prev_alpha_hat(axes[2], xref, y_array, ylabel, ylims)

        ylabel = @sprintf "\$ \\left< \\hat{\\alpha}_{t_{%d}} | x \\right> \$" alpha_idx
        y_array = proj_eig_to_x(Qx, ahat)
        plot_prev_alpha_hat(axes[3], xref, y_array, ylabel, ylims)

        redline_label = @sprintf "\$ y_{%d} = %.3f \$" alpha_idx observe_y
        axes[3].axvline(observe_y, color="red", label=redline_label)
        axes[3].legend()

        return fig, axes
    end

    function plot_prev_alpha_hat(ax, xref, ahat, ylabel, ylims)
        ax.plot(xref, ahat)
        ax.set_xlabel("x")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylims)
    end

    function proj_eig_to_x(Qx, vector)
        return Qx * vector
    end

end