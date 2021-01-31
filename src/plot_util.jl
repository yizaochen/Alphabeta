using PyPlot, Printf, PyCall

function plot_x_Qx_lambda(ax::PyCall.PyObject, idx_eigvector::Int64, LQ::Array{Float64,1}, Qx::Array{Float64,2}, xref::Array{Float64,2})
    lambda = LQ[idx_eigvector] 
    tau = 1 / lambda
    title = @sprintf "\$ \\lambda_{%d}=%.2E \$  \$ \\tau=%.2E \$" idx_eigvector lambda tau 
    xlabel = "x position"
    ylabel = @sprintf "\$ \\psi_{%d}(x) \$" idx_eigvector
    
    ax.plot(xref, Qx[:, idx_eigvector])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
end
    
function plot_alpha_t0(xref::Array{Float64,2}, V_eq::Array{Float64,2}, k_ref::Float64, peq::Array{Float64,2}, w0::Array{Float64,2}, rho_eq::Array{Float64,2}, alpha_t0_x::Array{Float64,2}, alpha_t0_norm::Float64, alpha_t0::Array{Float64,2})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
        
    ax = axes[1,1]
    ax = plot_potential(ax, xref, V_eq, k_ref)
    
    ax = axes[1,2]
    ax = plot_peq(ax, xref, peq, w0)
    
    ax = axes[2,1]
    ax = plot_rhoeq(ax, xref, rho_eq, alpha_t0_x, alpha_t0_norm)
    
    ax = axes[2,2]
    ax = barplot_alpha_t0(ax, alpha_t0)
    return fig, axes
end

function plot_beta_T(xref::Array{Float64,2}, beta_T::Array{Float64,2})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    photon_id = 4
    ax = axes[1]
    ax = plot_beta_T_x(ax, xref, photon_id)

    ax = axes[2]
    ax = barplot_beta_T(ax, beta_T)
    return fig, axes
end

function plot_beta_T_minus_1(y_beta::Array{Float64,2}, e_delta_t_y_beta::Array{Float64,2}, photon_id::Int64, xref::Array{Float64,2}, Qx::Array{Float64,2})
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    ax = axes[1]
    ax = barplot_y_beta(ax, y_beta, photon_id)

    ax = axes[2]
    ax = barplot_beta(ax, e_delta_t_y_beta, photon_id-1)

    ax = axes[3]
    ax = plot_beta_x(ax, xref, e_delta_t_y_beta, Qx, photon_id-1)
    return fig, axes
end

function plot_beta_hat_posterior(xref::Array{Float64,2}, beta_hat::Array{Float64,2}, posterior::Array{Float64,2}, photon_id::Int64, w0::Array{Float64,2}, y::Float64)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    ax = axes[1]
    ax = barplot_beta_hat(ax, beta_hat, photon_id)

    ax = axes[2]
    ax = plot_posterior(ax, xref, posterior, photon_id, w0, y)
    return fig, axes
end

function plot_alpha_t0_e_dt(xref::Array{Float64,2}, alpha_t0_e_delta_t::Array{Float64,2}, Qx::Array{Float64,2}, rho_eq::Array{Float64,2}, w0::Array{Float64,2})
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

    ax = axes[1]
    alpha_t0_e_delta_t_x = Qx * alpha_t0_e_delta_t
    ax = plot_alphat0_e_dt_rho_eq(ax, xref, alpha_t0_e_delta_t_x, rho_eq)
    
    ax = axes[2]
    p_delta_t = rho_eq .* alpha_t0_e_delta_t_x
    ax = plot_pdeltat_rhoeq_square(ax, xref, p_delta_t, rho_eq, w0)
    
    ax = axes[3]
    ax = barplot_alpha_t0_e_delta_t(ax, alpha_t0_e_delta_t)
    return fig, axes
end

function plot_photon_mat(xref::Array{Float64,2}, k_photon::Float64, photon_id::Int64, y_record::Array{Float64,2}, e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, psi_photon_psi::Array{Float64,2})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
     
    ax = axes[1]
    ax = plot_gaussian_filter(ax, y_record, photon_id, xref, e_norm, interpo_xs, Np, k_photon)
    
    ax = axes[2]
    ax = imshow_photon_mat(ax, psi_photon_psi, photon_id, fig)
    return fig, axes
end

function plot_alpha_hat(alpha_before_normalize::Array{Float64,2}, alpha_hat::Array{Float64,2}, photon_id::Int64, y_record::Array{Float64,2}, xref::Array{Float64,2}, Qx::Array{Float64,2}, w0::Array{Float64,2})
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,6))
    y = y_record[photon_id+1]
    
    ax = axes[1,1]
    ax = barplot_alpha_hat_before_normalize(ax, alpha_before_normalize, photon_id)
    
    ax = axes[1,2]
    ax = plot_alpha_x_before_normalize(ax, xref, alpha_before_normalize, Qx, y, photon_id)
    
    ax = axes[1,3]
    ax = plot_alpha_x_square_before_normalize(ax, xref, alpha_before_normalize, Qx, photon_id, w0)
    
    ax = axes[2,1]
    ax = barplot_alpha_hat(ax, alpha_hat, photon_id)
    
    ax = axes[2,2]
    ax = plot_alpha_hat_x(ax, xref, alpha_hat, Qx, y, photon_id)
    
    ax = axes[2,3]
    ax = plot_alpha_hat_x_square(ax, xref, alpha_hat, Qx, photon_id, w0)    
    return fig, axes
end

function plot_alpha_hat_e_dt(xref::Array{Float64,2}, alpha_hat::Array{Float64,2},  photon_id::Int64, alpha_hat_e_delta_t::Array{Float64,2}, delta_t::Float64, Qx::Array{Float64,2}, w0::Array{Float64,2})
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
     
    ax = axes[1,1]
    ax = barplot_alpha_hat(ax, alpha_hat, photon_id)
    
    ax = axes[1,2]
    ax = barplot_alpha_hat_edt(ax, alpha_hat_e_delta_t, photon_id, delta_t)

    ax = axes[2,1]
    ax = plot_alpha_hat_e_dt_x(ax, xref, alpha_hat, alpha_hat_e_delta_t, Qx, photon_id)    
    
    ax = axes[2,2]
    ax = plot_alpha_hat_e_delta_t_x_square(ax, xref, alpha_hat, alpha_hat_e_delta_t, Qx, photon_id, w0)
    return fig, axes
end

function barplot_alpha_hat_before_normalize(ax::PyCall.PyObject, alpha_before_normalize::Array{Float64,2}, photon_id::Int64)
    xarray = 1:72
    xticks = 1:5:72
    prev_photon_id = photon_id - 1
    ax.plot(xarray, alpha_before_normalize, "b.")
    ax.vlines(xarray, ymin=0, ymax=alpha_before_normalize)
    ax.set_xticks(xticks)
    ylabel = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t} \\mathbf{y}_{%d}  | \\psi_i \\rangle\$" prev_photon_id photon_id
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)")
    title = @sprintf "\$\\Vert \\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t} \\mathbf{y}_{%d}  |  \\Vert =%.3f \$" prev_photon_id photon_id norm(alpha_before_normalize)
    ax.set_title(title)
    return ax
end

function barplot_alpha_hat(ax::PyCall.PyObject, alpha_hat::Array{Float64,2}, photon_id::Int64)
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, alpha_hat, "b.")
    ax.vlines(xarray, ymin=0, ymax=alpha_hat)
    ax.set_xticks(xticks)
    ylabel = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | \\psi_i \\rangle\$" photon_id
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)")
    title = @sprintf "\$  \\Vert \\hat{\\alpha}_{t_{%d}}  \\Vert =%.3f \$" photon_id norm(alpha_hat)
    ax.set_title(title)
    return ax
end

function barplot_y_beta(ax::PyCall.PyObject, y_beta::Array{Float64,2}, photon_id::Int64)
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, y_beta, "b.")
    ax.vlines(xarray, ymin=0, ymax=y_beta)
    ax.set_xticks(xticks)
    ylabel = @sprintf "\$\\langle \\psi_i | \\mathbf{y}_{%d} |\\beta_{t_{%d}} \\rangle\$" photon_id photon_id
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)")
    title = @sprintf "\$ \\Vert | \\mathbf{y}_{%d} | \\beta_{t_{%d}} \\rangle \\Vert = %.3f \$" photon_id  photon_id norm(y_beta)
    ax.set_title(title)
    return ax
end

function barplot_beta(ax::PyCall.PyObject, beta::Array{Float64,2}, photon_id::Int64)
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, beta, "b.")
    ax.vlines(xarray, ymin=0, ymax=beta)
    ax.set_xticks(xticks)
    ylabel = @sprintf "\$\\langle \\psi_i |\\beta_{t_{%d}} \\rangle\$" photon_id 
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)")
    title = @sprintf "\$ \\Vert \\beta_{t_{%d}} \\Vert = %.3f \$" photon_id norm(beta)
    ax.set_title(title)
    return ax
end

function barplot_beta_hat(ax::PyCall.PyObject, beta_hat::Array{Float64,2}, photon_id::Int64)
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, beta_hat, "b.")
    ax.vlines(xarray, ymin=0, ymax=beta_hat)
    ax.set_xticks(xticks)
    ylabel = @sprintf "\$\\langle \\psi_i |\\hat{\\beta}_{t_{%d}} \\rangle\$" photon_id-1 
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)")
    title = @sprintf "\$ \\Vert \\hat{\\beta}_{t_{%d}} \\Vert = %.3f \$" photon_id-1 norm(beta_hat)
    ax.set_title(title)
    return ax
end

function barplot_alpha_hat_edt(ax::PyCall.PyObject, alpha_hat_e_delta_t::Array{Float64,2}, photon_id::Int64, delta_t::Float64)
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, alpha_hat_e_delta_t, "b.")
    ax.vlines(xarray, ymin=0, ymax=alpha_hat_e_delta_t)
    ax.set_xticks(xticks)
    ylabel = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t}| \\psi_i \\rangle\$" photon_id
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)")
    title = @sprintf "\$  \\Vert \\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t}|  \\Vert =%.3f \$  \$\\Delta t=%.0E\$s" photon_id norm(alpha_hat_e_delta_t) delta_t
    ax.set_title(title)
    return ax
end

function plot_beta_x(ax::PyCall.PyObject, xref::Array{Float64,2}, beta::Array{Float64,2}, Qx::Array{Float64,2}, photon_id::Int64)
    beta_x = Qx * beta
    ax.plot(xref, beta_x)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$\\langle x_{%d} | \\beta_{t_{%d}} \\rangle\$" photon_id photon_id
    ax.set_ylabel(ylabel)
    return ax
end

function plot_posterior(ax::PyCall.PyObject, xref::Array{Float64,2}, posterior::Array{Float64,2}, photon_id::Int64, w0::Array{Float64,2}, y::Float64)
    ax.plot(xref, posterior)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" photon_id-1
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$p(x_{%d}|\\mathbf{y})\$" photon_id-1
    ax.set_ylabel(ylabel)
    label = @sprintf "\$y_{%d}=%.3f\$" photon_id-1 y
    ax.axvline(y, color="red", alpha=0.3, label=label)
    title = @sprintf "\$\\int p(x_{%d}|\\mathbf{y}) dx_{%d} = %.3f\$" photon_id-1 photon_id-1 sum(w0 .* posterior)
    ax.set_title(title)
    ax.legend()
    return ax
end

function plot_alpha_x_before_normalize(ax::PyCall.PyObject, xref::Array{Float64,2}, alpha_before_normalize::Array{Float64,2},  Qx::Array{Float64,2}, y::Float64, photon_id::Int64)
    prev_photon_id = photon_id - 1
    alpha_before_normalize_x = Qx * alpha_before_normalize
    ax.plot(xref, alpha_before_normalize_x)
    ax.axvline(y, color="red", alpha=0.3)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t} \\mathbf{y}_{%d}  | x_{%d} \\rangle\$" prev_photon_id photon_id photon_id
    ax.set_ylabel(ylabel)
    return ax
end

function plot_alpha_hat_x(ax::PyCall.PyObject, xref::Array{Float64,2}, alpha_hat::Array{Float64,2}, Qx::Array{Float64,2}, y::Float64, photon_id::Int64)
    alpha_hat_x = Qx * alpha_hat
    ax.plot(xref, alpha_hat_x)
    ax.axvline(y, color="red", alpha=0.3)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | x_{%d} \\rangle\$" photon_id photon_id
    ax.set_ylabel(ylabel)
    return ax
end

function plot_alpha_hat_e_dt_x(ax::PyCall.PyObject, xref::Array{Float64,2}, alpha_hat::Array{Float64,2}, alpha_hat_e_delta_t::Array{Float64,2}, Qx::Array{Float64,2}, photon_id::Int64)
    next_photon_id = photon_id + 1
    alpha_hat_x = Qx * alpha_hat
    alpha_hat_e_delta_t_x = Qx * alpha_hat_e_delta_t
    label = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | x_{%d} \\rangle \$" photon_id photon_id
    ax.plot(xref, alpha_hat_x, color="blue", alpha=0.4, label=label)
    ax.plot(xref, alpha_hat_e_delta_t_x, color="red")
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" next_photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t}| x_{%d} \\rangle\$" photon_id next_photon_id
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax
end

function plot_beta_T_x(ax::PyCall.PyObject, xref::Array{Float64,2}, photon_id::Int64)
    ax.plot(xref, ones(length(xref)))
    xlabel = @sprintf "\$ x \$ (\$ \\AA \$)"
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$\\langle  x | \\beta_{t_{%d}} \\rangle\$" photon_id
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 2)
    return ax
end

function plot_alpha_x_square_before_normalize(ax::PyCall.PyObject, xref::Array{Float64,2}, alpha_before_normalize::Array{Float64,2},  Qx::Array{Float64,2}, photon_id::Int64, w0::Array{Float64,2})
    prev_photon_id = photon_id - 1
    alpha_before_normalize_x = Qx * alpha_before_normalize
    alpha_before_normalize_x_square = alpha_before_normalize_x .^ 2
    sum_alpha_before_normalize_x_square = sum(w0 .* alpha_before_normalize_x_square)
    ax.plot(xref, alpha_before_normalize_x_square)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$(\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t} \\mathbf{y}_{%d}  | x_{%d} \\rangle)^2\$" prev_photon_id photon_id photon_id
    ax.set_ylabel(ylabel)
    title = @sprintf "\$ \\sum_{k=1}^{193} w(x_k) (\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t} \\mathbf{y}_{%d}  | x_k \\rangle)^2 = %.3f \$" prev_photon_id photon_id sum_alpha_before_normalize_x_square
    ax.set_title(title)
    return ax
end

function plot_alpha_hat_x_square(ax::PyCall.PyObject, xref::Array{Float64,2}, alpha_hat::Array{Float64,2},  Qx::Array{Float64,2}, photon_id::Int64, w0::Array{Float64,2})
    alpha_hat_x = Qx * alpha_hat
    alpha_hat_x_square = alpha_hat_x .^ 2
    sum_alpha_hat_x_square = sum(w0 .* alpha_hat_x_square)
    ax.plot(xref, alpha_hat_x_square)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$(\\langle \\hat{\\alpha}_{t_{%d}} | x_{%d} \\rangle)^2 \$" photon_id photon_id
    ax.set_ylabel(ylabel)
    title = @sprintf "\$ \\sum_{k=1}^{193} w(x_k) (\\langle \\hat{\\alpha}_{t_{%d}} | x_k \\rangle)^2 = %.3f \$" photon_id sum_alpha_hat_x_square
    ax.set_title(title)
    return ax
end

function plot_alpha_hat_e_delta_t_x_square(ax::PyCall.PyObject, xref::Array{Float64,2}, alpha_hat::Array{Float64,2},  alpha_hat_e_delta_t::Array{Float64,2}, Qx::Array{Float64,2}, photon_id::Int64, w0::Array{Float64,2})
    next_photon_id = photon_id + 1
    alpha_hat_x = Qx * alpha_hat
    alpha_hat_x_square = alpha_hat_x .^ 2
    alpha_hat_e_delta_t_x = Qx * alpha_hat_e_delta_t
    alpha_hat_e_delta_t_x_square = alpha_hat_e_delta_t_x .^ 2
    sum_alpha_hat_e_delta_t_x_square = sum(w0 .* alpha_hat_e_delta_t_x_square)
    label = @sprintf "\$(\\langle \\hat{\\alpha}_{t_{%d}} | x \\rangle)^2\$" photon_id
    ax.plot(xref, alpha_hat_x_square, color="blue", alpha=0.4, label=label)
    ax.plot(xref, alpha_hat_e_delta_t_x_square, color="red")
    ax.axvline(50, color="grey", alpha=0.2)
    xlabel = @sprintf "\$ x_{%d} \$ (\$ \\AA \$)" next_photon_id
    ax.set_xlabel(xlabel)
    ylabel = @sprintf "\$(\\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t}| x_{%d} \\rangle)^2\$" photon_id next_photon_id
    ax.set_ylabel(ylabel)
    ax.legend()
    title = @sprintf "\$ \\sum_{k=1}^{193} w(x_k) \\langle \\hat{\\alpha}_{t_{%d}} | e^{-\\mathbf{H} \\Delta t}| x_k \\rangle)^2 = %.3f \$" photon_id  sum_alpha_hat_e_delta_t_x_square
    ax.set_title(title)
    return ax
end

function plot_gaussian_filter(ax::PyCall.PyObject, y_record::Array{Float64,2}, photon_id::Int64, xref::Array{Float64,2},  e_norm::Float64, interpo_xs::Array{Float64,1}, Np::Int64, k_photon::Float64)
    y = y_record[photon_id+1]
    y_idx = find_nearest_point(y, xref, e_norm, interpo_xs, Np)
    original_gaussian = get_gaussian(k_photon, xref, y_idx)
    sigma_photon = 1 / sqrt(2 * k_photon)

    ax.plot(xref, original_gaussian)
    label = @sprintf "\$ \\mu_{%d}=%.3f\$" photon_id y
    ax.axvline(y, color="red", label=label)
    ax.set_xlabel("x (\$ \\AA \$)")
    ylabel = @sprintf "Gaussian Filter \$ \\hat{\\mathbf{y}}_{%d} = f(x) \$" photon_id
    ax.set_ylabel(ylabel, fontsize=12)
    title = @sprintf "\$k=%.1f\$ kcal/mol/\$ \\AA^2\$ \$ \\sigma=%.3f \\AA \$" k_photon sigma_photon
    ax.set_title(title)
    ax.legend()
    return ax
end

function imshow_photon_mat(ax::PyCall.PyObject, psi_photon_psi::Array{Float64,2}, photon_id::Int64, fig::Figure)
    im = ax.imshow(psi_photon_psi, cmap="hot")
    title = @sprintf "\$ \\mathbf{y}_{%d}  = \\langle \\psi_i(x) | \\hat{\\mathbf{y}}_{%d} | \\psi_j(x) \\rangle  \\approx \\sum_{k=1}^{193} w(x_k) \\psi_i(x_k) f(x_k) \\psi_j(x_k) \$" photon_id photon_id
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_xticklabels(["1", "11", "21", "31", "41", "51", "61", "71"])
    ax.set_yticklabels(["1", "11", "21", "31", "41", "51", "61", "71"])
    ax.set_xlabel("\$ i \$")
    ax.set_ylabel("\$ j \$")
    return ax
end

function plot_alphat0_e_dt_rho_eq(ax::PyCall.PyObject, x::Array{Float64,2}, alpha_t0_e_delta_t_x::Array{Float64,2}, rho_eq::Array{Float64,2})
    ax.plot(x, alpha_t0_e_delta_t_x, lw=2, label="\$ \\langle \\alpha_{t_0} | e^{-\\mathbf{H} \\Delta t}| x \\rangle \$")
    ax.plot(x, rho_eq, "--", color="red", label="\$ \\rho_{eq}(x) \$")
    ax.set_xlabel("x (\$ \\AA \$)")
    ax.set_ylabel("\$\\langle \\alpha_{t_0} | e^{-\\mathbf{H} \\Delta t}| x \\rangle = \\rho(x, \\Delta t) \$")
    ax.legend()
    return ax
end

function plot_pdeltat_rhoeq_square(ax::PyCall.PyObject, x::Array{Float64,2}, p_delta_t::Array{Float64,2}, rho_eq::Array{Float64,2}, w0::Array{Float64,2})
    ax.plot(x, p_delta_t, lw=2, label="\$ p(x, \\Delta t) \$")
    ax.plot(x, rho_eq .* rho_eq, "--", color="red", label="\$ p_{eq}(x) \$")
    ax.set_xlabel("x (\$ \\AA \$)")
    ax.set_ylabel("\$p(x, \\Delta t) \$")
    title = @sprintf "\$ \\sum_{i=1}^{193}w_i p(x_i)dx=%.3f \$" sum(w0 .* p_delta_t)
    ax.set_title(title)
    ax.legend()
    return ax
end

function barplot_alpha_t0_e_delta_t(ax::PyCall.PyObject, alpha_t0_e_delta_t::Array{Float64,2})
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, alpha_t0_e_delta_t, "b.")
    ax.vlines(xarray, ymin=0, ymax=alpha_t0_e_delta_t)
    ax.set_xticks(xticks)
    ax.set_ylabel("\$\\langle \\alpha_{t_0} | e^{-\\mathbf{H} \\Delta t} | \\psi_i \\rangle\$", fontsize=12)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)", fontsize=12)
    alpha_t0_e_delta_t_norm = norm(alpha_t0_e_delta_t)
    title = @sprintf "\$  \\Vert\\langle \\alpha_{t_0} | e^{-\\mathbf{H} \\Delta t} | \\Vert =%.3f \$" alpha_t0_e_delta_t_norm
    ax.set_title(title)
    return ax
end

function plot_potential(ax::PyCall.PyObject, x::Array{Float64,2}, V::Array{Float64,2}, k_ref::Float64)
    ax.plot(x, V)
    ax.set_xlabel("x (\$ \\AA \$)")
    ax.set_ylabel("\$ V_{ref}(x) \$")
    title = @sprintf "k=%.3f kcal/mol/\$\\AA^2\$" k_ref 
    ax.set_title(title)
    return ax
end

function plot_peq(ax::PyCall.PyObject, x::Array{Float64,2}, peq::Array{Float64,2}, w0::Array{Float64,2})
    ax.plot(x, peq)
    ax.set_xlabel("x (\$ \\AA \$)")
    ax.set_ylabel("\$ p_{eq}(x) \$")
    title = @sprintf "\$ \\sum_{i=1}^{193}w_i p(x_i)dx=%.3f \$" sum(w0 .* peq)
    ax.set_title(title)
    ax.set_ylim(-0.05, 0.8)
    return ax
end

function plot_rhoeq(ax::PyCall.PyObject, x::Array{Float64,2}, rho_eq::Array{Float64,2}, alpha_t0_x::Array{Float64,2}, alpha_t0_norm::Float64)
    ax.plot(x, rho_eq, lw=2, label="\$ \\rho_{eq}(x)  \$")
    ax.plot(x, alpha_t0_x, "--", color="red", label="\$ \\langle \\alpha_{t_0} | x \\rangle \$")
    ax.set_xlabel("x (\$ \\AA \$)")
    ax.set_ylabel("\$\\langle \\alpha_{t_0} | x \\rangle = \\rho_{eq}(x) \$")
    ax.set_ylim(-0.05, 0.8)
    title = @sprintf "\$  \\Vert \\alpha_{t_0} \\Vert =%.3f \$" alpha_t0_norm
    ax.set_title(title)
    ax.legend()
    return ax
end

function barplot_alpha_t0(ax::PyCall.PyObject, alpha_t0::Array{Float64,2})
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, alpha_t0, "b.")
    ax.vlines(xarray, ymin=0, ymax=alpha_t0)
    ax.set_xticks(xticks)
    ax.set_ylabel("\$\\langle \\alpha_{t_0} | \\psi_i \\rangle\$", fontsize=12)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)", fontsize=12)
    plt.tight_layout()
    return ax
end

function barplot_beta_T(ax::PyCall.PyObject, beta_T::Array{Float64,2})
    xarray = 1:72
    xticks = 1:5:72
    ax.plot(xarray, beta_T, "b.")
    ax.vlines(xarray, ymin=0, ymax=beta_T)
    ax.set_xticks(xticks)
    ax.set_ylabel("\$\\langle \\psi_i | \\beta_{t_4} \\rangle\$", fontsize=12)
    ax.set_xlabel("\$ i \$, the index of eigenvector(\$ \\psi_i\$)", fontsize=12)
    title = @sprintf "\$  \\Vert \\beta_{t_4} \\Vert =%.3f \$" norm(beta_T)
    ax.set_title(title)
    return ax
end