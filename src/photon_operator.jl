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