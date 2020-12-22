using Dierckx

function linear_interpolate(x_i::Float64, x_j::Float64, y_i::Float64, y_j::Float64, x::Float64)
    m = (y_j - y_i) / (x_j - x_i) # Calculate m
    c = y_i - (m * x_i) # Calculate c
    return m * x + c
end

function smooth_psi(N::Int64, psi_sele::Array{Float64,1}, abrupt_boolean::Bool, idx_larger_than_1::Array{Int64,1}, xref::Array{Float64,1})
    psi_fix = zeros(N)
    psi_fix[:] = psi_sele[:]
    
    if abrupt_boolean
        boundary_left_idx = minimum(idx_larger_than_1) - 1
        boundary_right_idx = maximum(idx_larger_than_1) + 1
        
        x_i = xref[boundary_left_idx]
        x_j = xref[boundary_right_idx]
        y_i = psi_sele[boundary_left_idx]
        y_j = psi_sele[boundary_right_idx]
        
        # Fix value by linear interpolation
        for idx_between=boundary_left_idx:boundary_right_idx
            x = xref[idx_between]
            psi_fix[idx_between] = linear_interpolate(x_i, x_j, y_i, y_j, x)
        end
    end
    return psi_fix
end

function smooth_peq(N::Int64, p_old::Array{Float64,1}, abrupt_boolean::Bool, w0::Array{Float64,2}, xref::Array{Float64,2})
    p_fix = zeros(N,1)
    p_fix[:,1] = p_old
    if abrupt_boolean
        smoothing_factor = 0.1
        spl = Spline1D(xref[:,1], p_old; w=ones(length(xref[:,1])), k=3, bc="nearest", s=smoothing_factor)
        temp = spl(xref[:,1])
        p_smooth = sqrt.(temp .^ 2)
        p_smooth = p_smooth / sum(w0 .* p_smooth)
        p_smooth = max.(p_smooth, 1e-10)
        p_fix[:,1] = p_smooth;
    end
    return p_fix
end
