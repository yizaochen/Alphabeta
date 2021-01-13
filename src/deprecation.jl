function get_power_initial_guess_D(Nh::Int64, Np::Int64, xratio::Int64, xavg::Int64, peq::Array{Float64,2}, Nv::Int64, tau::Int64, x_record::Array{Float64,2}, dt::Float64)
    power_array = -2:11
    n_power = length(power_array)
    l_container = zeros(n_power,1)
    D_array = zeros(n_power)
    idx = 1
    for power in power_array
        D_test = 10^(float(power))
        D_array[idx] = D_test   
        l_container[idx] = get_loglikelihood(Nh, Np, xratio, xavg, peq, D_test, Nv, tau, x_record, dt)
        idx += 1
    end
    min_idx = argmin(l_container[:])
    min_idx_left = min_idx - 1
    min_idx_right = min_idx + 1

    if (min_idx == 1) | (min_idx == 14)
        return D_array[min_idx], (D_array, l_container)
    end

    if l_container[min_idx_right] > l_container[min_idx_left]
        return D_array[min_idx_right], (D_array, l_container)
    else
        return D_array[min_idx_left], (D_array, l_container)
    end
end