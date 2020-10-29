function get_y_revised(y::Array{Float64,1}, N::Int64)
    # Add additional two more points on the tail
    y_revised = zeros(N+2)
    y_revised[1:N] = y[:]
    y_revised[N+1] = y[end]
    y_revised[end] = y[end]
    return y_revised
end

function get_x_revised(x::Array{Float64,1}, N::Int64, e_norm::Float64)
    # Add additional two more points on the tail
    x_revised = zeros(N+2)
    x_revised[1:N] = x[:]
    x_revised[N+1] = x[end] + e_norm
    x_revised[end] = x[end] + 2 * e_norm
    return x_revised
end

function first_derivative(x::Array{Float64,1}, y::Array{Float64,1}, N::Int64, e_norm::Float64)
    y_revised = get_y_revised(y, N)
    x_revised = get_x_revised(x, N, e_norm)
    first_derivative = diff(y_revised) ./ diff(x_revised)
    return first_derivative # return array of y'(x)
end

function second_derivative(x::Array{Float64,1}, y::Array{Float64,1}, N::Int64, e_norm::Float64)
    first_deriv = first_derivative(x, y, N, e_norm)
    x_revised = get_x_revised(x, N, e_norm)
    second_derivative = diff(first_deriv) ./ diff(x_revised[1:N+1])
    return second_derivative # return array of y''(x)
end

function get_loc_number_of_abrupt(y::Array{Float64,1})
    threshold = 1.
    y = abs.(y)
    idx_larger_than_1 = findall(x -> x>=threshold, y)
    return length(idx_larger_than_1), idx_larger_than_1
end

function detect_abrupt(x::Array{Float64,1}, y::Array{Float64,1}, N::Int64, e_norm::Float64)
    second_deriv = second_derivative(x, y, N, e_norm)
    n_larger_than_1, idx_larger_than_1 = get_loc_number_of_abrupt(second_deriv)
    if n_larger_than_1 > 0
        return true, idx_larger_than_1
    else
        return false, idx_larger_than_1
    end
end