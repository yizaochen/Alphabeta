# Model evaluation
using LinearAlgebra

get_diff_norm(p_em::Array{Float64,1}, pref::Array{Float64,2}) = norm(p_em - pref)

function iteration_evaluation(n_iteration::Int64, p_container::Array{Float64,2}, peq::Array{Float64,2})
    x_array = 0:n_iteration
    y_array = zeros(Float64, n_iteration+1)
    y_array[1] = get_diff_norm(p_container[1,:], peq)
    for iter_id = 1:n_iteration
        y_array[iter_id+1] = get_diff_norm(p_container[iter_id+1,:], peq)
    end
    return x_array, y_array
end