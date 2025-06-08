
# include("/nfs/home2/nacevedo/RA/cuPDLP.jl/scripts/clean_iterative_refinement.jl")

struct SaddlePointOutput
    """
    The output primal solution vector.
    """
    primal_solution::Union{Vector{Float64}, CuVector{Float64}}

    """
    The output dual solution vector.
    """
    dual_solution::Union{Vector{Float64}, CuVector{Float64}}

    """
    One of the possible values from the TerminationReason enum.
    """
    termination_reason::TerminationReason

    """
    Extra information about the termination reason (may be empty).
    """
    termination_string::String

    """
    The total number of algorithmic iterations for the solve.
    """
    iteration_count::Int32

    """
    Detailed statistics about a subset of the iterations. The collection frequency
    is defined by algorithm parameters.
    """
    iteration_stats::Vector{IterationStats}

    # MINE : save primal-dual iterates
    primal_dual_iterates::Any
end

"""
Return the unscaled primal and dual solutions
"""
function unscaled_saddle_point_output(
    scaled_problem::Union{ScaledQpProblem, CuScaledQpProblem},
    primal_solution::AbstractVector{Float64},
    dual_solution::AbstractVector{Float64},
    termination_reason::TerminationReason,
    iterations_completed::Int64,
    iteration_stats::Vector{IterationStats},
    primal_dual_iterates::Any,
)
    # Unscale iterates.
    original_primal_solution =
        primal_solution ./ scaled_problem.variable_rescaling
    original_dual_solution = dual_solution ./ scaled_problem.constraint_rescaling
  
    return SaddlePointOutput(
        original_primal_solution,
        original_dual_solution,
        termination_reason,
        termination_reason_to_string(termination_reason),
        iterations_completed,
        iteration_stats,
        primal_dual_iterates,
    )
end

function weighted_norm(
    vec::CuVector{Float64},
    weights::Float64,
)
    tmp = CUDA.norm(vec)
    return sqrt(weights) * tmp
end

mutable struct CuSolutionWeightedAverage
    sum_primal_solutions::CuVector{Float64}
    sum_dual_solutions::CuVector{Float64}
    sum_primal_solutions_count::Int64
    sum_dual_solutions_count::Int64
    sum_primal_solution_weights::Float64
    sum_dual_solution_weights::Float64
    sum_primal_product::CuVector{Float64}
    sum_dual_product::CuVector{Float64}
end

mutable struct CuBufferAvgState
    avg_primal_solution::CuVector{Float64}
    avg_dual_solution::CuVector{Float64}
    avg_primal_product::CuVector{Float64}
    avg_primal_gradient::CuVector{Float64}
end

"""
Initialize weighted average
"""
function initialize_solution_weighted_average(
    primal_size::Int64,
    dual_size::Int64,
)
    return CuSolutionWeightedAverage(
        CUDA.zeros(Float64, primal_size),
        CUDA.zeros(Float64, dual_size),
        0,
        0,
        0.0,
        0.0,
        CUDA.zeros(Float64, dual_size),
        CUDA.zeros(Float64, primal_size),
    )
end

"""
Reset weighted average
"""
function reset_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
)
    solution_weighted_avg.sum_primal_solutions .=
        CUDA.zeros(Float64, length(solution_weighted_avg.sum_primal_solutions))
    solution_weighted_avg.sum_dual_solutions .=
        CUDA.zeros(Float64, length(solution_weighted_avg.sum_dual_solutions))
    solution_weighted_avg.sum_primal_solutions_count = 0
    solution_weighted_avg.sum_dual_solutions_count = 0
    solution_weighted_avg.sum_primal_solution_weights = 0.0
    solution_weighted_avg.sum_dual_solution_weights = 0.0

    solution_weighted_avg.sum_primal_product .= CUDA.zeros(Float64, length(solution_weighted_avg.sum_dual_solutions))
    solution_weighted_avg.sum_dual_product .= CUDA.zeros(Float64, length(solution_weighted_avg.sum_primal_solutions))
    return
end

"""
Update weighted average of primal solution
"""
function add_to_primal_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_primal_solutions_count >= 0
    solution_weighted_avg.sum_primal_solutions .+=
        current_primal_solution * weight
    solution_weighted_avg.sum_primal_solutions_count += 1
    solution_weighted_avg.sum_primal_solution_weights += weight
    return
end

"""
Update weighted average of dual solution
"""
function add_to_dual_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_solution::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_dual_solutions_count >= 0
    solution_weighted_avg.sum_dual_solutions .+= current_dual_solution * weight
    solution_weighted_avg.sum_dual_solutions_count += 1
    solution_weighted_avg.sum_dual_solution_weights += weight
    return
end

"""
Update weighted average of primal product
"""
function add_to_primal_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_primal_solutions_count >= 0
    solution_weighted_avg.sum_primal_product .+=
        current_primal_product * weight
    return
end

"""
Update weighted average of dual product
"""
function add_to_dual_product_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_dual_product::CuVector{Float64},
    weight::Float64,
)
    @assert solution_weighted_avg.sum_dual_solutions_count >= 0
    solution_weighted_avg.sum_dual_product .+=
        current_dual_product * weight
    return
end


"""
Update weighted average
"""
function add_to_solution_weighted_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    weight::Float64,
    current_primal_product::CuVector{Float64},
    current_dual_product::CuVector{Float64},
)
    add_to_primal_solution_weighted_average!(
        solution_weighted_avg,
        current_primal_solution,
        weight,
    )
    add_to_dual_solution_weighted_average!(
        solution_weighted_avg,
        current_dual_solution,
        weight,
    )

    add_to_primal_product_weighted_average!(
        solution_weighted_avg,
        current_primal_product,
        weight,
    )
    add_to_dual_product_weighted_average!(
        solution_weighted_avg,
        current_dual_product,
        weight,
    )
    return
end

"""
Compute average solutions
"""
function compute_average!(
    solution_weighted_avg::CuSolutionWeightedAverage,
    buffer_avg::CuBufferAvgState,
    problem::CuLinearProgrammingProblem,
)
    buffer_avg.avg_primal_solution .= solution_weighted_avg.sum_primal_solutions ./ solution_weighted_avg.sum_primal_solution_weights

    buffer_avg.avg_dual_solution .= solution_weighted_avg.sum_dual_solutions ./ solution_weighted_avg.sum_dual_solution_weights

    buffer_avg.avg_primal_product .= solution_weighted_avg.sum_primal_product ./ solution_weighted_avg.sum_primal_solution_weights

    buffer_avg.avg_primal_gradient .= -solution_weighted_avg.sum_dual_product ./ solution_weighted_avg.sum_dual_solution_weights
    buffer_avg.avg_primal_gradient .+= problem.objective_vector

end


mutable struct CuKKTrestart
    kkt_residual::Float64
end

"""
Compute weighted KKT residual for restarting
"""
function compute_weight_kkt_residual(
    problem::CuLinearProgrammingProblem,
    primal_iterate::CuVector{Float64},
    dual_iterate::CuVector{Float64},
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
    buffer_kkt::BufferKKTState,
    primal_weight::Float64,
    primal_norm_params::Float64, 
    dual_norm_params::Float64, 
)
    ## construct buffer_kkt
    buffer_kkt.primal_solution = primal_iterate
    buffer_kkt.dual_solution = dual_iterate
    buffer_kkt.primal_product = primal_product
    buffer_kkt.primal_gradient = primal_gradient

    compute_primal_residual!(problem, buffer_kkt)
    primal_objective = primal_obj(problem, buffer_kkt.primal_solution)
    l2_primal_residual = CUDA.norm([buffer_kkt.constraint_violation; buffer_kkt.lower_variable_violation; buffer_kkt.upper_variable_violation], 2)

    compute_dual_stats!(problem, buffer_kkt)
    dual_objective = buffer_kkt.dual_stats.dual_objective
    l2_dual_residual = CUDA.norm([buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation], 2)

    weighted_kkt_residual = sqrt(primal_weight * l2_primal_residual^2 + 1/primal_weight * l2_dual_residual^2 + abs(primal_objective - dual_objective)^2)

    return CuKKTrestart(weighted_kkt_residual)
end

mutable struct CuRestartInfo
    """
    The primal_solution recorded at the last restart point.
    """
    primal_solution::CuVector{Float64}
    """
    The dual_solution recorded at the last restart point.
    """
    dual_solution::CuVector{Float64}
    """
    KKT residual at last restart. This has a value of nothing if no restart has occurred.
    """
    last_restart_kkt_residual::Union{Nothing,CuKKTrestart} 
    """
    The length of the last restart interval.
    """
    last_restart_length::Int64
    """
    The primal distance moved from the restart point two restarts ago and the average of the iterates across the last restart.
    """
    primal_distance_moved_last_restart_period::Float64
    """
    The dual distance moved from the restart point two restarts ago and the average of the iterates across the last restart.
    """
    dual_distance_moved_last_restart_period::Float64
    """
    Reduction in the potential function that was achieved last time we tried to do a restart.
    """
    kkt_reduction_ratio_last_trial::Float64

    primal_product::CuVector{Float64}
    primal_gradient::CuVector{Float64}

    # MINE: number of IR applied
    ir_refinement_applied::Bool
    ir_refinements_number::Int64 
end

"""
Initialize last restart info
"""
function create_last_restart_info(
    problem::CuLinearProgrammingProblem,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
)
    return CuRestartInfo(
        copy(primal_solution),
        copy(dual_solution),
        nothing,
        1,
        0.0,
        0.0,
        1.0,
        copy(primal_product),
        copy(primal_gradient),
        false, # MINE: Applied IR
        0, # MINE: Number of IR applied
    )
end

"""
RestartScheme enum
-  `NO_RESTARTS`: No restarts are performed.
-  `FIXED_FREQUENCY`: does a restart every [restart_frequency] iterations where [restart_frequency] is a user-specified number.
-  `ADAPTIVE_KKT`: a heuristic based on the KKT residual to decide when to restart. 
"""
@enum RestartScheme NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT

"""
RestartToCurrentMetric enum
- `NO_RESTART_TO_CURRENT`: Always reset to the average.
- `KKT_GREEDY`: Decide between the average current based on which has a smaller KKT.
"""
@enum RestartToCurrentMetric NO_RESTART_TO_CURRENT KKT_GREEDY


mutable struct RestartParameters
    """
    Specifies what type of restart scheme is used.
    """
    restart_scheme::RestartScheme
    """
    Specifies how we decide between restarting to the average or current.
    """
    restart_to_current_metric::RestartToCurrentMetric
    """
    If `restart_scheme` = `FIXED_FREQUENCY` then this number determines the frequency that the algorithm is restarted.
    """
    restart_frequency_if_fixed::Int64
    """
    If in the past `artificial_restart_threshold` fraction of iterations no restart has occurred then a restart will be artificially triggered. The value should be between zero and one. Smaller values will have more frequent artificial restarts than larger values.
    """
    artificial_restart_threshold::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold improvement in the quality of the current/average iterate compared with that  of the last restart that will trigger a restart. The value of this parameter should be between zero and one. Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    sufficient_reduction_for_restart::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold
    improvement in the quality of the current/average iterate compared with that of the last restart that is neccessary for a restart to be triggered. If this thrshold is met and the quality of the iterates appear to be getting worse then a restart is triggered. The value of this parameter should be between zero and one, and greater than sufficient_reduction_for_restart. Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    necessary_reduction_for_restart::Float64
    """
    Controls the exponential smoothing of log(primal_weight) when the primal weight is updated (i.e., on every restart). Must be between 0.0 and 1.0 inclusive. At 0.0 the primal weight remains frozen at its initial value.
    """
    primal_weight_update_smoothing::Float64
end

"""
Construct restart parameters
"""
function construct_restart_parameters(
    restart_scheme::RestartScheme,
    restart_to_current_metric::RestartToCurrentMetric,
    restart_frequency_if_fixed::Int64,
    artificial_restart_threshold::Float64,
    sufficient_reduction_for_restart::Float64,
    necessary_reduction_for_restart::Float64,
    primal_weight_update_smoothing::Float64,
)
    @assert restart_frequency_if_fixed > 1
    @assert 0.0 < artificial_restart_threshold <= 1.0
    @assert 0.0 <
            sufficient_reduction_for_restart <=
            necessary_reduction_for_restart <=
            1.0
    @assert 0.0 <= primal_weight_update_smoothing <= 1.0
  
    return RestartParameters(
        restart_scheme,
        restart_to_current_metric,
        restart_frequency_if_fixed,
        artificial_restart_threshold,
        sufficient_reduction_for_restart,
        necessary_reduction_for_restart,
        primal_weight_update_smoothing,
    )
end

"""
Check if restart at average solutions
"""
function should_reset_to_average(
    current::CuKKTrestart,
    average::CuKKTrestart,
    restart_to_current_metric::RestartToCurrentMetric,
)
    if restart_to_current_metric == KKT_GREEDY
        return current.kkt_residual  >=  average.kkt_residual
    else
        return true # reset to average
    end
end

"""
Check restart criteria based on weighted KKT
"""
function should_do_adaptive_restart_kkt(
    problem::CuLinearProgrammingProblem,
    candidate_kkt::CuKKTrestart, 
    restart_params::RestartParameters,
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    buffer_kkt::BufferKKTState,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
)
    
    last_restart = compute_weight_kkt_residual(
        problem,
        last_restart_info.primal_solution,
        last_restart_info.dual_solution,
        last_restart_info.primal_product,
        last_restart_info.primal_gradient,
        buffer_kkt,
        primal_weight,
        primal_norm_params,
        dual_norm_params,
    )

    do_restart = false

    kkt_candidate_residual = candidate_kkt.kkt_residual
    kkt_last_residual = last_restart.kkt_residual       
    kkt_reduction_ratio = kkt_candidate_residual / kkt_last_residual

    if kkt_reduction_ratio < restart_params.necessary_reduction_for_restart
        if kkt_reduction_ratio < restart_params.sufficient_reduction_for_restart
            do_restart = true
        elseif kkt_reduction_ratio > last_restart_info.kkt_reduction_ratio_last_trial
            do_restart = true
        end
    end
    last_restart_info.kkt_reduction_ratio_last_trial = kkt_reduction_ratio
  
    return do_restart
end


"""
Check restart
"""
function run_restart_scheme(
    problem::CuLinearProgrammingProblem,
    solution_weighted_avg::CuSolutionWeightedAverage,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    last_restart_info::CuRestartInfo,
    iterations_completed::Int64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_weight::Float64,
    verbosity::Int64,
    restart_params::RestartParameters,
    primal_product::CuVector{Float64},
    dual_product::CuVector{Float64},
    buffer_avg::CuBufferAvgState,
    buffer_kkt::BufferKKTState,
    buffer_primal_gradient::CuVector{Float64},
    # MINE
    current_iteration_stats::Any,
    primal_dual_iterates::Any,
    qp_cache::Any,
    pdhg_iteration::Int,
    termination_criteria::Any=nothing, #TerminationCriteria,
    ir_over_restart::Bool=false,
    ir_type::String="scalar",
    ir_iteration_threshold::Int64=1000,
    iteration_limit::Int32=typemax(Int32),
    ir_tolerance_factor::Float64=1e-3,
    # last_restart_iteration::Int,
)
    if solution_weighted_avg.sum_primal_solutions_count > 0 &&
        solution_weighted_avg.sum_dual_solutions_count > 0
        # compute_average!(solution_weighted_avg, buffer_avg, problem)
    else
        return RESTART_CHOICE_NO_RESTART
    end

    restart_length = solution_weighted_avg.sum_primal_solutions_count
    artificial_restart = false
    do_restart = false
    
    if restart_length >= restart_params.artificial_restart_threshold * iterations_completed
        do_restart = true
        artificial_restart = true
    end

    if restart_params.restart_scheme == NO_RESTARTS
        reset_to_average = false
        candidate_kkt_residual = nothing
    else
        current_kkt_res = compute_weight_kkt_residual(
            problem,
            current_primal_solution,
            current_dual_solution,
            primal_product,
            buffer_primal_gradient,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )
        avg_kkt_res = compute_weight_kkt_residual(
            problem,
            buffer_avg.avg_primal_solution,
            buffer_avg.avg_dual_solution,
            buffer_avg.avg_primal_product,
            buffer_avg.avg_primal_gradient,
            buffer_kkt,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
        )

        reset_to_average = should_reset_to_average(
            current_kkt_res,
            avg_kkt_res,
            restart_params.restart_to_current_metric,
        )

        if reset_to_average
            candidate_kkt_residual = avg_kkt_res
        else
            candidate_kkt_residual = current_kkt_res
        end
    end

    if !do_restart
        # Decide if we are going to do a restart.
        if restart_params.restart_scheme == ADAPTIVE_KKT
            do_restart = should_do_adaptive_restart_kkt(
                problem,
                candidate_kkt_residual,
                restart_params,
                last_restart_info,
                primal_weight,
                buffer_kkt,
                primal_norm_params,
                dual_norm_params,
            )
        elseif restart_params.restart_scheme == FIXED_FREQUENCY &&
            restart_params.restart_frequency_if_fixed <= restart_length
            do_restart = true
        end
    end


                # NEW: Check statistics of last
                function detect_oscillation(cuv::Any; window=10, min_amplitude=0.1, min_crossings=3)
                    v = Array(cuv)  # Move to CPU for analysis[1]
                    n = length(v)
                    if n < window+1
                        return false
                    end
                    # Compute running mean
                    ref = [mean(v[max(1,i-window):i]) for i in 1:n]
                    global_ref = mean(v)
                    stds = [std(v[max(1,i-window):i]) for i in 1:n]
                    global_std = std(v)
                    # Outlier indices
                    crossings = 0
                    out_idx = [i for i in 1:n if abs(v[i] - global_ref) > min_amplitude * global_std]
                    for (i,index) in enumerate(out_idx)
                        if i > 1
                            if (v[out_idx[i-1]] - global_ref) * (v[index] - global_ref) < 0
                                crossings += 1
                            end 
                        end
                    end
    
                    # # Detect zero crossings with amplitude threshold
                    # crossings = 0
                    # for i in 2:n
                    #     if (v[i-1] - ref[i-1]) * (v[i] - ref[i]) < 0 && abs(v[i] - v[i-1]) > min_amplitude * stds[i]
                    #     # if (v[i-1] - global_ref) * (v[i] - global_ref) < 0 && abs(v[i] - v[i-1]) > min_amplitude*stds[i]
                    #         crossings += 1
                    #     end
                    # end
                    # @info "crossings" crossings
                    return crossings ≥ min_crossings
                end

                function detect_improvement(cuv::Any; window=10, min_amplitude=0.1, min_crossings=3)
                    v = Array(cuv)  # Move to CPU for analysis[1]
                    n = length(v)
                    # @info n 
                    # @info window +1
                    # if n < window+1
                    #     return false
                    # end
                    # @info "computing stats..."
                    # compute statistics
                    global_ref = mean(v)
                    global_std = std(v)
                    # Improvement counter
                    # @info v .- global_ref
                    v_end = v[end-Int(floor(n/2)):end]
                    crossings = sum(v_end .- global_ref .< -min_amplitude * global_std) #- 1e-6)
                    successes2 = 0
                    for i in 2:length(v_end)
                        if v_end[i-1] > v_end[i] 
                            successes2 += 1
                        end
                    end
                    @info "successes", crossings
                    @info "successes2", successes2
                    return crossings ≥ min_crossings || successes2 >= min_crossings #  
                end

                function curvature(x::AbstractVector)
                    n = length(x)
                    if n < 3
                        error("Need at least 3 points to compute curvature.")
                    end
                    return x[3:end] .- 2 .* x[2:end-1] .+ x[1:end-2]  # second discrete difference
                end

                function detect_flat_stage(cuv::Any; window=10, min_amplitude=0.1, min_crossings=3, tolerance=1e-6)
                    v = Array(cuv)  # Move to CPU for analysis[1]
                    n = length(v)
                    global_ref = mean(v)
                    global_std = std(v)
                    # Improvement counter
                    # @info v .- global_ref
                    v_end = v[end-Int(floor(n/2)):end]
                    if global_std <= tolerance && sum(abs.(v_end[end] - mean(v_end)) .<= min_amplitude * std(v_end)) >= min_crossings
                        @info "Curvature: " curvature(v_end)
                        return true
                    end
                    bad_crossings = sum(v_end .- global_ref .>= -min_amplitude * global_std) #- 1e-6)
                    @info "bad_crossings", bad_crossings
                    return bad_crossings >= min_crossings  && v_end[end] - global_ref .>= -min_amplitude * global_std && std(v_end) <= tolerance && mean(v_end) >= global_ref
                end
    

                function is_flat(x::Any; tol_rel=1e-6, tail=5)
                    n = length(x)
                    if n == 0
                        return true
                    elseif all(x .== x[1])
                        return true
                    end
                
                    Δ = maximum(x) - minimum(x)
                    rel_change = Δ / abs(x[1] != 0 ? x[1] : 1.0)
                
                    # Check relative change over all values
                    if rel_change < tol_rel
                        return true
                    end
                
                    # Optional: check last `tail` values for recent stagnation
                    if n > tail
                        tail_change = maximum(x[end-tail+1:end]) - minimum(x[end-tail+1:end])
                        rel_tail = tail_change / abs(x[end-tail] != 0 ? x[end-tail] : 1.0)
                        if rel_tail < tol_rel
                            return true
                        end
                    end
                
                    return false
                end
                

                # Flat-behavior detection with surr_min and surr_max
                function surr_min(x::Any, t_start::Int=1, t_end::Int=typemax(Int))
                    t_end = min(t_end, length(x))
                    min_x = x[t_start]
                    x_copy = Float64[]
                    for t in t_start:t_end
                        min_x = min(min_x, x[t])
                        push!(x_copy, min_x)
                    end
                    return x_copy
                end

                function surrogate(x::Any, surr_type::String="min", t_start::Int=1, t_end::Int=typemax(Int))
                    if surr_type == "min"
                        return surr_min(x, t_start, t_end)
                    elseif surr_type == "max"
                        return -surr_min(-x, t_start, t_end)
                    else
                        error("Unsupported surrogate type: $surr_type")
                    end
                end


                function is_flattening(x::Any; tolerance::Float64=1e-6, delta_t::Int=5)
                    max_surr = surrogate(x, "max")[2:end]
                    min_surr = surrogate(x, "min")[2:end]
                    # @info "Entering is flattening..."
                    # @info max_surr
                    # @info min_surr
                    # @info max_surr .- min_surr
                    if !any(max_surr .- min_surr .> tolerance)
                        # @info "TRUE"
                        return true
                    end

                    for t in delta_t:(length(x)-delta_t)
                        first_half_max = surrogate(x, "max", 1, t)[2:end]
                        first_half_min = surrogate(x, "min", 1, t)[2:end]
                        first_half_is_big = !any(first_half_max .- first_half_min .<= tolerance)
                        # @info "First half info for t=", t
                        # @info first_half_max 
                        # @info first_half_min
                        # @info first_half_max .- first_half_min

                        second_half_max = surrogate(x, "max", t, typemax(Int))
                        second_half_min = surrogate(x, "min", t, typemax(Int))
                        secnd_half_is_sml = !any(second_half_max .- second_half_min .> tolerance)
                        # @info second_half_max .- second_half_min
                        # if first_half_is_big && secnd_half_is_sml
                        if secnd_half_is_sml
                            # @info "TRUE"
                            return true
                        end
                    end
                    # error("stop")
                    return false
                end


                # function std_gpu(x::CuArray; dims=:)
                #     μ = mean(x; dims=dims)
                #     centered = x .- μ
                #     sq = centered .^ 2  # This should work
                #     sqrt.(mean(sq; dims=dims))
                # end
                n_last_iterations = 20
                oscillation_trigger = true
                flat_kkt_trigger = false
                near_bounds_trigger = false
                if pdhg_iteration >= max(n_last_iterations +1, ir_iteration_threshold) && last_restart_info.ir_refinements_number == 0 && CUDA.size(primal_dual_iterates[1])[1] > 0 && ir_over_restart

                    # # Primal iterates
                    # # @info primal_dual_iterates
                    # last_primal_iterates = primal_dual_iterates[1][end-n_last_iterations+1:end,:]
                    # last_primal_iterates_cpu = zeros(n_last_iterations, 3)
                    # for (i, row) in enumerate(last_primal_iterates)
                    #     last_primal_iterates_cpu[i,:] = Array(row)
                    # end
                    # for col_index in 1:size(last_primal_iterates_cpu)[2]
                    #     std_dev = std(last_primal_iterates_cpu[:,col_index])
                        
                    #     # Detect oscillation behavior
                    #     oscillation_detected = detect_oscillation(last_primal_iterates_cpu[:,col_index], window=n_last_iterations-1, min_amplitude=0.1, min_crossings=3)
                    #     # println("Oscillation detected: ", oscillation_detected)
                    #     if oscillation_detected
                    #         # @info "Oscillation detected (primal): ", oscillation_detected
                    #         # @info last_primal_iterates_cpu[:,col_index]
                    #         # @info "pdgh iter" pdhg_iteration
                    #         oscillation_trigger = true
                    #         break
                    #     end
                    # end
    
                    # # Dual iterates
                    # if !oscillation_trigger
                    #     last_dual_iterates = primal_dual_iterates[2][end-n_last_iterations+1:end,:]
                    #     # @info "matrix iterates" last_dual_iterates
                    #     last_dual_iterates_cpu = zeros(n_last_iterations, 3)
                    #     for (i, row) in enumerate(last_dual_iterates)
                    #         last_dual_iterates_cpu[i,:] = Array(row)
                    #     end
                    #     # oscillation_detected = false
                    #     for col_index in 1:size(last_dual_iterates_cpu)[2]
                    #         std_dev = std(last_dual_iterates_cpu[:,col_index])
                            
                    #         # Detect oscillation behavior
                    #         oscillation_detected = detect_oscillation(last_dual_iterates_cpu[:,col_index], window=n_last_iterations-1, min_amplitude=0.1, min_crossings=3)
                    #         # println("Oscillation detected: ", oscillation_detected)
                    #         if oscillation_detected
                    #             # @info "Oscillation detected (dual): ", oscillation_detected
                    #             # @info last_dual_iterates_cpu[:,col_index]
                    #             # @info "pdgh iter" pdhg_iteration
                    #             oscillation_trigger = true
                    #             break
                    #         end
                    #     end
                    # end
                    # obj_pdhg_iter = 7500
                    # If oscillation detected, check KKT residuals improvement
                    # n_last_iterations_kkt = n_last_iterations
                    if oscillation_trigger#|| pdhg_iteration >= 50

                        # @info "Oscillation detected: Checking KKT residuals..."
                        # @info CUDA.size(primal_dual_iterates[3]), n_last_iterations_kkt
                        last_kkt_values = primal_dual_iterates[3][end-n_last_iterations+1:end]
                        primal_near_bounds_number = primal_dual_iterates[4][end-n_last_iterations+1:end]
                        # @info "Last KKT values: ", last_kkt_values 
                        # improvement_detected = detect_improvement(last_kkt_values, window=n_last_iterations, min_amplitude=0.1, min_crossings=3)
                        # flat_kkt_trigger = detect_flat_stage(last_kkt_values, window=n_last_iterations, min_amplitude=0.9, min_crossings=8, tolerance=termination_criteria.eps_optimal_relative)
                        # flat_kkt_trigger = is_flat(last_kkt_values, tol_rel=termination_criteria.eps_optimal_relative, tail=n_last_iterations) || detect_oscillation(last_kkt_values, window=n_last_iterations, min_amplitude=0.5, min_crossings=15)
                        # near_bounds_trigger = is_flat(primal_near_bounds_number, tol_rel=termination_criteria.eps_optimal_relative, tail=n_last_iterations) || detect_oscillation(primal_near_bounds_number, window=n_last_iterations, min_amplitude=0.5, min_crossings=15)
                        # @info "last_kkt_values" last_kkt_values
                        # @info "primal near bounds #" primal_near_bounds_number
                        # @info "ITERATION", pdhg_iteration
                        # max_diff_kkt = maximum(last_kkt_values) - minimum(last_kkt_values) 
                        max_diff_bounds = maximum(primal_near_bounds_number) - minimum(primal_near_bounds_number)
                        flat_tolerance = min(max(mean(last_kkt_values)*1e-3, termination_criteria.eps_optimal_relative), 1e-3)
                        flat_tolerance_2 = min(max(max_diff_bounds*1e-1, 1e-1),10)
                        flat_kkt_trigger = is_flattening(last_kkt_values; tolerance=flat_tolerance, delta_t=5)
                        near_bounds_trigger = is_flattening(primal_near_bounds_number; tolerance=flat_tolerance_2, delta_t=5)
                        
                        # @info "Improvement detected: ", flat_kkt_trigger
                        # flat_kkt_trigger = !improvement_detected

                        if 100 <= pdhg_iteration <= 150
                            @info "ITERATION: ", pdhg_iteration
                            @info "KKT", last_kkt_values
                            @info "BDS", primal_near_bounds_number
                            @info "history of BDS", primal_dual_iterates[4][end-50+1:end]

                        elseif 251<= pdhg_iteration
                            @info "LAST ITERATIONS: ", pdhg_iteration
                            @info "All bounds history", primal_dual_iterates[4]
                        end
                        # if flat_kkt_trigger && !near_bounds_trigger
                        #     @info "Flat kkt trigger: ", flat_kkt_trigger
                        #     @info "\n====== iter:", pdhg_iteration
                        # elseif near_bounds_trigger && !flat_kkt_trigger
                        #     @info "Near bound trigger: ", near_bounds_trigger
                        #     @info "\n====== iter:", pdhg_iteration
                        # end
                        if flat_kkt_trigger && near_bounds_trigger #|| pdhg_iteration >= obj_pdhg_iter#   #
                            # sleep(5)

                            if !primal_dual_iterates[5]["first_flat_stage"] 
                                primal_dual_iterates[5]["first_flat_stage"] = true 
                                @info "## NOT TRIGGERED ##"
                                @info "Flat tolerance on iter k="*string(pdhg_iteration)*": ", flat_tolerance
                                @info "Last KKT values: ", last_kkt_values
                                @info "Flat tolerance 2 on iter k="*string(pdhg_iteration)*": ", flat_tolerance_2
                                @info "# of near bounds: ", primal_near_bounds_number

                                @info "FIRST STAGE TO TRUE IN ITER:", pdhg_iteration 
                            elseif primal_dual_iterates[5]["first_flat_stage"] && primal_dual_iterates[5]["non_flat_stage"] && !primal_dual_iterates[5]["second_flat_stage"] 
                                # @info "Did it save the bool??" primal_dual_iterates[5]["first_flat_stage"]
                                @info "## TRIGGERED ##"
                                @info "Flat tolerance on iter k="*string(pdhg_iteration)*": ", flat_tolerance
                                @info "Last KKT values: ", last_kkt_values
                                @info "Flat tolerance 2 on iter k="*string(pdhg_iteration)*": ", flat_tolerance_2
                                @info "# of near bounds: ", primal_near_bounds_number
                                @info "SECOND FLAT STAGE IN ITER:", pdhg_iteration
                                primal_dual_iterates[5]["second_flat_stage"] = true
                                do_restart = true
                                # error("WOAO")
                            end
                            # exit()
                            # @info primal_dual_iterates[4]
                            # error("stopping")
                        elseif !near_bounds_trigger # near bounds stop being flat
                            if primal_dual_iterates[5]["first_flat_stage"] && !primal_dual_iterates[5]["non_flat_stage"]
                                @info "## NOT TRIGGERED ##"
                                @info "Flat tolerance on iter k="*string(pdhg_iteration)*": ", flat_tolerance
                                @info "Last KKT values: ", last_kkt_values
                                @info "Flat tolerance 2 on iter k="*string(pdhg_iteration)*": ", flat_tolerance_2
                                @info "# of near bounds: ", primal_near_bounds_number
                                primal_dual_iterates[5]["non_flat_stage"] = true 
                                @info "NON FLAT STAGE TO TRUE IN ITER:", pdhg_iteration
                            end
                        end
                        # if pdhg_iteration >= 50
                        #     @info primal_dual_iterates[4]
                        #     error("Breaking...")
                        # else
                        #     @info "ITERATION" pdhg_iteration
                        # end
                        # exit()
                    end
                    # primal_dual_iterates
                    # current_iteration_stats
                end 

    ir_refinement_applied = false # MINE
    
    if !do_restart
        return RESTART_CHOICE_NO_RESTART
    else # MINE: If we do restart
        if reset_to_average
            if verbosity >= 6
                print("  Restarted to average")
            end
            # ir_iteration_threshold = 1
            # if !(ir_over_restart && (pdhg_iteration >= ir_iteration_threshold) && last_restart_info.ir_refinements_number == 0) # && 1<0# && last_restart_info.last_restart_length <= 2 # only 1 iter

            current_primal_solution .= buffer_avg.avg_primal_solution
            current_dual_solution .= buffer_avg.avg_dual_solution
            primal_product .= buffer_avg.avg_primal_product
            dual_product .= problem.objective_vector .- buffer_avg.avg_primal_gradient
            buffer_primal_gradient .= buffer_avg.avg_primal_gradient

        else
        # Current point is much better than average point.
            if verbosity >= 4
                print("  Restarted to current")
            end
        end

            # MINE: Apply IR over the average/current solution
            # 500 <= last_restart_info.last_restart_length <= 1000 && 0 >1
            # && isnothing(last_restart_info.last_restart_kkt_residual)
            # NOTE: I am not sure what "last_restart_length" is. I am assuming it is the distance of the current iteration and the last restart.
            #elseif
            if ir_over_restart && (pdhg_iteration >= ir_iteration_threshold) && last_restart_info.ir_refinements_number == 0 &&
                flat_kkt_trigger && near_bounds_trigger && primal_dual_iterates[5]["second_flat_stage"] # && last_restart_info.last_restart_length <= 2 # only 1 iter
                # if ir_over_restart && (pdhg_iteration >= ir_iteration_threshold) && last_restart_info.ir_refinements_number == 0 # && 1<0# && last_restart_info.last_restart_length <= 2 # only 1 iter
                println("threshold: ", ir_iteration_threshold)
                @info "candidate res: ", candidate_kkt_residual
                # println("buffer avg: ", buffer_avg)
                # println("average res: ", avg_kkt_res)

                # exit()
                # println("Buffer KKT: ", fieldnames(buffer_kkt))
                @info "I AM DOING ["*string(ir_type)*"] IR (iteration threshold:"*string(ir_iteration_threshold)*")"
                @info "cuPDLP iteration: "*string(pdhg_iteration)

                # println("PRE-Checking feasibility of IR solution...")
                # println("1) Primal feasibility violations:")
                # println("1.1) Lower bound: ", sum(problem.variable_lower_bound .> current_primal_solution))
                # println("1.2) Upper bound: ", sum(problem.variable_upper_bound .< current_primal_solution))
                # println("1.3) At least Ax>=b: ", sum(problem.constraint_matrix*current_primal_solution .< problem.right_hand_side)) #
    
                # println("CURRENT PDHG ITERATION: ", pdhg_iteration)
                # println("LAST RESTART LENGTH: ", last_restart_info.last_restart_length)
                # println("RESTART LENGTH: ", restart_length)
                # println("Last restart ir number: ", last_restart_info.ir_refinements_number)

                # # println("LAST RESTART 2: ", last_restart_info.primal_distance_moved_last_restart_period)
                # # println("LAST RESTART 3:", last_restart_info.kkt_reduction_ratio_last_trial)
                # # println("LAST RESTART 4:", last_restart_info.)


                # DO: Iterative refinement
                t0 = time()
                ir_output = iterative_refinement(
                problem,            # Potentially: Scaled problem of cuPDLP
                current_primal_solution,       
                current_dual_solution,
                # My inputs
                current_iteration_stats,
                qp_cache,
                termination_criteria.eps_optimal_relative, # tolerance
                ir_tolerance_factor,        # ir_tolerance_factor
                1.1,        # alpha
                ir_type,   # scaling type
                # "matrix",
                "scalar",   # scaling name
                1, # max iteration
                iteration_limit, #iteration limit
                )
                println("Time spent doing Refinement: ", time()- t0)
                
                if !isnothing(ir_output)

                    current_primal_solution .= ir_output.current_primal_solution
                    current_dual_solution .= ir_output.current_dual_solution
                    primal_product .= ir_output.primal_product
                    dual_product .= ir_output.dual_product
                    buffer_primal_gradient .= ir_output.primal_gradient

                end

                ir_refinement_applied = true
                
            end

        if verbosity >= 4
            print(" after ", rpad(restart_length, 4), " iterations")
            if artificial_restart
                println("*")
            else
                println("")
            end
        end
        reset_solution_weighted_average!(solution_weighted_avg)

        update_last_restart_info!(
            last_restart_info,
            current_primal_solution,
            current_dual_solution,
            buffer_avg.avg_primal_solution,
            buffer_avg.avg_dual_solution,
            primal_weight,
            primal_norm_params,
            dual_norm_params,
            candidate_kkt_residual,
            restart_length,
            primal_product,
            buffer_primal_gradient,
            ir_refinement_applied
        )

        if reset_to_average
            return RESTART_CHOICE_RESTART_TO_AVERAGE
        else
            return RESTART_CHOICE_WEIGHTED_AVERAGE_RESET
        end
    end
end

"""
Compute primal weight at restart
"""
function compute_new_primal_weight(
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    primal_weight_update_smoothing::Float64,
    verbosity::Int64,
)
    primal_distance = last_restart_info.primal_distance_moved_last_restart_period
    dual_distance = last_restart_info.dual_distance_moved_last_restart_period
    
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / primal_distance
        # Exponential moving average.
        # If primal_weight_update_smoothing = 1.0 then there is no smoothing.
        # If primal_weight_update_smoothing = 0.0 then the primal_weight is frozen.
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)

        primal_weight = exp(log_primal_weight)
        if verbosity >= 4
            Printf.@printf "  New computed primal weight is %.2e\n" primal_weight
        end

        return primal_weight
    else
        return primal_weight
    end
end

"""
Update last restart info
"""
function update_last_restart_info!(
    last_restart_info::CuRestartInfo,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    avg_primal_solution::CuVector{Float64},
    avg_dual_solution::CuVector{Float64},
    primal_weight::Float64,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    candidate_kkt_residual::Union{Nothing,CuKKTrestart},
    restart_length::Int64,
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
    ir_refinement_applied::Bool, # MINE
)
    last_restart_info.primal_distance_moved_last_restart_period =
        weighted_norm(
            avg_primal_solution - last_restart_info.primal_solution,
            primal_norm_params,
        ) / sqrt(primal_weight)
    last_restart_info.dual_distance_moved_last_restart_period =
        weighted_norm(
            avg_dual_solution - last_restart_info.dual_solution,
            dual_norm_params,
        ) * sqrt(primal_weight)
    last_restart_info.primal_solution .= current_primal_solution
    last_restart_info.dual_solution .= current_dual_solution

    last_restart_info.last_restart_length = restart_length
    last_restart_info.last_restart_kkt_residual = candidate_kkt_residual

    last_restart_info.primal_product .= primal_product
    last_restart_info.primal_gradient .= primal_gradient

    last_restart_info.ir_refinement_applied = ir_refinement_applied 
    last_restart_info.ir_refinements_number += ir_refinement_applied
end


function point_type_label(point_type::PointType)
    if point_type == POINT_TYPE_CURRENT_ITERATE
        return "current"
    elseif point_type == POINT_TYPE_AVERAGE_ITERATE
        return "average"
    elseif point_type == POINT_TYPE_ITERATE_DIFFERENCE
        return "difference"
    else
        return "unknown PointType"
    end
end


function generic_final_log(
    problem::Union{QuadraticProgrammingProblem, CuQuadraticProgrammingProblem},
    current_primal_solution::Vector{Float64},
    current_dual_solution::Vector{Float64},
    last_iteration_stats::IterationStats,
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
)
    if verbosity >= 1
        print("Terminated after $iteration iterations: ")
        println(termination_reason_to_string(termination_reason))
    end

    method_specific_stats = last_iteration_stats.method_specific_stats
    if verbosity >= 3
        for convergence_information in last_iteration_stats.convergence_information
            Printf.@printf(
                "For %s candidate:\n",
                point_type_label(convergence_information.candidate_type)
            )
            # Print more decimal places for the primal and dual objective.
            Printf.@printf(
                "Primal objective: %f, ",
                convergence_information.primal_objective
            )
            Printf.@printf(
                "dual objective: %f, ",
                convergence_information.dual_objective
            )
            Printf.@printf(
                "corrected dual objective: %f \n",
                convergence_information.corrected_dual_objective
            )
        end
    end
    if verbosity >= 4
        Printf.@printf(
            "Time (seconds):\n - Basic algorithm: %.2e\n - Full algorithm:  %.2e\n",
            method_specific_stats["time_spent_doing_basic_algorithm"],
            last_iteration_stats.cumulative_time_sec,
        )
    end

    if verbosity >= 7
        for convergence_information in last_iteration_stats.convergence_information
            print_infinity_norms(convergence_information)
        end
        print_variable_and_constraint_hardness(
            problem,
            current_primal_solution,
            current_dual_solution,
        )
    end
end

"""
Initialize primal weight
"""
function select_initial_primal_weight(
    problem::CuLinearProgrammingProblem,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_importance::Float64,
    verbosity::Int64,
)
    rhs_vec_norm = weighted_norm(problem.right_hand_side, dual_norm_params)
    obj_vec_norm = weighted_norm(problem.objective_vector, primal_norm_params)
    if obj_vec_norm > 0.0 && rhs_vec_norm > 0.0
        primal_weight = primal_importance * (obj_vec_norm / rhs_vec_norm)
    else
        primal_weight = primal_importance
    end
    if verbosity >= 6
        println("Initial primal weight = $primal_weight")
    end
    return primal_weight
end

