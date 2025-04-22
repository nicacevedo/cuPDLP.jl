
struct AdaptiveStepsizeParams
    reduction_exponent::Float64
    growth_exponent::Float64
end

struct ConstantStepsizeParams end

struct PdhgParameters
    l_inf_ruiz_iterations::Int
    l2_norm_rescaling::Bool
    pock_chambolle_alpha::Union{Float64,Nothing}
    primal_importance::Float64
    scale_invariant_initial_primal_weight::Bool
    verbosity::Int64
    record_iteration_stats::Bool
    termination_evaluation_frequency::Int32
    termination_criteria::TerminationCriteria
    restart_params::RestartParameters
    step_size_policy_params::Union{
        AdaptiveStepsizeParams,
        ConstantStepsizeParams,
    }
end

mutable struct CuPdhgSolverState
    current_primal_solution::CuVector{Float64}
    current_dual_solution::CuVector{Float64}
    current_primal_product::CuVector{Float64}
    current_dual_product::CuVector{Float64}
    solution_weighted_avg::CuSolutionWeightedAverage 
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
end


mutable struct CuBufferState
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    delta_primal_product::CuVector{Float64}
end


function define_norms(
    primal_size::Int64,
    dual_size::Int64,
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end
  

function pdhg_specific_log(
    # problem::QuadraticProgrammingProblem,
    iteration::Int64,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        # "   %5d inv_step_size=%9g ",
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        CUDA.norm(current_primal_solution),
        CUDA.norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end

function pdhg_final_log(
    problem::Union{QuadraticProgrammingProblem, CuQuadraticProgrammingProblem},
    avg_primal_solution::Vector{Float64},
    avg_dual_solution::Vector{Float64},
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
    last_iteration_stats::IterationStats,
)

    if verbosity >= 2
        # infeas = max_primal_violation(problem, avg_primal_solution)
        # primal_obj_val = primal_obj(problem, avg_primal_solution)
        # dual_stats =
        #     compute_dual_stats(problem, avg_primal_solution, avg_dual_solution)
        
        println("Avg solution:")
        Printf.@printf(
            "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
            last_iteration_stats.convergence_information[1].l_inf_primal_residual,
            last_iteration_stats.convergence_information[1].primal_objective,
            last_iteration_stats.convergence_information[1].l_inf_dual_residual,
            last_iteration_stats.convergence_information[1].dual_objective
        )
        Printf.@printf(
            "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_primal_solution, 1),
            CUDA.norm(avg_primal_solution),
            CUDA.norm(avg_primal_solution, Inf)
        )
        Printf.@printf(
            "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_dual_solution, 1),
            CUDA.norm(avg_dual_solution),
            CUDA.norm(avg_dual_solution, Inf)
        )
    end

    generic_final_log(
        problem,
        avg_primal_solution,
        avg_dual_solution,
        last_iteration_stats,
        verbosity,
        iteration,
        termination_reason,
    )
end

function power_method_failure_probability(
    dimension::Int64,
    epsilon::Float64,
    k::Int64,
)
    if k < 2 || epsilon <= 0.0
        return 1.0
    end
    return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) * sqrt(dimension) * (1.0 - epsilon)^(k - 1 / 2) # FirstOrderLp.jl old version (epsilon * (k - 1)) instead of sqrt(epsilon * (k - 1)))
end

function estimate_maximum_singular_value(
    matrix::SparseMatrixCSC{Float64,Int64};
    probability_of_failure = 0.01::Float64,
    desired_relative_error = 0.1::Float64,
    seed::Int64 = 1,
)
    epsilon = 1.0 - (1.0 - desired_relative_error)^2
    x = randn(Random.MersenneTwister(seed), size(matrix, 2))

    number_of_power_iterations = 0
    while power_method_failure_probability(
        size(matrix, 2),
        epsilon,
        number_of_power_iterations,
    ) > probability_of_failure
        x = x / norm(x, 2)
        x = matrix' * (matrix * x)
        number_of_power_iterations += 1
    end
    
    return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
    number_of_power_iterations
end

"""
Kernel to compute primal solution in the next iteration
"""
function compute_next_primal_solution_kernel!(
    objective_vector::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    current_primal_solution::CuDeviceVector{Float64},
    current_dual_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_variables::Int64,
    delta_primal::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            delta_primal[tx] = current_primal_solution[tx] - (step_size / primal_weight) * (objective_vector[tx] - current_dual_product[tx])
            delta_primal[tx] = min(variable_upper_bound[tx], max(variable_lower_bound[tx], delta_primal[tx]))
            delta_primal[tx] -= current_primal_solution[tx]
        end
    end
    return 
end

"""
Compute primal solution in the next iteration
"""
function compute_next_primal_solution!(
    problem::CuLinearProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal::CuVector{Float64},
    delta_primal_product::CuVector{Float64},
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_next_primal_solution_kernel!(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        current_primal_solution,
        current_dual_product,
        step_size,
        primal_weight,
        problem.num_variables,
        delta_primal,
    )

    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix, delta_primal, 0, delta_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    
end

"""
Kernel to compute dual solution in the next iteration
"""
function compute_next_dual_solution_kernel!(
    right_hand_side::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_product::CuDeviceVector{Float64},
    delta_primal_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    extrapolation_coefficient::Float64,
    num_equalities::Int64,
    num_constraints::Int64,
    delta_dual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])

            delta_dual[tx] -= current_dual_solution[tx]
        end
    elseif num_equalities + 1 <= tx <= num_constraints
        @inbounds begin
            delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])
            delta_dual[tx] = max(delta_dual[tx], 0.0)

            delta_dual[tx] -= current_dual_solution[tx]
        end
    end
    return 
end

"""
Compute dual solution in the next iteration
"""
function compute_next_dual_solution!(
    problem::CuLinearProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    delta_dual::CuVector{Float64},
    extrapolation_coefficient::Float64 = 1.0,
)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        delta_primal_product,
        step_size,
        primal_weight,
        extrapolation_coefficient,
        problem.num_equalities,
        problem.num_constraints,
        delta_dual,
    )

    # next_dual_product .= problem.constraint_matrix_t * next_dual
    # CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

"""
Update primal and dual solutions
"""
function update_solution_in_solver_state!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    # solver_state.current_primal_solution .= copy(buffer_state.next_primal)
    solver_state.current_primal_solution .+= buffer_state.delta_primal
    solver_state.current_primal_product .+= buffer_state.delta_primal_product

    solver_state.current_dual_solution .+= buffer_state.delta_dual
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, solver_state.current_dual_solution, 0, solver_state.current_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    
    weight = solver_state.step_size
    
    add_to_solution_weighted_average!(
        solver_state.solution_weighted_avg,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        weight,
        solver_state.current_primal_product,
        solver_state.current_dual_product,
    )
end

"""
Compute iteraction and movement for AdaptiveStepsize
"""
function compute_interaction_and_movement(
    solver_state::CuPdhgSolverState,
    problem::CuLinearProgrammingProblem,
    buffer_state::CuBufferState,
)
    primal_dual_interaction = CUDA.dot(buffer_state.delta_primal_product, buffer_state.delta_dual) 
    interaction = abs(primal_dual_interaction) 

    norm_delta_primal = CUDA.norm(buffer_state.delta_primal)
    norm_delta_dual = CUDA.norm(buffer_state.delta_dual)

    movement = 0.5 * solver_state.primal_weight * norm_delta_primal^2 + (0.5 / solver_state.primal_weight) * norm_delta_dual^2

    return interaction, movement
end

"""
Take PDHG step with AdaptiveStepsize
"""
function take_step!(
    step_params::AdaptiveStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    step_size = solver_state.step_size
    done = false

    while !done
        solver_state.total_number_iterations += 1

        compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal,
            buffer_state.delta_primal_product,
        )

        compute_next_dual_solution!(
            problem,
            solver_state.current_dual_solution,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal_product,
            solver_state.current_primal_product,
            buffer_state.delta_dual,
        )


        interaction, movement = compute_interaction_and_movement(
            solver_state,
            problem,
            buffer_state,
        )

        solver_state.cumulative_kkt_passes += 1

        if interaction > 0
            step_size_limit = movement / interaction
            if movement == 0.0
                # The algorithm will terminate at the beginning of the next iteration
                solver_state.numerical_error = true
                break
            end
        else
            step_size_limit = Inf
        end

        if step_size <= step_size_limit
            update_solution_in_solver_state!(
                problem,
                solver_state, 
                buffer_state,
            )
            done = true
        end


        first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(step_params.reduction_exponent)) * step_size_limit

        second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(step_params.growth_exponent)) * step_size

        step_size = min(first_term, second_term)
        
    end  
    solver_state.step_size = step_size
end

"""
Take PDHG step with ConstantStepsize
"""
function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    compute_next_primal_solution!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal,
        buffer_state.delta_primal_product,
    )
    

    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        solver_state.step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal_product,
        solver_state.current_primal_product,
        buffer_state.delta_dual,
    )

    solver_state.cumulative_kkt_passes += 1

    update_solution_in_solver_state!(
        problem,
        solver_state, 
        buffer_state,
    )
end

"""
Main algorithm: given parameters and LP problem, return solutions
"""
function optimize(
    params::PdhgParameters,
    original_problem::Union{QuadraticProgrammingProblem, CuQuadraticProgrammingProblem},
    # initial_solver_state::Any = nothing # ::Union{CuPdhgSolverState, Nothing} = nothing # MINE
    ir_instead_average::Bool=false,
    ir_type::String="scalar",
    # lp_original_problem::Any=nothing,
    init_primal_dual_solution::Any = nothing,
    init_problem_params::Any = nothing,
    )

    # # Skip preporcessing if GPU
    # println("Type of the problem", typeof(original_problem))
    # println("Is a cuquad", isa(original_problem, CuQuadraticProgrammingProblem))
    # flush(stdout)

    # GPU Checked: We already know is valid if CuQuadratic
    if !isa(original_problem, CuQuadraticProgrammingProblem)
        validate(original_problem) 
    end
    # GPU Checked: Cache is calculating norms with norm/CUDA.norm functions accordingly
    qp_cache = cached_quadratic_program_info(original_problem)

    # @info "original problem (init)"
    # @info CUDA.norm(original_problem.constraint_matrix, 2)
    # @info CUDA.norm(original_problem.objective_matrix, 2)
    # @info CUDA.norm(original_problem.right_hand_side, 2)
    

    start_rescaling_time = time()
    # GPU Checked: same results for both scalings 
    if !isa(original_problem, CuQuadraticProgrammingProblem)
        scaled_problem = rescale_problem(
            params.l_inf_ruiz_iterations,
            params.l2_norm_rescaling,
            params.pock_chambolle_alpha,
            params.verbosity,
            original_problem,
        )
    else 
        scaled_problem = no_rescale_problem(
        # scaled_problem = rescale_problem(
            params.l_inf_ruiz_iterations,
            params.l2_norm_rescaling,
            params.pock_chambolle_alpha,
            params.verbosity,
            original_problem,
        )
    end
    
    # @info "Scaled problem"
    # @info CUDA.norm(scaled_problem.scaled_qp.constraint_matrix, 2)
    # @info CUDA.norm(scaled_problem.scaled_qp.objective_matrix, 2)
    # @info CUDA.norm(scaled_problem.scaled_qp.right_hand_side, 2)
    

    rescaling_time = time() - start_rescaling_time
    Printf.@printf(
        "Preconditioning Time (seconds): %.2e\n",
        rescaling_time,
    )

    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    # GPU Checked: transforms qp to lp in most of parts, and also GPU if necessary
    d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
    d_problem = d_scaled_problem.scaled_qp
    buffer_lp = qp_cpu_to_gpu(original_problem)

    # @info "\n// Rescaled problem CPU-GPU thing //"
    # @info CUDA.norm(d_problem.constraint_matrix, 2)
    # # @info CUDA.norm(d_problem.objective_matrix, 2)
    # @info CUDA.norm(d_problem.right_hand_side, 2)
    # @info CUDA.norm(buffer_lp.constraint_matrix, 2)
    # # @info CUDA.norm(buffer_lp.objective_matrix, 2)
    # @info CUDA.norm(buffer_lp.right_hand_side, 2)


    # initialization 
    # MINE: Condition for a fist solver state
    if isnothing(init_primal_dual_solution)
        solver_state = CuPdhgSolverState(
            CUDA.zeros(Float64, primal_size),    # current_primal_solution
            CUDA.zeros(Float64, dual_size),      # current_dual_solution
            CUDA.zeros(Float64, dual_size),      # current_primal_product
            CUDA.zeros(Float64, primal_size),    # current_dual_product
            initialize_solution_weighted_average(primal_size, dual_size),
            0.0,                 # step_size
            1.0,                 # primal_weight
            false,               # numerical_error
            0.0,                 # cumulative_kkt_passes
            0,                   # total_number_iterations
            nothing,
            nothing,
        )

        buffer_state = CuBufferState(
            CUDA.zeros(Float64, primal_size),             # delta_primal (Is primal - step_size/primal_weight*etc)
            CUDA.zeros(Float64, dual_size),     # delta_dual
            CUDA.zeros(Float64, dual_size),     # delta_primal_product
        )

        buffer_avg = CuBufferAvgState(
            CUDA.zeros(Float64, primal_size),      # avg_primal_solution
            CUDA.zeros(Float64, dual_size),        # avg_dual_solution
            CUDA.zeros(Float64, dual_size),        # avg_primal_product
            CUDA.zeros(Float64, primal_size),      # avg_primal_gradient
        )

        buffer_original = BufferOriginalSol(
            CUDA.zeros(Float64, primal_size),      # primal
            CUDA.zeros(Float64, dual_size),        # dual
            CUDA.zeros(Float64, dual_size),        # primal_product
            CUDA.zeros(Float64, primal_size),      # primal_gradient
        )

    else # MINE: If it is an array (SKIP IN GPU VERSION)
        # Original values
        i=0
        i+=1
        print("I got here "*string(i))
        x_k, y_k = init_primal_dual_solution
        x_k, y_k = (CuVector{Float64}(x_k), CuVector{Float64}(y_k))
        A,b,c,l,u = init_problem_params
        # Scaled problem values
        # println(typeof(x_k))
        # println(typeof(y_k))
        # println(typeof(d_scaled_problem.variable_rescaling))
        # println(typeof(d_scaled_problem.constraint_rescaling))
        # (x_k .* d_scaled_problem.variable_rescaling), (y_k .* d_scaled_problem.constraint_rescaling) = (
        #     x_k .* d_scaled_problem.variable_rescaling, 
        #     y_k .* d_scaled_problem.constraint_rescaling
        # )
        # d_problem.constraint_matrix, d_problem.variable_lower_bound, d_problem.variable_upper_bound = (
        #     d_problem.constraint_matrix,
        #     d_problem.right_hand_side,
        #     d_problem.objective_vector,
        #     d_problem.variable_lower_bound,
        #     d_problem.variable_upper_bound
        # )

        i+=1
        print("I got here "*string(i))
        primal_size, dual_size = (length(x_k), length(y_k))
        i+=1
        print("I got here "*string(i))
        solver_state = CuPdhgSolverState(
            (x_k .* d_scaled_problem.variable_rescaling),#CuVector{Float64}((x_k .* d_scaled_problem.variable_rescaling)),     #   current_primal_solution
            (y_k .* d_scaled_problem.constraint_rescaling),#CuVector{Float64}((y_k .* d_scaled_problem.constraint_rescaling)),     #   current_dual_solution
            d_problem.constraint_matrix * (x_k .* d_scaled_problem.variable_rescaling), #CuVector{Float64}(d_problem.constraint_matrix * (x_k .* d_scaled_problem.variable_rescaling)),   #   current_primal_product
            d_problem.constraint_matrix' * (y_k .* d_scaled_problem.constraint_rescaling), #CuVector{Float64}(d_problem.constraint_matrix' * (y_k .* d_scaled_problem.constraint_rescaling)),  #   current_dual_product
            initialize_solution_weighted_average(primal_size, dual_size), # This is computed later if 0 apparently
            0.0,                 # step_size
            1.0,                 # primal_weight
            false,               # numerical_error
            0.0,                 # cumulative_kkt_passes
            0,                   # total_number_iterations
            nothing,
            nothing,
        ) 
        i+=1
        print("I got here "*string(i))
        buffer_state = CuBufferState(
            CuVector{Float64}( min.(d_problem.variable_upper_bound, max.(d_problem.variable_lower_bound, (x_k .* d_scaled_problem.variable_rescaling))) - (x_k .* d_scaled_problem.variable_rescaling) ),    # delta_primal (Is primal - step_size/primal_weight*etc inside of the max)
            CuVector{Float64}( (y_k .* d_scaled_problem.constraint_rescaling) - (y_k .* d_scaled_problem.constraint_rescaling) ),    # delta_dual (similar to primal idea)
            CUDA.zeros(Float64, dual_size),     # delta_primal_product (probably something like zero for init)
        )
        i+=1
        print("I got here "*string(i))
        buffer_avg = CuBufferAvgState(
            CuVector{Float64}((x_k .* d_scaled_problem.variable_rescaling)),      # avg_primal_solution
            CuVector{Float64}((y_k .* d_scaled_problem.constraint_rescaling)),        # avg_dual_solution
            CuVector{Float64}(d_problem.constraint_matrix*(x_k .* d_scaled_problem.variable_rescaling)),        # avg_primal_product
            CuVector{Float64}(d_problem.objective_vector - d_problem.constraint_matrix' *(y_k .* d_scaled_problem.constraint_rescaling)),      # avg_primal_gradient
        )
        i+=1
        println("I got here "*string(i))
        println(typeof(A))
        println(typeof(CUDA.CUSPARSE.CuSparseMatrixCSR(A)))
        # x_k, y_k = init_primal_dual_solution # recover non-CuVector for operations
        buffer_original = BufferOriginalSol(
            x_k,      # primal
            y_k,        # dual
            CUDA.CUSPARSE.CuSparseMatrixCSR(A)*x_k,        # primal_product
            (CuVector{Float64}(c)-CUDA.CUSPARSE.CuSparseMatrixCSR(A)'*y_k),      # primal_gradient
        )

        i+=1
        print("I got here "*string(i))
    end

    buffer_kkt = BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    buffer_kkt_infeas = BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )

    buffer_primal_gradient = CUDA.zeros(Float64, primal_size)
    buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product

    # stepsize
    if params.step_size_policy_params isa AdaptiveStepsizeParams
        solver_state.cumulative_kkt_passes += 0.5
        solver_state.step_size = 1.0 / norm(scaled_problem.scaled_qp.constraint_matrix, Inf)
    else
        desired_relative_error = 0.2
        maximum_singular_value, number_of_power_iterations =
            estimate_maximum_singular_value(
                scaled_problem.scaled_qp.constraint_matrix,
                probability_of_failure = 0.001,
                desired_relative_error = desired_relative_error,
            )
        solver_state.step_size =
            (1 - desired_relative_error) / maximum_singular_value
        solver_state.cumulative_kkt_passes += number_of_power_iterations
    end

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            d_scaled_problem.scaled_qp,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        d_scaled_problem.scaled_qp,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
        buffer_primal_gradient,
    )

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading(params.verbosity)

    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 100 || # Limit of saving info (?) (MINE: changed the <= 10 limit to a diff value)
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            ### average ###
            if solver_state.numerical_error || solver_state.solution_weighted_avg.sum_primal_solutions_count == 0 || solver_state.solution_weighted_avg.sum_dual_solutions_count == 0
                buffer_avg.avg_primal_solution .= solver_state.current_primal_solution
                buffer_avg.avg_dual_solution .= solver_state.current_dual_solution
                buffer_avg.avg_primal_product .= solver_state.current_primal_product
                buffer_avg.avg_primal_gradient .= buffer_primal_gradient
            else
                compute_average!(solver_state.solution_weighted_avg, buffer_avg, d_problem)
            end

            ### KKT ###
            current_iteration_stats = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                buffer_avg.avg_primal_solution,
                buffer_avg.avg_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_AVERAGE_ITERATE, 
                buffer_avg.avg_primal_product,
                buffer_avg.avg_primal_gradient,
                buffer_original,
                buffer_kkt,
                buffer_kkt_infeas,
                buffer_lp,
            )
            # if iteration == 1052 #|| iteration == 13
            #     println("Iter "*string(iteration))
            #     println("norm of primal sol:")
            #     println(CUDA.norm(solver_state.current_primal_solution))
            #     println("norm of dual sol:")
            #     println(CUDA.norm(solver_state.current_dual_solution))
            #     println("relative KKT values from cuPDLP:")
            #     println(current_iteration_stats.convergence_information[1].relative_l_inf_primal_residual)
            #     println(current_iteration_stats.convergence_information[1].relative_l_inf_dual_residual)
            #     println(current_iteration_stats.convergence_information[1].relative_optimality_gap)
            #     println("KKT normal:")
            #     println(norm([
            #         current_iteration_stats.convergence_information[1].l_inf_primal_residual,
            #         current_iteration_stats.convergence_information[1].l_inf_dual_residual,
            #         current_iteration_stats.convergence_information[1].primal_objective - current_iteration_stats.convergence_information[1].dual_objective,
            #     ],1))
            #     # exit()
            # elseif iteration == 1051 #|| iteration == 12
            #     println("Iter "*string(iteration))
            #     println("norm of primal sol:")
            #     println(CUDA.norm(solver_state.current_primal_solution))
            #     println("norm of dual sol:")
            #     println(CUDA.norm(solver_state.current_dual_solution))
            #     println("relative KKT values from cuPDLP:")
            #     println(current_iteration_stats.convergence_information[1].relative_l_inf_primal_residual)
            #     println(current_iteration_stats.convergence_information[1].relative_l_inf_dual_residual)
            #     println(current_iteration_stats.convergence_information[1].relative_optimality_gap)  
            #     println("KKT normal:")
            #     println(norm([
            #         current_iteration_stats.convergence_information[1].l_inf_primal_residual,
            #         current_iteration_stats.convergence_information[1].l_inf_dual_residual,
            #         current_iteration_stats.convergence_information[1].primal_objective - current_iteration_stats.convergence_information[1].dual_objective,
            #     ],1))
            # end

            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                primal_size,
                dual_size,
                solver_state.step_size,
                solver_state.primal_weight,
            )
            
            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            if iteration < 10 && (termination_reason == TERMINATION_REASON_PRIMAL_INFEASIBLE || termination_reason == TERMINATION_REASON_DUAL_INFEASIBLE)
                termination_reason = false
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats, params.verbosity)
            end

            if termination_reason != false
                # ** Terminate the algorithm **
                # This is the only place the algorithm can terminate. Please keep it this way.
                
                # GPU to CPU
                if isa(original_problem, QuadraticProgrammingProblem)
                    avg_primal_solution = zeros(primal_size)
                    avg_dual_solution = zeros(dual_size)
                    gpu_to_cpu!(
                        buffer_avg.avg_primal_solution,
                        buffer_avg.avg_dual_solution,
                        avg_primal_solution,
                        avg_dual_solution,
                    )

                    pdhg_final_log(
                        scaled_problem.scaled_qp,
                        avg_primal_solution,
                        avg_dual_solution,
                        params.verbosity,
                        iteration,
                        termination_reason,
                        current_iteration_stats,
                    )
                elseif isa(original_problem, CuQuadraticProgrammingProblem)
                    # Keep it GPU
                    avg_primal_solution = buffer_avg.avg_primal_solution
                    avg_dual_solution = buffer_avg.avg_dual_solution


                    # pdhg_final_log(
                    #     scaled_problem.scaled_qp,
                    #     avg_primal_solution,
                    #     avg_dual_solution,
                    #     params.verbosity,
                    #     iteration,
                    #     termination_reason,
                    #     current_iteration_stats,
                    # )
                end

                return unscaled_saddle_point_output(
                    scaled_problem,
                    avg_primal_solution,
                    avg_dual_solution,
                    termination_reason,
                    iteration - 1,
                    iteration_stats,
                )
            end

            buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product

            prev_last_restart_info = deepcopy(last_restart_info)

            current_iteration_stats.restart_used = run_restart_scheme(
                d_scaled_problem.scaled_qp,
                solver_state.solution_weighted_avg,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                last_restart_info,
                iteration - 1,
                primal_norm_params,
                dual_norm_params,
                solver_state.primal_weight,
                params.verbosity,
                params.restart_params,
                solver_state.current_primal_product,
                solver_state.current_dual_product,
                buffer_avg,
                buffer_kkt,
                buffer_primal_gradient,
                #Mine
                qp_cache,
                iteration, # PDGH iteration
                params.termination_criteria,
                ir_instead_average,    # IR instead of restart
                ir_type,
            )

            # if last_restart_info.ir_refinement_applied
            #     if last_restart_info.ir_refinements_number - prev_last_restart_info.ir_refinements_number == 1
            #         println("FIRST cuPDLP output after IR:")
            #         # println("Iter "*string(iteration))
            #         println("norm of primal sol:")
            #         println(CUDA.norm(last_restart_info.primal_solution))
            #         println("norm of dual sol:")
            #         println(CUDA.norm(last_restart_info.dual_solution))
            #         println("KKT values from last_restart_info:")
            #         println(last_restart_info.last_restart_kkt_residual)  
            #     end
            # end

            if current_iteration_stats.restart_used != RESTART_CHOICE_NO_RESTART
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )
                solver_state.ratio_step_sizes = 1.0
            end
        end

        time_spent_doing_basic_algorithm_checkpoint = time()

        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhg_specific_log(
                # problem,
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end

        if !last_restart_info.ir_refinement_applied || last_restart_info.ir_refinements_number - prev_last_restart_info.ir_refinements_number != 1
            take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)
        end

        # if last_restart_info.ir_refinement_applied
        #     if last_restart_info.ir_refinements_number - prev_last_restart_info.ir_refinements_number == 1
        #         println("FOURTH cuPDLP output after IR:")
        #         # println("Iter "*string(iteration))
        #         println("norm of primal sol:")
        #         println(CUDA.norm(solver_state.current_primal_solution))
        #         println("norm of dual sol:")
        #         println(CUDA.norm(solver_state.current_dual_solution))
        #         # println("KKT values from last_restart_info:")
        #         # println(last_restart_info.last_restart_kkt_residual)  
        #     end
        # end

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end
