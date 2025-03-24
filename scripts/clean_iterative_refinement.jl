include("/nfs/home2/nacevedo/RA/cuPDLP.jl/src/cuPDLP.jl")


# Util constructors
struct ProblemParams
    constraint_matrix
    right_hand_side
    objective_vector 
    variable_upper_bound 
    variable_lower_bound 
end

struct IRSolution
    current_primal_solution
    current_dual_solution 
    primal_product 
    dual_product
    primal_gradient
end


# Util functions 

# 1. General problem to "quasi standard form" (no inequalities)
function LP_to_quasi_standard_form(lp::Union{QuadraticProgrammingProblem, CuLinearProgrammingProblem})
    # PDLP format (Scaled problem)
    # min c'x
    #           A = [AE; AI]
    #           b = [bE; bI]
    # AE*x = bE
    # AI*x >= bI -> AI*x - slacks = bI  
    # l <= x <= u
    A = lp.constraint_matrix # sparse format matrix
    # You can write A = [AE 0; AI -I], with I of slacks
    n_inequalities = size(A, 1) - lp.num_equalities # num of slacks to add
    if n_inequalities > 0 
        # Identify matrix to add slack variables
        I = sparse(LinearAlgebra.I, n_inequalities, n_inequalities)

        # Add n_eq columns and rows of zeros to I
        Z_I = [
            spzeros(lp.num_equalities, n_inequalities); 
            -I
            ]
        A = [A Z_I] #hcat(A, I)

        # Add slack variables to the objective function
        c = lp.objective_vector
        c = [c; spzeros(n_inequalities)]

        # Add slack variables to the upper bound
        u = lp.variable_upper_bound
        u = [u; Inf*(spzeros(n_inequalities).+1)]

        # Add slack variables to the lower bound
        l = lp.variable_lower_bound
        l = [l; spzeros(n_inequalities)]

        # Update the LP
        lp.constraint_matrix = A
        lp.objective_vector = c
        lp.variable_upper_bound = u
        lp.variable_lower_bound = l
        lp.num_equalities = size(A, 1)
        lp.objective_matrix =  sparse(Int64[], Int64[], Float64[], size(c, 1), size(c, 1))
    end
    return lp
end
        

function compute_current_KKT(
    lp::Union{QuadraticProgrammingProblem, CuLinearProgrammingProblem},
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
)
    # Preliminars
    buffer_kkt = cuPDLP_KKT_buffer(lp, current_primal_solution, current_dual_solution)
    qp_cache = cuPDLP.cached_quadratic_program_info(lp) # As in "optimize" function (line 462)
    cuLPP = cuPDLP.qp_cpu_to_gpu(lp)
    convergence_info = cuPDLP.compute_convergence_information(
        cuLPP, #        problem::CuLinearProgrammingProblem,
        qp_cache,#      qp_cache::CachedQuadraticProgramInfo,
        CuVector{Float64}(current_primal_solution),#    primal_iterate::CuVector{Float64},
        CuVector{Float64}(current_dual_solution),#    dual_iterate::CuVector{Float64},
        1.0, #          eps_ratio::Float64,
        cuPDLP.POINT_TYPE_AVERAGE_ITERATE,# candidate_type::PointType,
        CuVector{Float64}(lp.constraint_matrix*current_primal_solution),#      primal_product::CuVector{Float64},
        CuVector{Float64}(lp.objective_vector - lp.constraint_matrix'*current_dual_solution),#  primal_gradient::CuVector{Float64},
        buffer_kkt  #   buffer_kkt::BufferKKTState,
    )

    # KKT computation 
    current_kkt = norm([
        convergence_info.relative_l_inf_primal_residual,
        convergence_info.relative_l_inf_dual_residual,
        convergence_info.relative_optimality_gap
    ],1)

    return current_kkt
end


# MINE: restart input paraters
# problem::CuLinearProgrammingProblem,          
# solution_weighted_avg::CuSolutionWeightedAverage,
# current_primal_solution::CuVector{Float64},
# current_dual_solution::CuVector{Float64},
# last_restart_info::CuRestartInfo,
# iterations_completed::Int64,
# primal_norm_params::Float64,
# dual_norm_params::Float64,
# primal_weight::Float64,
# verbosity::Int64,
# restart_params::RestartParameters,
# primal_product::CuVector{Float64},
# dual_product::CuVector{Float64},
# buffer_avg::CuBufferAvgState,
# buffer_kkt::BufferKKTState,
# buffer_primal_gradient::CuVector{Float64},


function call_cuPDLP(
    lp::Union{QuadraticProgrammingProblem, CuLinearProgrammingProblem},
    tolerance::Float64,
    time_sec_limit::Union{Float64, Int64},
    iteration_limit::Union{Float64, Int64}
)

    # Initialization
    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(lp);
    redirect_stdout(oldstd)

    # Restart criteria (same as original)
    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    # Termination criteria parameters
    termination_params = cuPDLP.construct_termination_criteria(
        optimality_norm = cuPDLP.L_INF, # L2 , L_INF
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1e-8, # This is not primal infeasibility tolerance
        eps_dual_infeasible = 1e-8, # This is not dual infeasibility tolerance
        time_sec_limit = time_sec_limit,
        iteration_limit = iteration_limit, #typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    # cuPDLP parameters 
    params = cuPDLP.PdhgParameters(
        10,             # l_inf_ruiz_iterations::Int
        false,          # l2_norm_rescaling::Bool
        1.0,            # pock_chambolle_alpha::Union{Float64,Nothing}
        1.0,            # primal_importance::Float64
        true,           # scale_invariant_initial_primal_weight::Bool
        2,              # verbosity::Int64
        true,           # record_iteration_stats::Bool
        64,              # termination_evaluation_frequency::Int32 (Default: 64) (Mine: 1)
        termination_params, # termination_criteria::TerminationCriteria
        restart_params,     # restart_params::RestartParameters
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6), # step_size_policy_params::Union{
                                                #     AdaptiveStepsizeParams,
                                                #     ConstantStepsizeParams,
                                                # }
    )

    # Optimize
    output = cuPDLP.optimize(
        params,
        lp
        )
    
    return output
end


# Input:    Violations
# Output:   New solution from scaled/shifted problem
function scalar_refinement(
    lp::Union{QuadraticProgrammingProblem, CuLinearProgrammingProblem},
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    # problem_parameters::ProblemParams
    # shifted_parameters::ProblemParamse
    tolerance::Float64
    )

    # shifted parameters
    b_bar = lp.right_hand_side - lp.constraint_matrix * current_primal_solution
    l_bar = lp.variable_lower_bound - current_primal_solution
    u_bar = lp.variable_upper_bound - current_primal_solution
    c_bar = lp.objective_vector - lp.constraint_matrix' * current_dual_solution
    
    # Maximum primal/dual violation
    delta_P = maximum([
        maximum(abs.(b_bar)),
        maximum(l_bar),
        maximum(-u_bar)
    ])
    delta_D = maximum([
        c_bar[current_primal_solution .> (lp.variable_lower_bound + lp.variable_upper_bound) / 2] ; 
        -c_bar[current_primal_solution .<= (lp.variable_lower_bound + lp.variable_upper_bound) / 2]
        ], 
        init=0 # at least 0
    )

    # Scaling factor
    Delta_P = min( 1 / delta_P , alpha*Delta_P ) 
    Delta_D = min( 1 / delta_D , alpha*Delta_D ) 

    # Solve scaled-shifted problem
    scaled_lp = cuPDLP.QuadraticProgrammingProblem(
        l_bar,
        u_bar,
        lp.objective_matrix,
        c_bar,
        0,
        A,
        b_bar,
        lp.num_equalities
    )

    # Call cuPDLP to optimize new problem
    output = call_cuPDLP(
        scaled_lp,
        tolerance,
        time_sec_limit,
        iteration_limit
    )
    
    # New solutions
    x_k = x_k + output.primal_solution / Delta_P
    y_k = y_k + output.dual_solution / Delta_D
    
    return x_k, y_k
end


function iterative_refinement(
    problem::CuLinearProgrammingProblem,            # Potentially: Scaled problem of cuPDLP
    current_primal_solution::CuVector{Float64},       
    current_dual_solution::CuVector{Float64},
    # My inputs
    tolerance::Float64,
    ir_tolerance_factor::Float64,
    scaling_type::String,
    scaling_name::String=="scalar"
    )

    # Problem to quadratic form 
    lp = LP_to_quasi_standard_form(problem)

    # Initial solution
    slack_solution = CuVector{Float64}(problem.constraint_matrix[problem.num_equalities+1:end,:]*current_primal_solution - problem.right_hand_side[problem.num_equalities+1:end])  # s = AIx - bI
    x_k = [current_primal_solution; slack_solution] # Concatenate (x,s)
    y_k = current_dual_solution

    # Current KKT 
    current_kkt = compute_current_KKT(
    lp,
    x_k,
    y_k
    )

    # Problem parameters
    A = lp.constraint_matrix
    b = lp.right_hand_side
    c = lp.objective_vector
    l = lp.variable_lower_bound
    u = lp.variable_upper_bound

    # Initialization parameters 
    if scaling_type == "scalar"
        Delta_P = 1
        Delta_D = 1
    elseif scaling_type == "matrix"
        Delta_P = ones(length(b)) 
        Delta_D = ones(length(c))
    end

    # Iterations of IR
    k = 0
    while true
        k+=1

        # Refined solution
        if scaling_type == "scalar"
            x_k, y_k = scalar_refinement(
                lp,
                x_k,
                y_k,
                10^(-ceil(-log10(current_kkt))) * ir_tolerance_factor,
                )
        elseif scaling_type == "matrix"
            # matrix_refinemet()
            continue
        end


        current_kkt = compute_current_KKT(
            lp,
            x_k,
            y_k
        )

        # Termination criteria 
        if current_kkt <= tolerance
            new_primal_solution = x_k[1:problem.num_equalities]
            new_dual_solution = y_k
            new_primal_product = problem.constraint_matrix * new_primal_solution
            new_primal_gradient = problem.constraint_matrix'*new_dual_solution
            new_dual_product = (problem.objective_vector - new_primal_gradient)

            return IRSolution(
                new_primal_solution,
                new_dual_solution,
                new_primal_product,
                new_dual_product,
                new_primal_gradient
            )

        end
    end
end