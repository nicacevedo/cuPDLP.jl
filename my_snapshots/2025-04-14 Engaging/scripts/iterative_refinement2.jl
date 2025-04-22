# import ArgParse
import GZip
import JSON3

# import cuPDLP

include("/nfs/home2/nacevedo/RA/cuPDLP.jl/src/cuPDLP.jl")
# folder_path = "/nfs/home2/nacevedo/RA/cuPDLP.jl/src/"
# for file in readdir(folder_path)
#     if endswith(file, ".jl")
#         include(joinpath(folder_path, file))
#     end
# end
# include("../src/primal_dual_hybrid_gradient_gpu.jl")

# My imports
# import Formatting
using LinearAlgebra
using SparseArrays
using Statistics 
# using Arpack # for large sparse matrices max singular values (requires installation)
# using PROPACK # for large sparse matrices max/min singular values (requires installation)

using CUDA

@enum OptimalityNorm L_INF L2 # Mine


function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    parameters::cuPDLP.PdhgParameters,
    output_dir::String,
    instance_path::String,
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
  
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
  
    function inner_solve()
        lower_file_name = lowercase(basename(instance_path))
        if endswith(lower_file_name, ".mps") ||
            endswith(lower_file_name, ".mps.gz") ||
            endswith(lower_file_name, ".qps") ||
            endswith(lower_file_name, ".qps.gz")
            lp = cuPDLP.qps_reader_to_standard_form(instance_path)
        else
            error(
                "Instance has unrecognized file extension: ", 
                basename(instance_path),
            )
        end
    
        if parameters.verbosity >= 1
            println("Instance: ", instance_name)
        end

        output::cuPDLP.SaddlePointOutput = cuPDLP.optimize(parameters, lp)
    
        log = cuPDLP.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = cuPDLP.POINT_TYPE_AVERAGE_ITERATE
    
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)
    
        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end     

    inner_solve()
   
    return
end

function warm_up(lp::cuPDLP.QuadraticProgrammingProblem)
    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = cuPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-4,
        eps_optimal_relative = 1.0e-4,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = Inf,
        iteration_limit = 100,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = cuPDLP.PdhgParameters(
        10,     #   l_inf_ruiz_iterations::Int
        false,  #   l2_norm_rescaling::Bool
        1.0,    #   pock_chambolle_alpha::Union{Float64,Nothing}
        1.0,    #   primal_importance::Float64
        true,   #   scale_invariant_initial_primal_weight::Bool
        0,      #   verbosity::Int64
        true,   #   record_iteration_stats::Bool
        64,     #   termination_evaluation_frequency::Int32 
        termination_params_warmup,  #   termination_criteria::TerminationCriteria
        restart_params,             #   restart_params::RestartParameters      
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6), #   step_size_policy_params::Union{
                                                #        AdaptiveStepsizeParams,
                                                #        ConstantStepsizeParams,
                                                #    }
    )


    
    
    
    
    
    
    
    
    
    
    

    cuPDLP.optimize(params_warmup, lp);
end


# function parse_command_line()
#     arg_parse = ArgParse.ArgParseSettings()

#     ArgParse.@add_arg_table! arg_parse begin
#         "--instance_path"
#         help = "The path to the instance to solve in .mps.gz or .mps format."
#         arg_type = String
#         required = truea

#         "--output_directory"
#         help = "The directory for output files."
#         arg_type = String
#         required = true

#         "--iter_tolerance"
#         help = "IR iters KKT tolerance of the solution."
#         arg_type = Float64
#         default = 1e-3

#         "--obj_tolerance"
#         help = "Final KKT tolerance of the solution."
#         arg_type = Float64
#         default = 1e-8

#         "--time_sec_limit"
#         help = "Time limit."
#         arg_type = Float64
#         default = 3600.0

#         "--max_iter"
#         help = "Maximum of desired iterations in IR"
#         arg_type = Int64
#         default = 10
#     end

#     return ArgParse.parse_args(arg_parse)
# end






# MY CODE

# My functions
function LP_to_quasi_standard_form(lp::cuPDLP.QuadraticProgrammingProblem)
    # Change LP to standard for
    # Available atrributes of lp.*:
        # qp.variable_lower_bound,
        # qp.variable_upper_bound,
        # qp.objective_matrix,
        # qp.objective_vector,
        # qp.objective_offset,
        # qp.constraint_matrix,
        # right_hand_side,
        # num_equalities,

    # PDLP format
    # min c'x
    #           A = [AE; AI]
    #           b = [bE; bI]
    # AE*x = bE
    # AI*x >= bI -> AI*x - slacks = bI  
    # l <= x <= u
    A = lp.constraint_matrix # sparse format matrix
    # println(" objective matrix: ", lp.objective_matrix)
    # println("A nonsparse: ", Matrix(A))
    # # println("sparse A: ", A)
    # println("num_eq: ", lp.num_equalities)
    # println("shape of sparse A: ", size(A))

    # You can write A = [AE 0; AI -I], with I of slacks
    n_inequalities = size(A, 1) - lp.num_equalities # num of slacks to add
    if n_inequalities > 0 
        # Identify matrix to add slack variables
        I = sparse(LinearAlgebra.I, n_inequalities, n_inequalities)
        # I = Matrix{Float64}

        # Add n_eq columns and rows of zeros to I
        Z_I = [
            spzeros(lp.num_equalities, n_inequalities); 
            -I
            ]
        A = [A Z_I] #hcat(A, I)

        # Add slack variables to the objective function
        c = lp.objective_vector
        c = [c; spzeros(n_inequalities)]
        # c = vcat(c, zeros(n_inequalities))

        # Add slack variables to the upper bound
        u = lp.variable_upper_bound
        u = [u; Inf*(spzeros(n_inequalities).+1)]
        # u = vcat(u, Inf*ones(n_inequalities))

        # Add slack variables to the lower bound
        l = lp.variable_lower_bound
        l = [l; spzeros(n_inequalities)]
        # l = vcat(l, zeros(n_inequalities))

        # Update the LP
        lp.constraint_matrix = A
        lp.objective_vector = c
        lp.variable_upper_bound = u
        lp.variable_lower_bound = l
        lp.num_equalities = size(A, 1)
        lp.objective_matrix =  sparse(Int64[], Int64[], Float64[], size(c, 1), size(c, 1))
        # println("lp after adding slacks: ", lp)

    end
    return lp
end
                            

function call_pdlp(lp, tolerance, time_sec_limit, save_log::Bool, out_dict::Dict, iteration_limit)
    
    # Change LP to standard form
    println("LP to standard form...")
    lp = LP_to_quasi_standard_form(lp)
    println("LP to standard form done")

    # Wee need CUDA from here
    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(lp);
    redirect_stdout(oldstd)

    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuPDLP.construct_termination_criteria(
        optimality_norm = cuPDLP.L_INF, # L2 , L_INF
        eps_optimal_absolute = tolerance,#1e-6,
        eps_optimal_relative = tolerance,#1e-6,
        eps_primal_infeasible = 1e-8, # This is not primal infeasibility tolerance
        eps_dual_infeasible = 1e-8, # This is not dual infeasibility tolerance
        time_sec_limit = time_sec_limit,
        iteration_limit = iteration_limit, #typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )


 
    params = cuPDLP.PdhgParameters(
        10,             # l_inf_ruiz_iterations::Int
        false,          # l2_norm_rescaling::Bool
        1.0,            # pock_chambolle_alpha::Union{Float64,Nothing}
        1.0,            # primal_importance::Float64
        true,           # scale_invariant_initial_primal_weight::Bool
        2,              # verbosity::Int64
        true,           # record_iteration_stats::Bool
        1,              # termination_evaluation_frequency::Int32 (Default: 64)
        termination_params, # termination_criteria::TerminationCriteria
        restart_params,     # restart_params::RestartParameters
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6), # step_size_policy_params::Union{
                                                #     AdaptiveStepsizeParams,
                                                #     ConstantStepsizeParams,
                                                # }
    )

    output = cuPDLP.optimize(
    params,
    lp
    )



    # Save output as in solve.jl
    if save_log
        instance_name = out_dict["instance_name"]
        output_dir = out_dict["output_dir"]
        current_iter_k = out_dict["current_iter_k"]

        # Check if the output directory exists, and create it if not
        if !isdir(output_dir)
            mkdir(output_dir)
        end
            
        # Save the logs (originally from solve.jl)
        log = cuPDLP.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = cuPDLP.POINT_TYPE_AVERAGE_ITERATE

        print("Saving log from call_pdlp...")

        instance_name = replace(instance_name, ".mps" => "")

        # println(instance_name * "_k" * string(current_iter_k) * "_summary.json")
        summary_output_path = joinpath(output_dir, instance_name * "_k" * string(current_iter_k) * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end

        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_k" * string(current_iter_k) * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end

        primal_output_path = joinpath(output_dir, instance_name  * "_k" * string(current_iter_k) * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)

        dual_output_path = joinpath(output_dir, instance_name  * "_k" * string(current_iter_k) * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)

        print("Successfuly saved files from call_pdlp")

    end # save_log end


    return params, lp, output
end


function cuPDLP_KKT_buffer(lp_0_stand, x_k, y_k)

    # Calculate the KKT error of the problem
    primal_size = size(x_k)[1]
    dual_size = size(y_k)[1]
    num_eq = lp_0_stand.num_equalities

    # Buffer initialized with zeros (as in cuPDLP code)
    buffer_original = cuPDLP.BufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
    )

    # Buffer state also initialized in zero (as in cuPDLP code)
    buffer_kkt = cuPDLP.BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        cuPDLP.CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )

    return buffer_kkt
end


function cuPDLP_from_initial_sol(
    lp, tolerance, time_sec_limit, save_log, instance_name, output_path,
    x_k, y_k, A,b,l,u,c
    )

    #     # Unscale iterates.
#     buffer_original.original_primal_solution .=
#     primal_solution ./ scaled_problem.variable_rescaling
# buffer_original.original_primal_gradient .=
#     primal_gradient .* scaled_problem.variable_rescaling
# buffer_original.original_dual_solution .=
#     dual_solution ./ scaled_problem.constraint_rescaling
# buffer_original.original_primal_product .=
#     primal_product .* scaled_problem.constraint_rescaling


    # 1. My own initial state
    println("Generating intial state from a previous solution...")
    primal_size = length(x_k)
    dual_size = length(y_k)
    solver_state = cuPDLP.CuPdhgSolverState(
        CuVector{Float64}(x_k), #   current_primal_solution::CuVector{Float64}
        CuVector{Float64}(y_k), #   current_dual_solution::CuVector{Float64}
        CuVector{Float64}(A*x_k),       #   current_primal_product::CuVector{Float64}
        CuVector{Float64}(A'y_k),    #   current_dual_product::CuVector{Float64}
        cuPDLP.initialize_solution_weighted_average(primal_size, dual_size),   #   solution_weighted_avg::CuSolutionWeightedAverage 
        0.0,        #   step_size::Float64
        1.0,        #   primal_weight::Float64
        false,      #   numerical_error::Bool
        0.0,        #   cumulative_kkt_passes::Float64
        0,          #   total_number_iterations::Int64
        nothing,    #   required_ratio::Union{Float64,Nothing}
        nothing,    #   ratio_step_sizes::Union{Float64,Nothing}
    )
    # also check: update_solution_in_solver_state! in pdhg_gpu.jl

    # DELTAS
    # delta_primal[tx] = current_primal_solution[tx] - (step_size / primal_weight) * (objective_vector[tx] - current_dual_product[tx])
    # delta_primal[tx] = min(variable_upper_bound[tx], max(variable_lower_bound[tx], delta_primal[tx]))
    # delta_primal[tx] -= current_primal_solution[tx]

    # delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])
    # delta_dual[tx] -= current_dual_solution[tx]


    # c = lp.objective_vector
    # A = lp.constraint_matrix
    # b = lp.right_hand_side
    # l = lp.variable_lower_bound
    # u = lp.variable_upper_bound

    # 2. My own buffer
    buffer_state = cuPDLP.CuBufferState(
        CUDA.zeros(Float64, primal_size),      # delta_primal
        CUDA.zeros(Float64, dual_size),        # delta_dual
        CUDA.zeros(Float64, dual_size),        # delta_primal_product
    )

    buffer_avg = cuPDLP.CuBufferAvgState(
        CuVector{Float64}(x_k),      # avg_primal_solution
        CuVector{Float64}(y_k),        # avg_dual_solution
        CuVector{Float64}(A*x_k),        # avg_primal_product
        CuVector{Float64}(c - A'y_k),      # avg_primal_gradient
    )

    buffer_original = cuPDLP.BufferOriginalSol(
        CuVector{Float64}(x_k),      # primal
        CuVector{Float64}(y_k),        # dual
        CuVector{Float64}(A*x_k),        # primal_product
        CuVector{Float64}(c - A'y_k),      # primal_gradient
    )
    

    # VIOLATIONS

#     @inbounds begin
#         constraint_violation[tx] = right_hand_side[tx] - activities[tx] #(activities = A*x)
#     end
# end
# if num_equalities + 1 <= tx <= num_constraints
#     @inbounds begin
#         constraint_violation[tx] = max(right_hand_side[tx] - activities[tx], 0.0)

    # @inbounds begin
    #     lower_variable_violation[tx] = max(variable_lower_bound[tx] - primal_vec[tx], 0.0)
    #     upper_variable_violation[tx] = max(primal_vec[tx] - variable_upper_bound[tx], 0.0)


    # DUAL OBJECTIVE CONTRIBUTION
    # @inbounds begin
    #     if reduced_costs[tx] > 0.0
    #         dual_objective_contribution_array[tx] = variable_lower_bound[tx] * reduced_costs[tx]
    #     elseif reduced_costs[tx] < 0.0
    #         dual_objective_contribution_array[tx] = variable_upper_bound[tx] * reduced_costs[tx]
    #     else
    #         dual_objective_contribution_array[tx] = 0.0

    # COMPUTE DUAL STATS
    # CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockIneq compute_dual_residual_kernel!(
    #     buffer_kkt.dual_solution,
    #     problem.num_equalities,
    #     problem.num_constraints - problem.num_equalities,
    #     buffer_kkt.dual_stats.dual_residual,
    # )  
    # buffer_kkt.dual_stats


    buffer_kkt = cuPDLP.BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CuVector{Float64}(max.(l - x_k, 0)),      # lower_variable_violation
        CuVector{Float64}(max.(x_k - u, 0)),      # upper_variable_violation
        CuVector{Float64}(b - A*x_k),        # constraint_violation (see above)
        CuVector{Float64}(A'*y_k),      # dual_objective_contribution_array [this is just inference]
        CuVector{Float64}(c-A'*y_k),      # reduced_costs_violations [this is not so clear]
        cuPDLP.CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - lp.num_equalities),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    buffer_kkt_infeas = cuPDLP.BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CuVector{Float64}(max.(l - x_k, 0)),      # lower_variable_violation
        CuVector{Float64}(max.(x_k - u, 0)),      # upper_variable_violation
        CuVector{Float64}(b - A*x_k),        # constraint_violation (see above)
        CuVector{Float64}(A'*y_k),      # dual_objective_contribution_array [this is just inference]
        CuVector{Float64}(c-A'*y_k),      # reduced_costs_violations [this is not so clear]
        cuPDLP.CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - lp.num_equalities),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )

    buffer_primal_gradient = buffer_original.original_primal_gradient#CUDA.zeros(Float64, primal_size)
    # buffer_primal_gradient .= d_scaled_problem.scaled_qp.objective_vector .- solver_state.current_dual_product


    # KKT RELATED BUT MAY NOT BE NECESSARY
    qp_cache = cuPDLP.cached_quadratic_program_info(lp) # As in "optimize" function (line 462)
    cuLPP = cuPDLP.qp_cpu_to_gpu(lp)

    # OPTIMIZATION

    # Wee need CUDA from here
    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(lp);
    redirect_stdout(oldstd)

    restart_params = cuPDLP.construct_restart_parameters(
        cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
        cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuPDLP.construct_termination_criteria(
        optimality_norm = cuPDLP.L_INF, # L2 , L_INF
        eps_optimal_absolute = tolerance,#1e-6,
        eps_optimal_relative = tolerance,#1e-6,
        eps_primal_infeasible = 1e-8, # This is not primal infeasibility tolerance
        eps_dual_infeasible = 1e-8, # This is not dual infeasibility tolerance
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )


 
    params = cuPDLP.PdhgParameters(
        10,             # l_inf_ruiz_iterations::Int
        false,          # l2_norm_rescaling::Bool
        1.0,            # pock_chambolle_alpha::Union{Float64,Nothing}
        1.0,            # primal_importance::Float64
        true,           # scale_invariant_initial_primal_weight::Bool
        2,              # verbosity::Int64
        true,           # record_iteration_stats::Bool
        1,              # termination_evaluation_frequency::Int32 (Default: 64)
        termination_params, # termination_criteria::TerminationCriteria
        restart_params,     # restart_params::RestartParameters
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6), # step_size_policy_params::Union{
                                                #     AdaptiveStepsizeParams,
                                                #     ConstantStepsizeParams,
                                                # }
    )

    println("Optimizing from initial solver state...")
    output = cuPDLP.optimize(
    params,
    lp,
    (x_k, y_k), # primal_dual_solution
    (A,b,c,l,u) # problem_params
    )

    #     # Unscale iterates.
    #     buffer_original.original_primal_solution .=
    #     primal_solution ./ scaled_problem.variable_rescaling
    # buffer_original.original_primal_gradient .=
    #     primal_gradient .* scaled_problem.variable_rescaling
    # buffer_original.original_dual_solution .=
    #     dual_solution ./ scaled_problem.constraint_rescaling
    # buffer_original.original_primal_product .=
    #     primal_product .* scaled_problem.constraint_rescaling


    save_log = false   
    if save_log
        println("SAVING LOG...")
        # instance_name = out_dict["instance_name"]
        output_dir = output_path * "/subPDLP"
        # current_iter_k = out_dict["current_iter_k"]

        # Check if the output directory exists, and create it if not
        if !isdir(output_dir)
            mkdir(output_dir)
        end

        # Save the logs (originally from solve.jl)
        log = cuPDLP.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = cuPDLP.POINT_TYPE_AVERAGE_ITERATE

        print("Saving log from cuPDLP_from_initial_sol...")

        instance_name = replace(instance_name, ".mps" => "")

        # println(instance_name * "_k" * string(current_iter_k) * "_summary.json")
        summary_output_path = joinpath(output_dir, instance_name * "_k" * string(current_iter_k) * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end

        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_k" * string(current_iter_k) * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end

        primal_output_path = joinpath(output_dir, instance_name  * "_k" * string(current_iter_k) * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)

        dual_output_path = joinpath(output_dir, instance_name  * "_k" * string(current_iter_k) * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)

        print("Successfuly saved files from cuPDLP_from_initial_sol")
    end

    return output

end



# New version (2025)
function iterative_refinement(
    instance_path::String,
    output_path::String,
    iterative_tol = 1e-3,
    objective_tol=1e-8,
    time_sec_limit=300,
    max_iter =100,
    alpha=1.1,
    save_log=false, # save the log on each iteration
    no_alpha=false, # no alpha scaling
    scaling_bound=1e50,
    tol_decrease_factor=0.1, # min(KKT*decrease factor, last_tol*decrease_factor)
    pdlp_iteration_limit=1000,
)


    # Instance name
    instance_name = replace(split(instance_path, "/")[end], ".mps"=>"")
    instance_name = replace(split(instance_path, "/")[end], ".gz"=>"")

    # Output metrics
    out = Dict(
        "instance" => instance_path,
        "iterative_tol" => iterative_tol,
        "objective_tol" => objective_tol,
        "candidate_type" => cuPDLP.POINT_TYPE_AVERAGE_ITERATE,
        "termination_reason" => cuPDLP.TerminationReason[],
        "alpha" => alpha,
        "primal_objective" => Float64[],
        "dual_objective" => Float64[],
        # "l_inf_primal_residual" => Float64[],
        # "l_inf_dual_residual" => Float64[],
        "relative_l_inf_primal_residual" => Float64[], 
        "l_inf_primal_residual" => Float64[],  # (1 + norm) correction (this need 1 more transformation)
        "relative_l_inf_dual_residual" => Float64[],
        "l_inf_dual_residual" => Float64[], # (1 + norm) correction
        "relative_optimality_gap" => Float64[], # There is no "l_inf" or whatsoever
        "optimality_gap" => Float64[], # (1 + norm) correction
        "Delta_P" => Float64[],
        "Delta_D" => Float64[],
        "time_sec_limit" => time_sec_limit,
        "max_iter" => max_iter,
        "last_iteration" => 0,
        "blackbox_time" => Float64[],
        "total_time" => 0.0,
        "A_condition_number" => 0.0,
        # "A_spectral_norm" => 0.0,
        "P_condition_number" => 0.0,
        # "P_spectral_norm"=> 0.0,
        "iterative_tol_k"=>Float64[iterative_tol]
    )

    # Initial parameters
    Delta_P = 1
    Delta_D = 1

    # Read the instance from the path and change it to standard form
    println("Trying to read instance from ", instance_path)
    lp_0 = cuPDLP.qps_reader_to_standard_form(instance_path)
    println("Instance read successfully")


    # Initial iteration of the blackbox PDLP algorithm
    println("Initial iteration of the blackbox PDLP algorithm")
    println("iterative tol: ", iterative_tol)
    println("time sec limit: ", time_sec_limit)
    println("save log: ", save_log)
    remaining_time = time_sec_limit
    total_time = time()
    t_start_k = time()
    params, lp_0_stand, output = call_pdlp(
        lp_0,           # LP in standard form
        iterative_tol,  # IR tolerance
        time_sec_limit, # Time limit for PDLP
        save_log,       # Save logs or not on each iteration of IR
        Dict(
        "instance_name" => instance_name, # or some variation of name
        "output_dir" => output_path,
        "current_iter_k" => 0,
        ),               # Output dictionary, for saving logs
        (objective_tol == iterative_tol) * typemax(Int32) + (objective_tol < iterative_tol) * pdlp_iteration_limit # iteration_limit: First call of PDLP with a limit, for hard-to-converge instances. KKT_0
    )
    lp = lp_0_stand
    t_pdlp_0 = time() - t_start_k

    # Initial optimization results ("approximate primal–dual solution")
    x_k = output.primal_solution
    y_k = output.dual_solution

    println("primal solution size: ", size(x_k))    
    println("  dual solution size: ", size(y_k))

    # Original parameters of the problem
    c = lp.objective_vector
    A = lp.constraint_matrix
    b = lp.right_hand_side
    l = lp.variable_lower_bound
    u = lp.variable_upper_bound
    # optimal_primal_cost = sum(c.*x_k)
    # optimal_dual_cost = sum(b.*y_k)

    ##################################################################################
    # NEW: Compute the condition number of A, and also of [A b; c' 0], to check the
    # illness of the problem
    ##################################################################################
    println("Computing condition # of A and P...")
    # 1. Condition number of A
    # 1.1 Some approximation
    # sigma_max, bnd, nprod, ntprod = tsvdvals(A, k=1) # Sparse Sing. Values Computation
    # sigma_min, bnd, nprod, ntprod = tsvdvals_irl(A, k=1) # Sparse Sing. Values Computation
    # "Infinity norm":
    # A_norms = sum(abs.(A) , dims = 2)
    sigma_max = maximum(sum(abs.(A) , dims = 2))
    out["A_condition_number"] = sigma_max/minimum(sum(abs.(A) , dims = 2))
    println("Abs ratio of A (cond. # approx.): ", out["A_condition_number"])
    # # 1.2 Exact calculation
    # singular_values = svdvals(Matrix(A))
    # sigma_max = maximum(singular_values)
    # sigma_min = minimum(singular_values[singular_values .> 0])
    # # out["A_condition_number"] = sigma_max / sigma_min 
    # # out["A_spectral_norm"] = sigma_max
    # println("Cond. # of A done: ", sigma_max / sigma_min )

    # 2. Condition number of the problem
    # 2.1 Some approximation
    # sigma_max, bnd, nprod, ntprod =  tsvdvals([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0], k=1) # Sparse Sing. Values Computation
    # sigma_max, bnd, nprod, ntprod =  tsvdvals_irl([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0], k=1) # Sparse Sing. Values Computation
    # "Infinity norm":
    # P_norms = sum(abs.([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]) , dims = 2)
    # out["P_condition_number"] = maximum(sum(abs.([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]) , dims = 2))/minimum(sum(abs.([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]) , dims = 2))
    out["P_condition_number"] = maximum(sum(abs.([A b; c' 0]) , dims = 2))/minimum(sum(abs.([A b; c' 0]) , dims = 2))
    println("Abs ratio of P (cond. # approx.): ", out["P_condition_number"])
    # sigma_max = maximum(sum(abs.(A) , dims = 2))
    # println("Abs ratio of P (v2) (cond. # approx.): ", (sigma_max + norm(b) + norm(c))/min(sigma_max, norm(b), norm(c)))
    # # # 2.2 Exact calculation
    # # sigma_max = maximum(singular_values)
    # singular_values = svdvals(Matrix([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]))
    # sigma_max = maximum(singular_values)
    # sigma_min = minimum(singular_values[singular_values .> 0])
    # # out["P_condition_number"] = sigma_max/sigma_min
    # # out["P_spectral_norm"] = sigma_max
    # println("Cond. # of P done: ", sigma_max/sigma_min)
    if alpha == 1024
        alpha = out["P_condition_number"]
        out["alpha"] = alpha
    end
    ##################################################################################
    
    # Compute the KKT error
    buffer_kkt = cuPDLP_KKT_buffer(lp_0_stand, x_k, y_k)
    qp_cache = cuPDLP.cached_quadratic_program_info(lp_0_stand) # As in "optimize" function (line 462)
    cuLPP = cuPDLP.qp_cpu_to_gpu(lp_0_stand)
    convergence_info = cuPDLP.compute_convergence_information(
        cuLPP, #        problem::CuLinearProgrammingProblem,
        qp_cache,#      qp_cache::CachedQuadraticProgramInfo,
        CuVector{Float64}(x_k),#    primal_iterate::CuVector{Float64},
        CuVector{Float64}(y_k),#    dual_iterate::CuVector{Float64},
        1.0, #          eps_ratio::Float64,
        cuPDLP.POINT_TYPE_AVERAGE_ITERATE,# candidate_type::PointType,
        CuVector{Float64}(A*x_k),#      primal_product::CuVector{Float64},
        CuVector{Float64}(c - A'y_k),#  primal_gradient::CuVector{Float64},
        buffer_kkt  #   buffer_kkt::BufferKKTState,
    )


    # Update the output metrics
    push!(out["termination_reason"], output.termination_reason)
    push!(out["blackbox_time"], t_pdlp_0)
    push!(out["primal_objective"], convergence_info.primal_objective)
    push!(out["dual_objective"], convergence_info.dual_objective)
    push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
    push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
    push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
    push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
    push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
    push!(out["optimality_gap"], abs(convergence_info.primal_objective - convergence_info.dual_objective))
    # Scalar version
    push!(out["Delta_P"], Delta_P) # careful with the scalar version meaning 
    push!(out["Delta_D"], Delta_D) # careful with the scalar version meaning

    # println("Termination condition:")
    # println("dir. dual: ", convergence_info.l_inf_dual_residual)
    # println("rel. dual: ", convergence_info.relative_l_inf_dual_residual * (1 + norm(c, Inf)))
    # println("dir. primal: ", convergence_info.l_inf_primal_residual)
    # println("rel. primal: ", convergence_info.relative_l_inf_primal_residual * (1 + norm(b, Inf)))
    # println("dir. gap: ", out["optimality_gap"])
    # println("rel. gap: ", out["relative_optimality_gap"] * (1 + abs(convergence_info.primal_objective) + abs(convergence_info.dual_objective)))

    # Algorithm loop
    k = 0
    while true
        # Compute the shifted parameters and optimality measures
        b_bar = b - A * x_k
        l_bar = l - x_k
        u_bar = u - x_k
        c_bar = c - A' * y_k
        # A_bar = A .- 0.0

        # Scalar version
        delta_P = maximum([
            maximum(abs.(b_bar)),
            maximum(l_bar),
            maximum(-u_bar)
        ])

        # Scalar version
        delta_D = maximum([
            0,
            # maximum(abs.(c_bar)),
            maximum([
                c_bar[x_k .> (l + u) / 2] ; 
                -c_bar[x_k .<= (l + u) / 2]
                ], init=0)
        ])

        println("delta_P: ", delta_P)
        println("delta_D: ", delta_D)

        # Check the optimality condition for objective tolerance (KKT residual, max iters, and time limit)
        # (old from IR) if delta_P <= objective_tol && delta_2 <= objective_tol && delta_S <= objective_tol || k >= max_iter || remaining_time <= 0
        if ((
            out["relative_optimality_gap"][end]             <= objective_tol && 
            out["relative_l_inf_primal_residual"][end]      <= objective_tol && 
            out["relative_l_inf_dual_residual"][end]        <= objective_tol
            # convergence_info.relative_optimality_gap          <= objective_tol && 
            # convergence_info.relative_l_inf_primal_residual   <= objective_tol && 
            # convergence_info.relative_l_inf_dual_residual     <= objective_tol
            ) || 
            k >= max_iter || 
            remaining_time <= 0 ||
            out["termination_reason"][end] == "TERMINATION_REASON_TIME_LIMIT"
            )

            total_time = time() - total_time

            println("Optimality tolerance or max, iterations achieved in iteration: k=", k)

            # Print deltas
            println("delta_P: ", delta_P)
            println("delta_D: ", delta_D)

            # Update the output metrics
            out["last_iteration"] = k
            out["total_time"] = total_time
            return out
        end

        # Scalar version
        # Compute the scaling factors
        if no_alpha
            Delta_P = 1 / delta_P
            Delta_D = 1 / delta_D

            # # v6
            # Delta_P = (1 / delta_P) * (k == 0) + min( 1 / delta_P , alpha*Delta_P ) * (k > 0) 
            # Delta_D = (1 / delta_D) * (k == 0) + min( 1 / delta_D , alpha*Delta_D ) * (k > 0) 

            # Log test_3
            # Delta_P = log(1 + 1 / delta_P)
            # Delta_D = log(1 + 1 / delta_D)
            # v3: 1/delta decay # v4: decay x2
            # Delta_P = ( 1 / delta_P )^(1/(2*(k+1)))
            # Delta_D = ( 1 / delta_D )^(1/(2*(k+1)))
        else # with alpha scaling
            Delta_P = min( 1 / delta_P , alpha*Delta_P ) 
            Delta_D = min( 1 / delta_D , alpha*Delta_D ) 
            # Log test_3
            # Delta_P = min( log(1 + 1 / delta_P) , alpha*Delta_P ) 
            # Delta_D = min( log(1 + 1 / delta_D) , alpha*Delta_D ) 
            # v3: 1/delta decay
            # Delta_P = min( ( 1 / delta_P )^(1/(2*(k+1))) , alpha*Delta_P ) 
            # Delta_D = min( ( 1 / delta_D )^(1/(2*(k+1))) , alpha*Delta_D ) 
        end

        # Bound on the scaling
        Delta_P = max.( min.(Delta_P, scaling_bound), 1)#1/scaling_bound )
        Delta_D = max.( min.(Delta_D, scaling_bound), 1)#1/scaling_bound )

        # Print Deltas
        println("Delta_P: ", Delta_P)
        println("Delta_D: ", Delta_D)


        # Build the new LP
        b_bar = b_bar * Delta_P
        l_bar = l_bar * Delta_P
        u_bar = u_bar * Delta_P
        c_bar = c_bar * Delta_D


        # Redefine iterative refinement tolerance in terms of KKT_0
        skip_IR = false
        kkt_k = norm([out["relative_l_inf_primal_residual"], out["relative_l_inf_dual_residual"], out["relative_optimality_gap"]], Inf)
        if k > 1
            iterative_tol_k = max( min(kkt_k*tol_decrease_factor, iterative_tol_k*tol_decrease_factor), objective_tol ) # redefinition of epsilon_IR
        # elseif k>1
        println("\n\n\n\nINITIAL ITERATIVE TOL\n\n\n\n")
        #     iterative_tol_k = 1e-1/(10^(k-2))
        elseif k==1 # switch this
            println("\n\n\n\nSOLVING VIA DIRECT CUPDLP\n\n\n\n")

            # Delta_P = 1
            # Delta_D = 1
            # Delta_2 = ones(length(Delta_2))
            # Delta_3 = ones(length(Delta_3))
            iterative_tol_k = objective_tol
            println("Iteration ", k + 1, ", of ", instance_name, ", solved via scalar")
            remaining_time -= (time() - t_start_k)
            println("remaining_time: ", remaining_time)
            t_start_k = time()
            output = cuPDLP_from_initial_sol(
                lp_0_stand, 
                iterative_tol_k, 
                remaining_time, 
                true, #save_log
                instance_name, 
                output_path,
                x_k, 
                y_k, 
                A,
                b,
                l,
                u,
                c
            )
            t_pdlp_k = time() - t_start_k
            skip_IR = true
        else  
            # iterative_tol_k does not exists yet on k=0
            global iterative_tol_k = max( kkt_k*tol_decrease_factor, objective_tol )
        end

        if !skip_IR

            lp_k = cuPDLP.QuadraticProgrammingProblem(
                l_bar,
                u_bar,
                lp_0_stand.objective_matrix,
                c_bar,
                0,
                A, # A_bar,
                b_bar,
                lp_0_stand.num_equalities
            )

            # Solve the new LP (blackbox PDLP algorithm)
            println("Iteration ", k + 1)
            remaining_time -= (time() - t_start_k)
            println("remaining_time: ", remaining_time)
            t_start_k = time()
            params, lp_k, output = call_pdlp(
                lp_k,
                iterative_tol_k, #/(10^k), # v7: shrinking tolerance
                remaining_time,
                save_log,
                Dict(
                "instance_name" => instance_name, # or some variation of name
                "output_dir" => output_path,
                "current_iter_k" => k,
                ),
                typemax(Int32) # iteration_limit: No iteration limit on the subproblems (could be change to same as iter 0). KKT_0
            )
            t_pdlp_k = time() - t_start_k

            # Scalar version
            x_k = x_k + output.primal_solution / Delta_P
            y_k = y_k + output.dual_solution / Delta_D

        else

            x_k = output.primal_solution#x_k + output.primal_solution / Delta_P
            y_k = output.dual_solution # y_k + output.dual_solution / Delta_D
        end

        optimal_primal_cost = sum(c.*x_k)
        optimal_dual_cost = sum(b.*y_k)
        println("optimal_primal_cost on k=",k,": ", optimal_primal_cost)
        println("optimal_dual_cost on k=",k,": ", optimal_dual_cost)

        # Calculate the KKT error of the problem
        # qp_cache = cuPDLP.cached_quadratic_program_info(lp_0_stand) # As in "optimize" function (line 462)
        # lp_cuLPP = cuPDLP.qp_cpu_to_gpu(lp_0_stand)
        convergence_info = cuPDLP.compute_convergence_information(
            cuLPP,#::CuLinearProgrammingProblem,
            qp_cache,#::CachedQuadraticProgramInfo,
            CuVector{Float64}(x_k),#::CuVector{Float64},
            CuVector{Float64}(y_k),#::CuVector{Float64},
            1.0,#::Float64,
            cuPDLP.POINT_TYPE_AVERAGE_ITERATE,#::PointType,
            CuVector{Float64}(A*x_k),#::CuVector{Float64},
            CuVector{Float64}(c - A'y_k),#::CuVector{Float64},
            buffer_kkt#::BufferKKTState,
        )

        # Update the output metrics
        push!(out["termination_reason"], output.termination_reason)
        push!(out["blackbox_time"], t_pdlp_k)
        push!(out["primal_objective"], convergence_info.primal_objective)
        push!(out["dual_objective"], convergence_info.dual_objective)
        push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
        push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
        push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
        push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
        push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
        push!(out["optimality_gap"], abs(convergence_info.primal_objective - convergence_info.dual_objective))
        # Scalar version
        push!(out["Delta_P"], Delta_P) # careful with the scalar version meaning 
        push!(out["Delta_D"], Delta_D) # careful with the scalar version meaning
        push!(out["iterative_tol_k"], iterative_tol_k)
        # Matrix version
        # push!(out["D1_condition_number"], maximum(Delta_1)/minimum(Delta_1)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        # push!(out["D2_condition_number"], maximum(Delta_2)/minimum(Delta_2)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        k += 1
    end
end


# New version (2025)
function M_iterative_refinement(
    instance_path::String,
    output_path::String,
    iterative_tol = 1e-3,
    objective_tol=1e-8,
    time_sec_limit=300,
    max_iter =100,
    alpha=1.1,
    save_log=false, # save the log on each iteration
    scaling_type="D3_eq_D2inv", # "D3_eq_D2inv", "D3_eq_D2_eq_I", "D3_eq_D2_and_swap", "D3_dual_violation", "D3_dual_violation_swap", "D3_D2_iterative_swap", "D3_D2_mix", "D3_D2_mix_pure", "D123_pure", "D123_pure_max"
    scaling_bound=1e50, # bound on the scaling values
    tol_decrease_factor=0.1, # min(KKT*decrease factor, last_tol*decrease_factor)
    pdlp_iteration_limit=1000,
)


    # Instance name
    instance_name = replace(split(instance_path, "/")[end], ".mps"=>"")
    instance_name = replace(split(instance_path, "/")[end], ".gz"=>"")

    # Output metrics
    out = Dict(
        "instance" => instance_path,
        "iterative_tol" => iterative_tol,
        "objective_tol" => objective_tol,
        "candidate_type" => cuPDLP.POINT_TYPE_AVERAGE_ITERATE,
        "termination_reason" => cuPDLP.TerminationReason[],
        "alpha" => alpha,
        "primal_objective" => Float64[],
        "dual_objective" => Float64[],
        # "l_inf_primal_residual" => Float64[],
        # "l_inf_dual_residual" => Float64[],
        "relative_l_inf_primal_residual" => Float64[], 
        "l_inf_primal_residual" => Float64[],  # (1 + norm) correction (this need 1 more transformation)
        "relative_l_inf_dual_residual" => Float64[],
        "l_inf_dual_residual" => Float64[], # (1 + norm) correction
        "relative_optimality_gap" => Float64[], # There is no "l_inf" or whatsoever
        "optimality_gap" => Float64[], # (1 + norm) correction
        "D1_condition_number" => Float64[],
        "D2_condition_number" => Float64[],
        "D3_condition_number" => Float64[],
        "D1_max" => Float64[],
        "D2_max" => Float64[],
        "D3_max" => Float64[],
        "time_sec_limit" => time_sec_limit,
        "max_iter" => max_iter,
        "last_iteration" => 0,
        "blackbox_time" => Float64[],
        "total_time" => 0.0,
        "A_condition_number" => 0.0,
        # "A_spectral_norm" => 0.0,
        "P_condition_number" => 0.0,
        # "P_spectral_norm"=> 0.0,
        "iterative_tol_k"=>Float64[iterative_tol]
    )

    # Read the instance from the path and change it to standard form
    println("Trying to read instance from ", instance_path)
    lp_0 = cuPDLP.qps_reader_to_standard_form(instance_path)
    println("Instance read successfully")


    # Initial iteration of the blackbox PDLP algorithm
    println("Initial iteration of the blackbox PDLP algorithm")
    remaining_time = time_sec_limit
    total_time = time()
    t_start_k = time()
    params, lp_0_stand, output = call_pdlp(
        lp_0,           # LP in standard form
        iterative_tol,  # IR tolerance
        time_sec_limit, # Time limit for PDLP
        save_log,       # Save logs or not on each iteration of IR
        Dict(
        "instance_name" => instance_name, # or some variation of name
        "output_dir" => output_path,
        "current_iter_k" => 0,
        ),               # Output dictionary, for saving logs
        (objective_tol == iterative_tol) * typemax(Int32) + (objective_tol < iterative_tol) * pdlp_iteration_limit # iteration_limit: First call of PDLP with a limit, for hard-to-converge instances. KKT_0
    )
    lp = lp_0_stand
    t_pdlp_0 = time() - t_start_k

    # Initial optimization results ("approximate primal–dual solution")
    x_k = output.primal_solution
    y_k = output.dual_solution

    println("primal solution size: ", size(x_k))    
    println("  dual solution size: ", size(y_k))

    # Original parameters of the problem
    c = lp.objective_vector
    A = lp.constraint_matrix
    b = lp.right_hand_side
    l = lp.variable_lower_bound
    u = lp.variable_upper_bound
    # optimal_primal_cost = sum(c.*x_k)
    # optimal_dual_cost = sum(b.*y_k)


    # Initial parameters
    Delta_1 = ones(length(b)) 
    Delta_2 = ones(length(l)) 
    Delta_3 = ones(length(c)) 

    ##################################################################################
    # NEW: Compute the condition number of A, and also of [A b; c' 0], to check the
    # illness of the problem
    ##################################################################################
    println("Computing condition # of A and P...")
    # 1. Condition number of A
    # 1.1 Some approximation
    # sigma_max, bnd, nprod, ntprod = tsvdvals(A, k=1) # Sparse Sing. Values Computation
    # sigma_min, bnd, nprod, ntprod = tsvdvals_irl(A, k=1) # Sparse Sing. Values Computation
    # "Infinity norm":
    # A_norms = sum(abs.(A) , dims = 2)
    sigma_max = maximum(sum(abs.(A) , dims = 2))
    out["A_condition_number"] = sigma_max/minimum(sum(abs.(A) , dims = 2))
    println("Abs ratio of A (cond. # approx.): ", out["A_condition_number"])
    # # 1.2 Exact calculation
    # singular_values = svdvals(Matrix(A))
    # sigma_max = maximum(singular_values)
    # sigma_min = minimum(singular_values[singular_values .> 0])
    # # out["A_condition_number"] = sigma_max / sigma_min 
    # # out["A_spectral_norm"] = sigma_max
    # println("Cond. # of A done: ", sigma_max / sigma_min )

    # 2. Condition number of the problem
    # 2.1 Some approximation
    # sigma_max, bnd, nprod, ntprod =  tsvdvals([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0], k=1) # Sparse Sing. Values Computation
    # sigma_max, bnd, nprod, ntprod =  tsvdvals_irl([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0], k=1) # Sparse Sing. Values Computation
    # "Infinity norm":
    # P_norms = sum(abs.([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]) , dims = 2)
    # out["P_condition_number"] = maximum(sum(abs.([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]) , dims = 2))/minimum(sum(abs.([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]) , dims = 2))
    out["P_condition_number"] = maximum(sum(abs.([A b; c' 0]) , dims = 2))/minimum(sum(abs.([A b; c' 0]) , dims = 2))
    println("Abs ratio of P (cond. # approx.): ", out["P_condition_number"])
    # sigma_max = maximum(sum(abs.(A) , dims = 2))
    # println("Abs ratio of P (v2) (cond. # approx.): ", (sigma_max + norm(b) + norm(c))/min(sigma_max, norm(b), norm(c)))
    # # # 2.2 Exact calculation
    # # sigma_max = maximum(singular_values)
    # singular_values = svdvals(Matrix([(A./sigma_max) (b./norm(b)); (c'./norm(c)) 0]))
    # sigma_max = maximum(singular_values)
    # sigma_min = minimum(singular_values[singular_values .> 0])
    # # out["P_condition_number"] = sigma_max/sigma_min
    # # out["P_spectral_norm"] = sigma_max
    # println("Cond. # of P done: ", sigma_max/sigma_min)
    if alpha == 1024
        alpha = out["P_condition_number"]
        out["alpha"] = alpha
    end
    ##################################################################################
    
    # Compute the KKT error
    buffer_kkt = cuPDLP_KKT_buffer(lp_0_stand, x_k, y_k)
    qp_cache = cuPDLP.cached_quadratic_program_info(lp_0_stand) # As in "optimize" function (line 462)
    cuLPP = cuPDLP.qp_cpu_to_gpu(lp_0_stand)
    convergence_info = cuPDLP.compute_convergence_information(
        cuLPP,#::CuLinearProgrammingProblem,
        qp_cache,#::CachedQuadraticProgramInfo,
        CuVector{Float64}(x_k),#::CuVector{Float64},
        CuVector{Float64}(y_k),#::CuVector{Float64},
        1.0,#::Float64,
        cuPDLP.POINT_TYPE_AVERAGE_ITERATE,#::PointType,
        CuVector{Float64}(A*x_k),#::CuVector{Float64},
        CuVector{Float64}(c - A'y_k),#::CuVector{Float64},
        buffer_kkt#::BufferKKTState,
    )


    # Update the output metrics
    push!(out["termination_reason"], output.termination_reason)
    push!(out["blackbox_time"], t_pdlp_0)
    push!(out["primal_objective"], convergence_info.primal_objective)
    push!(out["dual_objective"], convergence_info.dual_objective)
    push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
    push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
    push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
    push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
    push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
    push!(out["optimality_gap"], abs(convergence_info.primal_objective - convergence_info.dual_objective))
    # Matrix version
    push!(out["D1_condition_number"], maximum(Delta_1)/minimum(Delta_1)) # Normal matrix: cond_number = |delta_max|/|delta_min|
    push!(out["D2_condition_number"], maximum(Delta_2)/minimum(Delta_2)) # Normal matrix: cond_number = |delta_max|/|delta_min|
    push!(out["D3_condition_number"], maximum(Delta_3)/minimum(Delta_3)) # Normal matrix: cond_number = |delta_max|/|delta_min|
    push!(out["D1_max"], maximum(Delta_1)) 
    push!(out["D2_max"], maximum(Delta_2)) 
    push!(out["D3_max"], maximum(Delta_3)) 


    # Redefine iterative refinement tolerance in terms of KKT_0
    kkt_0 = norm([out["relative_l_inf_primal_residual"], out["relative_l_inf_dual_residual"], out["relative_optimality_gap"]], Inf)
    iterative_tol = max( iterative_tol, 10^floor(log10(kkt_0)) ) # redefinition of epsilon_IR

    # Algorithm loop
    k = 0
    while true
        # Compute the shifted parameters and optimality measures
        b_bar = b - A * x_k
        l_bar = l - x_k
        u_bar = u - x_k
        c_bar = c - A' * y_k
        A_bar = A .- 0.0

        # Matrix version
        delta_1 = abs.(b_bar) # Eq. constraints violation. To use in D1: (D1 A D2^-1) x = D1 (b - Ax^*)
        println("Max violation of Ax=b (delta_1): ", maximum(delta_1))
        # delta_P = maximum([
        #     maximum(abs.(b_bar)),
        #     maximum(l_bar),
        #     maximum(-u_bar)
        # ])

        # Matrix version
        delta_2 = max.( # l/u constraint violation. To use in D2: D2 (u - x^*) >= x >= D2 (l - x^*)
            zeros(size(l_bar)), # Safety card for only negative violations (which is the desired case)
            l_bar,
            # l_bar .* (1 + sqrt(l_finite'*l_finite)), # rel. correction (mine) [delte?]
            -u_bar,
            # -u_bar .* (1 + sqrt(u_finite'*u_finite)), # rel. correction (mine) [delte?]
        )   
        println("Max violation of l/u bounds (delta_2): ", maximum(delta_2))

        scaling_types_with_3 = [
            "D3_dual_violation", "D3_dual_violation_swap", "D3_D2_iterative_swap", "D3_D2_iterative_swap_indep", "D3_D2_mix", "D3_D2_mix_pure", 
            "D123_pure", "D123_pure_max", "D3_D2_mix_max", "D3_D2_mix_max_pure"]
            # , "D2_D3_adaptive", "D2_D3_adaptive_v2","D2_D3_adaptive_v3",
            #  "D2_D3_adaptive_v4", "D2_D3_adaptive_v5", "D2_D3_adaptive_v7"]
        if (scaling_type in scaling_types_with_3 || contains(scaling_type, "adaptive"))
            c_bar_sign = Vector{Float64}(c_bar) # Opposite sign c_bar's
            c_bar_sign[x_k .<= (l + u) / 2] .= -Vector{Float64}(c_bar[x_k .<= (l + u) / 2])
            # c_bar_plus = Vector{Float64}(c_bar) # original
            # c_bar_plus[x_k .<= (l + u) / 2] .= -Inf # original
            # c_bar_minus = Vector{Float64}(c_bar) # original
            # c_bar_minus[x_k .> (l + u) / 2] .= Inf # original
            delta_3 = max.(
                zeros(size(c_bar)), # In case there is no positive c_bar_sign
                # c_bar_plus,
                # -c_bar_minus
                c_bar_sign
            )   
            # delta_D = maximum([
            #     0,
            #     # maximum(abs.(c_bar)),
            #     maximum([
            #         c_bar[x_k .> (l + u) / 2] ; 
            #         -c_bar[x_k .<= (l + u) / 2]
            #         ], init=0)
            # ])
            println("Max violation of dual c-A'y (delta_3): ", maximum(delta_3))
        end

        # Check the optimality condition for objective tolerance (KKT residual, max iters, and time limit)
        # (old from IR) if delta_P <= objective_tol && delta_2 <= objective_tol && delta_S <= objective_tol || k >= max_iter || remaining_time <= 0
        if ((
            out["relative_optimality_gap"][end]             <= objective_tol && 
            out["relative_l_inf_primal_residual"][end]      <= objective_tol && 
            out["relative_l_inf_dual_residual"][end]        <= objective_tol
            # convergence_info.relative_optimality_gap        <= objective_tol &&
            # convergence_info.relative_l_inf_primal_residual <= objective_tol && 
            # convergence_info.relative_l_inf_dual_residual   <= objective_tol
            ) || 
            k >= max_iter || 
            remaining_time <= 0 ||
            out["termination_reason"][end] == "TERMINATION_REASON_TIME_LIMIT"
            )

            total_time = time() - total_time

            println("Optimality tolerance or max, iterations achieved in iteration: k=", k)

            # Print deltas
            # println("delta_1: ", delta_1)
            # println("delta_2: ", delta_2)

            # Update the output metrics
            out["last_iteration"] = k
            out["total_time"] = total_time
            return out
        end

        # Matrix version
        # Compute the scaling factors
        println("Computing the scaling factors... (Delta1/Delta2/Delta3)")
        # scaling_types: "D3_eq_D2inv", "D3_eq_D2_eq_I", "D3_eq_D2_and_swap", "D3_dual_violation"
        Delta_1 = min.(1 ./delta_1, alpha * Delta_1 ) # 1 ./  max.(delta_1, 1 ./ (alpha * Delta_1) )#  
        if scaling_type in ["D123_pure", "D123_pure_max"]
            Delta_1 = 1 ./delta_1 # 1 ./  max.(delta_1, 1 ./ (alpha * Delta_1) )#  
        end
        Delta_2 = min.(1 ./delta_2, alpha * Delta_2 )  # 1 ./  max.(delta_2, 1 ./ (alpha * Delta_2) )
        if scaling_type=="D3_eq_D2inv" 
            Delta_3 = 1 ./ Delta_2
        elseif scaling_type in ["D3_eq_D2_eq_I", "D3_eq_D2_eq_I_indep"] # D3 = D2 = I (only equalities scaled) 
            Delta_2 = ones(length(Delta_2))
            Delta_3 = Delta_2
            if scaling_type == "D3_eq_D2_eq_I_indep"
                Delta_1 = 1 ./delta_1  # Indep of alpha
            end
        elseif scaling_type=="D3_eq_D2_and_swap"
            Delta_3 = Vector{Float64}(Delta_2) # Now D3 is D2, and D2 is D2inv
            Delta_2 = 1 ./ Delta_3 # swap
        elseif scaling_type=="D3_dual_violation" # Just multiply by D3 indep. as in IR scalar
            Delta_3 = min.(1 ./delta_3, alpha * Delta_3 ) # WARNING: Not so clear how to recover the dual y
        elseif scaling_type=="D3_dual_violation_swap"
            Delta_3 = min.(1 ./delta_3, alpha * Delta_3 ) # Dual violation on (c-A'y), but D2 = D3inv (recovers y)
            Delta_2 = 1 ./ Delta_3 # swap
        elseif scaling_type in ["D3_D2_iterative_swap", "D3_D2_iterative_swap_indep"]
            if k % 2 == 0   # If its an even number
                if scaling_type == "D3_D2_iterative_swap_indep"
                    Delta_2 = 1 ./delta_2 # redefine D2 indep of alpha
                end
                Delta_3 = 1 ./ Delta_2 # same as "D3_eq_D2inv": D3 = D2inv
            else            # If its an odd number
                if scaling_type == "D3_D2_iterative_swap_indep"
                    Delta_3 = 1 ./delta_3 # redefine D3 indep of alpha
                else
                    Delta_3 = min.(1 ./delta_3, alpha * Delta_3 ) # Scaling (c-A'y), and D2 = D3inv
                end
                Delta_2 = 1 ./ Delta_3 # swap
            end
        elseif scaling_type=="D3_D2_mix"
            Delta_2 = min.(delta_3, 1 ./delta_2, alpha*Delta_2) # Not so clear if min nor scale w. alpha (NOT 1/Delta_2 to not have numerical issues w/ max in D3)
            Delta_3 = 1 ./ Delta_2
        elseif scaling_type=="D3_D2_mix_max"
            Delta_3 = min.(delta_2, 1 ./delta_3, alpha*Delta_3) # Not so clear if min nor scale w. alpha (NOT 1/Delta_2 to not have numerical issues w/ max in D3)
            Delta_2 = 1 ./ Delta_3
        elseif scaling_type in ["D3_D2_mix_pure", "D123_pure"]
            Delta_2 = min.(delta_3, 1 ./delta_2) # Not so clear if min nor scale w. alpha (NOT 1/Delta_2 to not have numerical issues w/ max in D3)
            Delta_3 = 1 ./ Delta_2
        elseif scaling_type in ["D3_D2_mix_max_pure", "D123_pure_max"]
            Delta_3 = min.(1 ./delta_3, delta_2) # equivalent to max in delta_2 of D123 pure
            Delta_2 = 1 ./ Delta_3
        elseif scaling_type == "D2_D3_adaptive"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            # Check if one is getting too far away
            # If there is no much violation difference, be conservative
            if (abs(log10(primal_res) - log10(primal_res)) <= 2 || 
                ( 
                    log10(primal_res) - log10(objective_tol) <= 2 &&
                    log10(dual_res) - log10(objective_tol) <= 2              
                ))
                Delta_3 = ones(length(Delta_2))
                Delta_2 = ones(length(Delta_2))
            # If primal is in a worse situation, focus in primal violation
            elseif log10(primal_res) > log10(primal_res) 
                Delta_2 = min.(1 ./delta_2, alpha * Delta_2 )
                Delta_3 = 1 ./ Delta_2
                # Delta_1 = 1 ./ delta_1
            # If dual is in a worse situation, focus in dual violation
            else 
                Delta_3 = min.(1 ./delta_3, alpha * Delta_3 )
                Delta_2 = 1 ./ Delta_3
                # Delta_1 = 1 ./ delta_1
            end
        elseif scaling_type == "D2_D3_adaptive_v2"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            # Check if one is getting too far away
            # If primal is in a worse situation, focus in primal violation
            if log10(primal_res) > log10(primal_res) 
                Delta_2 = min.(1 ./delta_2, alpha * Delta_2 )
                Delta_3 = 1 ./ Delta_2
            # If dual is in a worse situation, focus in dual violation
            else 
                Delta_3 = min.(1 ./delta_3, alpha * Delta_3 )
                Delta_2 = 1 ./ Delta_3
            end
        # elseif scaling_type == "D2_D3_adaptive_v3"
        #     # Adaptive method for nonconvergent methods
        #     primal_res = convergence_info.relative_l_inf_primal_residual
        #     dual_res = convergence_info.relative_l_inf_dual_residual

        #     # Check if one is getting too far away
        #     # If primal is in a worse situation, focus in primal violation
        #     if log10(primal_res) < log10(primal_res) 
        #         Delta_2 = 1 ./ delta_2 #min.(1 ./delta_2, alpha * Delta_2 )
        #         Delta_3 = 1 ./ Delta_2
        #         # Delta_1 = 1 ./ delta_1
        #     # If dual is in a worse situation, focus in dual violation
        #     else 
        #         Delta_3 = 1 ./ delta_3#min.(1 ./delta_3, alpha * Delta_3 )
        #         Delta_2 = 1 ./ Delta_3
        #         # Delta_1 = 1 ./ delta_1
        #     end
        elseif scaling_type == "D2_D3_adaptive_v4"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            # Check if one is getting too far away
            # If primal is in a worse situation, focus in primal violation
            if log10(primal_res) > log10(dual_res) 
                Delta_2 = max.( delta_2, alpha * Delta_2 ) #min.(1 ./delta_2, alpha * Delta_2 )
                Delta_3 = 1 ./ Delta_2
                Delta_1 = max.( delta_1, alpha * Delta_1 )
            # If dual is in a worse situation, focus in dual violation
            else 
                Delta_3 = max.( delta_3, alpha * Delta_3 )#min.(1 ./delta_3, alpha * Delta_3 )
                Delta_2 = 1 ./ Delta_3
                Delta_1 = max.( delta_1, alpha * Delta_1 )
            end
        elseif scaling_type == "D2_D3_adaptive_v5"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            if maximum(delta_1) > 0 # If not zero violation
                # delta correction
                n_delta_1 = floor(log10(maximum(delta_1)))
                delta_1 = delta_1 * 10^(-n_delta_1)
                Delta_1 = max.( delta_1, alpha * Delta_1 )
            else 
                Delta_1 = ones(length(Delta_1))
            end 

            # Check if one is getting too far away
            # If primal is in a worse situation, focus in primal violation
            if log10(primal_res) - log10(dual_res) >= 1
                if maximum(delta_2) > 0
                    # delta correction
                    n_delta_2 = floor(log10(maximum(delta_2)))
                    delta_2 = delta_2 * 10^(-n_delta_2) 
                    Delta_2 = max.( delta_2, alpha * Delta_2 ) #max.(1 ./delta_2, alpha * Delta_2 )
                    Delta_3 = 1 ./ Delta_2
                else 
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3))
                end
                # If dual is in a worse situation, focus in dual violation
            elseif log10(primal_res) - log10(dual_res) <= -1
                if maximum(delta_3) > 0
                    # delta correction
                    n_delta_3 = floor(log10(maximum(delta_3)))
                    delta_3 = delta_3 * 10^(-n_delta_3)
                    Delta_3 = max.( delta_3, alpha * Delta_3 )#max.(1 ./delta_3, alpha * Delta_3 )
                    Delta_2 = 1 ./ Delta_3
                else
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3)) 
                end
            else 
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3))
            end
        elseif scaling_type == "D2_D3_adaptive_v3"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            if maximum(delta_1) > 0 # If not zero violation
                # delta correction
                n_delta_1 = floor(log10(maximum(delta_1)))
                delta_1 = delta_1 * 10^(-n_delta_1)
                Delta_1 = max.( delta_1, alpha * Delta_1 )
            else 
                Delta_1 = ones(length(Delta_1))
            end 

            # Check if one is getting too far away
            # If primal is in a worse situation, focus in primal violation
            if log10(primal_res) > log10(dual_res) 
                if maximum(delta_2) > 0
                    # delta correction
                    n_delta_2 = floor(log10(maximum(delta_2)))
                    delta_2 = delta_2 * 10^(-n_delta_2) 
                    Delta_2 = max.( delta_2, alpha * Delta_2 ) #max.(1 ./delta_2, alpha * Delta_2 )
                    Delta_3 = 1 ./ Delta_2
                else 
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3))
                end
                # If dual is in a worse situation, focus in dual violation
            elseif log10(primal_res) <= log10(dual_res)
                if maximum(delta_3) > 0
                    # delta correction
                    n_delta_3 = floor(log10(maximum(delta_3)))
                    delta_3 = delta_3 * 10^(-n_delta_3)
                    Delta_3 = max.( delta_3, alpha * Delta_3 )#max.(1 ./delta_3, alpha * Delta_3 )
                    Delta_2 = 1 ./ Delta_3
                else
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3)) 
                end
            end
        elseif scaling_type == "D2_D3_adaptive_v7"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            if maximum(delta_1) > 0# If not zero violation
                # delta correction
                n_delta_1 = floor(log10(maximum(delta_1)))
                delta_1 = delta_1 * 10^(-n_delta_1)
                Delta_1 = max.( delta_1, alpha ) # indep of alpha
            else 
                Delta_1 = ones(length(Delta_1))
            end 

            # Check if one is getting too far away
            # If primal is in a worse situation, focus in primal violation
            if log10(primal_res) - log10(dual_res) >= 1
                if maximum(delta_2) > 0
                    # delta correction
                    n_delta_2 = floor(log10(maximum(delta_2)))
                    delta_2 = delta_2 * 10^(-n_delta_2) 
                    Delta_2 = max.( delta_2, alpha )
                    Delta_3 = 1 ./ Delta_2
                else 
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3))
                end
                # If dual is in a worse situation, focus in dual violation
            elseif log10(primal_res) - log10(dual_res) <= -1
                if maximum(delta_3) > 0
                    # delta correction
                    n_delta_3 = floor(log10(maximum(delta_3)))
                    delta_3 = delta_3 * 10^(-n_delta_3)
                    Delta_3 = max.( delta_3, alpha )
                    Delta_2 = 1 ./ Delta_3
                else
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3)) 
                end
            else 
                # Identity matrices
                Delta_2 = ones(length(Delta_2))
                Delta_3 = ones(length(Delta_3)) 
            end
        elseif scaling_type == "D2_D3_adaptive_v8"
            # Adaptive method for nonconvergent methods
            primal_res = convergence_info.relative_l_inf_primal_residual
            dual_res = convergence_info.relative_l_inf_dual_residual

            if maximum(delta_1) > 0 # If not zero violation
                # delta correction
                n_delta_1 = floor(log10(maximum(delta_1)))
                delta_1 = delta_1 * 10^(-n_delta_1)
                Delta_1 = max.( delta_1, alpha ) # indep of alpha
            else 
                Delta_1 = ones(length(Delta_1))
            end 

            # Check if one is getting too far away
            # If primal is in a worse situation, focus in primal violation
            if log10(primal_res) > log10(dual_res) 
                if maximum(delta_2) > 0
                    # delta correction
                    n_delta_2 = floor(log10(maximum(delta_2)))
                    delta_2 = delta_2 * 10^(-n_delta_2) 
                    Delta_2 = max.( delta_2, alpha )
                    Delta_3 = 1 ./ Delta_2
                else 
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3))
                end
                # If dual is in a worse situation, focus in dual violation
            elseif log10(primal_res) <= log10(dual_res)
                if maximum(delta_3) > 0
                    # delta correction
                    n_delta_3 = floor(log10(maximum(delta_3)))
                    delta_3 = delta_3 * 10^(-n_delta_3)
                    Delta_3 = max.( delta_3, alpha )
                    Delta_2 = 1 ./ Delta_3
                else
                    # Identity matrices
                    Delta_2 = ones(length(Delta_2))
                    Delta_3 = ones(length(Delta_3)) 
                end
            end
        else
            println("Scaling method not available")
            exit()
        end
        println("Bounding the value of the scaling factors by: ", scaling_bound)
        # Scaling bound check
        Delta_1 = max.( min.(Delta_1, scaling_bound), 1/scaling_bound )
        Delta_2 = max.( min.(Delta_2, scaling_bound), 1/scaling_bound )
        Delta_3 = max.( min.(Delta_3, scaling_bound), 1/scaling_bound )
        # Delta_P = 1 / maximum([delta_P, 1/(alpha * Delta_P)])
        # Delta_D = 1 / maximum([delta_D, 1/(alpha * Delta_D)])

        # Build the new LP (Matrix version)
        println("Constructing the scaling matrices (D1/D2/D3)")
        D_1 = sparse(LinearAlgebra.I, length(Delta_1), length(Delta_1)).-0.0
        D_1[diagind(D_1)] = Delta_1
        # Print diagonal elements of D_1 which are different from 1.1
        println("D_1 lambda_max: ", maximum(D_1[diagind(D_1)]))
        println("D_1 lambda_min: ", minimum(D_1[diagind(D_1)]))

        D_2 = sparse(LinearAlgebra.I, length(Delta_2),length(Delta_2)).-0.0
        D_2[diagind(D_2)] = Delta_2
        D_2_inv = sparse(LinearAlgebra.I, length(Delta_2),length(Delta_2)).-0.0
        D_2_inv[diagind(D_2_inv)] = 1 ./ Delta_2
        # if inverted_D2
        #     D_2[diagind(D_2)] = 1 ./Delta_2
        # else 
        #     D_2[diagind(D_2)] = Delta_2
        # end
        # D_2_inv[diagind(D_2_inv)] = 1 ./ Delta_2
        println("D_2 lambda_max: ", maximum(D_2[diagind(D_2)]))
        println("D_2 lambda_min: ", minimum(D_2[diagind(D_2)]))
        
        D_3 = sparse(LinearAlgebra.I, length(Delta_3),length(Delta_3)).-0.0
        D_3[diagind(D_3)] = Delta_3
        println("D_3 lambda_max: ", maximum(D_3[diagind(D_3)]))
        println("D_3 lambda_min: ", minimum(D_3[diagind(D_3)]))

        # Scaling
        b_bar = D_1 * b_bar # Must have this scaling
        l_bar = D_2 * l_bar # D_2 inverse is 1/D_2
        u_bar = D_2 * u_bar # D_2 inverse is 1/D_2
        c_bar = D_3 * c_bar # D_2 inverse is 1/D_2
        A_bar = D_1 * A_bar * D_2_inv # New A



        # Redefine iterative refinement tolerance in terms of KKT_0
        skip_IR = false
        kkt_k = norm([out["relative_l_inf_primal_residual"], out["relative_l_inf_dual_residual"], out["relative_optimality_gap"]], Inf)
        if k > 1
            iterative_tol_k = max( min(kkt_k*tol_decrease_factor, iterative_tol_k*tol_decrease_factor), objective_tol ) # redefinition of epsilon_IR
        # elseif k>1
        println("\n\n\n\nINITIAL ITERATIVE TOL\n\n\n\n")
        #     iterative_tol_k = 1e-1/(10^(k-2))
        elseif k==1 # switch this
            println("\n\n\n\nSOLVING VIA DIRECT CUPDLP\n\n\n\n")

            Delta_1 = ones(length(Delta_1))
            Delta_2 = ones(length(Delta_2))
            Delta_3 = ones(length(Delta_3))
            iterative_tol_k = objective_tol
            println("Iteration ", k + 1, ", of ", instance_name, ", solved via ", scaling_type)
            remaining_time -= (time() - t_start_k)
            println("remaining_time: ", remaining_time)
            t_start_k = time()
            output = cuPDLP_from_initial_sol(
                lp_0_stand, 
                iterative_tol_k, 
                remaining_time, 
                true, #save_log
                instance_name, 
                output_path,
                x_k, 
                y_k, 
                A,
                b,
                l,
                u,
                c
            )
            t_pdlp_k = time() - t_start_k
            skip_IR = true
        else  
            # iterative_tol_k does not exists yet on k=0
            global iterative_tol_k = max( kkt_k*tol_decrease_factor, objective_tol )
        end

        if !skip_IR
            
        
            lp_k = cuPDLP.QuadraticProgrammingProblem(
                l_bar,
                u_bar,
                lp_0_stand.objective_matrix,
                c_bar,
                0,
                A_bar,
                b_bar,
                lp_0_stand.num_equalities
            )

            # Solve the new LP (blackbox PDLP algorithm)
            println("Iteration ", k + 1, ", of ", instance_name, ", solved via ", scaling_type)
            remaining_time -= (time() - t_start_k)
            println("remaining_time: ", remaining_time)
            t_start_k = time()
            params, lp_k, output = call_pdlp(
                lp_k,
                iterative_tol_k, #/(10^k), # v7: shrinking tolerance
                remaining_time,
                save_log,
                Dict(
                "instance_name" => instance_name, # or some variation of name
                "output_dir" => output_path,
                "current_iter_k" => k,
                ),
                typemax(Int32)
            )
            t_pdlp_k = time() - t_start_k
            # Matrix version
            # Retrieve the solution to the original problem
            x_k =  x_k + D_2_inv * output.primal_solution
            y_k =  y_k + D_1 * output.dual_solution 
        else
            # Retrieve the solution to the original problem
            x_k =  output.primal_solution
            y_k =  output.dual_solution  
        end


        optimal_primal_cost = sum(c.*x_k)
        optimal_dual_cost = sum(b.*y_k)
        println("optimal_primal_cost on k=",k,": ", optimal_primal_cost)
        println("optimal_dual_cost on k=",k,": ", optimal_dual_cost)

        # Calculate the KKT error of the problem
        # qp_cache = cuPDLP.cached_quadratic_program_info(lp_0_stand) # As in "optimize" function (line 462)
        # lp_cuLPP = cuPDLP.qp_cpu_to_gpu(lp_0_stand)
        convergence_info = cuPDLP.compute_convergence_information(
            cuLPP,#::CuLinearProgrammingProblem,
            qp_cache,#::CachedQuadraticProgramInfo,
            CuVector{Float64}(x_k),#::CuVector{Float64},
            CuVector{Float64}(y_k),#::CuVector{Float64},
            1.0,#::Float64,
            cuPDLP.POINT_TYPE_AVERAGE_ITERATE,#::PointType,
            CuVector{Float64}(A*x_k),#::CuVector{Float64},
            CuVector{Float64}(c - A'y_k),#::CuVector{Float64},
            buffer_kkt#::BufferKKTState,
        )

        # Update the output metrics
        push!(out["termination_reason"], output.termination_reason)
        push!(out["blackbox_time"], t_pdlp_k)
        push!(out["primal_objective"], convergence_info.primal_objective)
        push!(out["dual_objective"], convergence_info.dual_objective)
        push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
        push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
        push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
        push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
        push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
        push!(out["optimality_gap"], abs(convergence_info.primal_objective - convergence_info.dual_objective))
        # # Scalar version
        # push!(out["Delta_P"], Delta_P) # careful with the scalar version meaning 
        # push!(out["Delta_D"], Delta_D) # careful with the scalar version meaning
        # Matrix version
        push!(out["D1_condition_number"], maximum(Delta_1)/minimum(Delta_1)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        push!(out["D2_condition_number"], maximum(Delta_2)/minimum(Delta_2)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        push!(out["D3_condition_number"], maximum(Delta_3)/minimum(Delta_3)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        push!(out["D1_max"], maximum(Delta_1)) 
        push!(out["D2_max"], maximum(Delta_2)) 
        push!(out["D3_max"], maximum(Delta_3)) 
        push!(out["iterative_tol_k"], iterative_tol_k)
        k += 1
    end
end



# OLD VERSIONS
# # Read the command line
# args = parse_command_line()
# instance_dir = args["instance_path"]
# instance_name = basename(instance_dir)
# instance_name = replace(instance_name, r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "") # delete extension from instance_name
# output_dir = args["output_directory"]
# tol_it_ref = args["iter_tolerance"]
# tol_objective = args["obj_tolerance"]
# time_sec_limit = args["time_sec_limit"]
# max_iter = args["max_iter"]



# # Batch version
# println("-----------------------------------")
# println("Instance: ", instance_name, " max_iter: ", max_iter)
# # instance_path = instance_dir * instance_name
# # println("Instance path: ", )
# if max_iter == 0
#     output = iterative_refinement(instance_dir, tol_objective, tol_objective, time_sec_limit, max_iter)
# else
#     output = iterative_refinement(instance_dir, tol_it_ref, tol_objective, time_sec_limit, max_iter)
# end

# # Save the output in .json format
# output_path = output_dir * "/" * instance_name * "_out_k"*string(output["last_iteration"])*".json"
# open(output_path, "w") do io
#     write(io, JSON3.write(output, allow_inf = true))
# end




# function main()

#     # create a house problem
#     # kappa = 5
#     for kappa in [0.01, 0.1, 0.5, 1, 2]
#         for delta_exp in 8:12#[7]#[4,5,6] # k0 PDHG cannot solve it
#             # delta_exp = 8
#             delta = 10.0^(-delta_exp)
#             build_house_problem(kappa,delta_exp) #delta_exp = 10^(-delta_exp)
            

#             # instance_name = "2club200v15p5scn.mps.gz"
#             # instance_dir = "./MIPLIB/"
#             # instance_dir = "/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/"
#             instance_dir = "instance/"
#             # instance_path = instance_dir * instance_name 
#             # output_dir = "./output/MIPLIB_batch_scalar/test"
#             output_dir = "./output/mps1"
#             tol_it_ref = 1e-3
#             tol_objective = 1e-8
#             time_sec_limit = 600 # 10 mins per instance
#             max_iter_limit = 1e3 # Infinite iterations
#             # Read all the files in the instance directory
#             # instance_files = readdir(instance_dir)

#             alphas = [1e1, 1e2, 1e3, 1e4]#[1e3, 1e4, 1e5]#[1+1e-3,1+1e-2, 1+1e-1,1.5]

#             # Instances
#             instances = Dict(
#                 "house_instances"=>String["house_k$(kappa)_d$(delta)"],
#                 # "tiny_instances"=>String[ # "app1-2",
#                 #     "ns1456591",  "graph20-80-1rand", "blp-ic98", "piperout-d20", "ns1828997", 
#                 #     "neos-4292145-piako", "neos-960392", "d20200", "mushroom-best", "bppc6-02", "neos-1354092", 
#                 #     "neos-933638", "neos-4300652-rahue", "n2seq36q", "bppc6-06", "neos-933966", "ns1430538", 
#                 #     "neos-5195221-niemur", "neos-5193246-nerang", "germanrr", "ger50-17-trans-dfn-3t", 
#                 #     "ger50-17-trans-pop-3t", "neos-5196530-nuhaka", "neos-5266653-tugela", "stockholm", "neos-953928", 
#                 #     "dws008-03", "neos-1122047", "eva1aprime6x6opt", "supportcase23", "cmflsp50-24-8-8", "sorrell7", 
#                 #     "physiciansched5-3", "bab5", "pb-grow22", "gmut-76-40", "opm2-z8-s0", "neos-913984", "mzzv42z", 
#                 #     "neos-498623", "sct5", "ns930473", "iis-hc-cov", "neos-4954274-beardy", "neos-824661", "reblock420",
#                 #      "supportcase37", "chromaticindex512-7", "fhnw-binschedule2", "mzzv11", "neos-5013590-toitoi", 
#                 #      "neos-5188808-nattai", "brazil3", "t1722", "dws012-01", "neos-1171448", "leo1", "ci-s4", "neos-826224",
#                 #       "cmflsp40-24-10-7", "unitcal_7", "neos-4359986-taipa", "satellites2-60-fs", "shipsched", 
#                 #       "fhnw-schedule-paira200", "blp-ic97", "neos-4805882-barwon", "ns1631475", "neos-3372571-onahau",
#                 #        "neos-1593097", "rmatr200-p5", "neos-827175", "30n20b8", "sct32", "neos-932721", 
#                 #        "lr1dr04vc05v17a-t360", "ns1856153", "sct1", "rmatr200-p10", "2club200v15p5scn", "fiball",
#                 #         "supportcase40", "neos-950242", "v150d30-2hopcds", "momentum1", "ex1010-pi", "neos-578379", "neos-738098", "ns1830653"
#                 #     ],
#                 # "small_instances"=>String[
#                 #     "30n20b8", "hgms30", "var-smallemery-m6j6", "nursesched-medium-hint03", 
#                 #     "neos-2629914-sudost",
#                 #     "neos-4966258-blicks", "neos-3755335-nizao", "neos-3695882-vesdre", "neos6",
#                 #     "cmflsp40-24-10-7", 
#                 #     "triptim7", "neos-2746589-doon", "reblock420",
#                 #     "neos-872648", "neos-4760493-puerua", 
#                 #     "fhnw-schedule-pairb200", "sct1",
#                 #     "t1717", "iis-hc-cov", 
#                 #     "gmut-75-50", 
#                 #     "t1722", "ex1010-pi",
#                 #     "neos-5221106-oparau", "neos-1354092", "neos-827175",
#                 #      "radiationm40-10-02",
#                 #     "nw04", "neos-4359986-taipa", "neos-960392", "map18", "neos-932721",
#                 #     "gmut-76-40",
#                 # ],
#                 # "medium_instances"=>String[
#                 #     "ds-big", "neos-4647030-tutaki", "neos-5129192-manaia", "graph40-80-1rand",
#                 #     "neos-5049753-cuanza", "seqsolve1", "neos-5102383-irwell", "bab6",
#                 #     "neos-5123665-limmat", "shs1014", "shs1042", "wnq-n100-mw99-14",
#                 #     "fhnw-binschedule0", "fhnw-binschedule1", "neos-4321076-ruwer",
#                 #     "physiciansched3-3", "neos-5079731-flyers", "neos-3322547-alsek",
#                 #     "neos-4647027-thurso", "ns1644855", "datt256", "kosova1",
#                 #     "neos-4533806-waima", "neos-4647032-veleka", "z26", "neos-5118851-kowhai",
#                 #     "neos-4972437-bojana", "hgms62", "in", "zeil",
#                 # ]
#             )
            

#             # Iterate over the instances (test version)
#             # for instance_name in reverse(instance_files[1:end-1])# ["ci-s4.mps.gz"]# reverse(instance_files)#[15:end]
#             for instance_size in keys(instances)
#                 for instance_name in instances[instance_size]
#                     instance_name=instance_name*".mps"#".mps.gz"
#                     for alpha in alphas
#                         for max_iter_limit in [0,max_iter_limit]
#                             println("alhpa/max_iter: ", (alpha, max_iter_limit))
                            
#                             println("-----------------------------------")
#                             println("Instance: ", instance_name, " max_iter_limit: ", max_iter_limit)
#                             instance_path = instance_dir * instance_name
#                             if max_iter_limit > 0
#                                 println("Non base")
#                                 output = iterative_refinement(instance_path, tol_it_ref, tol_objective, time_sec_limit, max_iter_limit, alpha)
#                             elseif alpha == alphas[1] # Baseline
#                                 output = iterative_refinement(instance_path, tol_objective, tol_objective, time_sec_limit, max_iter_limit)
#                             else
#                                 continue
#                             end
#                             println(output)

#                             # Save the output in .json format
#                             # output_path = output_dir * "/" * instance_size * "/" *  instance_name * "_output_k"*string(output["last_iteration"])*"_alpha_"*string(alpha)*".json"
#                             output_path = output_dir * "/" * instance_size * "/" *  instance_name * "_output_k"*string(output["last_iteration"])*"_alpha_"*string(alpha)*".json"

#                             open(output_path, "w") do io
#                                 write(io, JSON3.write(output, allow_inf = true))
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# main()