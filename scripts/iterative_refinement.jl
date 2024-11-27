import ArgParse
import GZip
import JSON3

import cuPDLP

# My imports
# import JSON3
# import Formatting
using LinearAlgebra
using SparseArrays
using CUDA
# CUDA.set_runtime_version!(local_toolkit=true)
# CUDA.set_runtime_version!(v"v0.12.1.1")

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
        10,
        false,
        1.0,
        1.0,
        true,
        0,
        true,
        64,
        termination_params_warmup,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6),
    )

    cuPDLP.optimize(params_warmup, lp);
end


function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--instance_path"
        help = "The path to the instance to solve in .mps.gz or .mps format."
        arg_type = String
        required = true

        "--output_directory"
        help = "The directory for output files."
        arg_type = String
        required = true

        "--iter_tolerance"
        help = "IR iters KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-3

        "--obj_tolerance"
        help = "Final KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-8

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0

        "--max_iter"
        help = "Maximum of desired iterations in IR"
        arg_type = Int64
        default = 10
    end

    return ArgParse.parse_args(arg_parse)
end






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

    # You can write A = [AE 0; AI -I], with I of slacks
    n_inequalities = size(A, 1) - lp.num_equalities # num of slacks to add
    if n_inequalities > 0 
        # Identify matrix to add slack variables
        I = SparseArrays.sparse(LinearAlgebra.I, n_inequalities, n_inequalities)

        # Add n_eq columns and rows of zeros to I
        Z_I = [
            zeros(lp.num_equalities, n_inequalities); 
            -I
            ]
        A = [A Z_I] #hcat(A, I)

        # Add slack variables to the objective function
        c = lp.objective_vector
        c = [c; zeros(n_inequalities)]

        # Add slack variables to the upper bound
        u = lp.variable_upper_bound
        u = [u; Inf*ones(n_inequalities)]

        # Add slack variables to the lower bound
        l = lp.variable_lower_bound
        l = [l; zeros(n_inequalities)]

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
                            


function call_pdlp(lp, tolerance, time_sec_limit)


    
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
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1e-8,
        eps_dual_infeasible = 1e-8,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = cuPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        2,
        true,
        64,
        termination_params,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6),  
    )

    output = cuPDLP.optimize(
    params,
    lp
)
    return params, lp, output
end



function iterative_refinement(
    instance_path::String,
    initial_tol = 1e-3,
    objective_tol=1e-8,
    time_sec_limit=300,
    max_iter =100
)

    # Initial parameters
    Delta_P = 1
    Delta_D = 1
    alpha = 1.1# + initial_tol # (Think about this alpha > 1)

    # Output metrics
    out = Dict(
        "instance" => instance_path,
        "initial_tol" => initial_tol,
        "objective_tol" => objective_tol,
        "candidate_type" => cuPDLP.POINT_TYPE_AVERAGE_ITERATE,
        "primal_objective" => Float64[],
        "dual_objective" => Float64[],
        "l_inf_primal_residual" => Float64[],
        "l_inf_dual_residual" => Float64[],
        "relative_l_inf_primal_residual" => Float64[], 
        "relative_l_inf_dual_residual" => Float64[],
        "relative_optimality_gap" => Float64[],
        "time_sec_limit" => time_sec_limit,
        "max_iter" => max_iter,
        "max_time" => time_sec_limit,
        "final iteration" => 0,
        "blackbox_time" => Float64[],
        "total_time" => 0.0,
    )

    # Read the instance from the path
    println("Trying to read instance from ", instance_path)
    lp_0 = cuPDLP.qps_reader_to_standard_form(instance_path)
    println("Instance read successfully")


    # Initial iteration of the blackbox PDLP algorithm
    println("Initial iteration of the blackbox PDLP algorithm")
    remaining_time = time_sec_limit
    total_time = time()
    t_start_k = time()
    params, lp_0_stand, output = call_pdlp(
        lp_0,
        initial_tol,
        time_sec_limit
    )
    lp = lp_0_stand
    # println(output.iteration_stats[3])#convergence_information)
    # println(output)
    println("-------------------------------------------")
    # println(output.iteration_stats[end].step_size)#last iterate information
    # println(output.iteration_stats[end].primal_weight)#last iterate information
    # println(output.iteration_stats[end].convergence_information[1]) # actual convergence information
    # println("bb_primal_objective",output.iteration_stats[end].convergence_information[1].primal_objective) # Optimal objective value
    # println("bb_dual_objective",output.iteration_stats[end].convergence_information[1].dual_objective) # Optimal dual objective value
    # println("bb_relative_optimality_gap",output.iteration_stats[end].convergence_information[1].relative_optimality_gap) # relative_optimality_gap
    t_pdlp_0 = time() - t_start_k

    # Initial optimization results ("approximate primal–dual solution")
    x_k = output.primal_solution
    y_k = output.dual_solution

    println("dual solution size", size(y_k))
    c = lp.objective_vector
    A = lp.constraint_matrix
    b = lp.right_hand_side
    l = lp.variable_lower_bound
    u = lp.variable_upper_bound
    optimal_primal_cost = sum(c.*x_k)
    optimal_dual_cost = sum(b.*y_k)

    # Calculate the KKT error of the problem
    primal_size = size(x_k)[1]
    dual_size = size(y_k)[1]
    num_eq = lp.num_equalities
    buffer_original = cuPDLP.BufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
    )

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

    # Compute the KKT error
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
    push!(out["blackbox_time"], t_pdlp_0)
    push!(out["primal_objective"], convergence_info.primal_objective)
    push!(out["dual_objective"], convergence_info.dual_objective)
    push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
    push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
    push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
    push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
    push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)

    # Algorithm loop
    k = 0
    while true
        # Compute the shifted parameters and optimality measures
        b_bar = b - A * x_k
        l_bar = l - x_k
        u_bar = u - x_k
        c_bar = c - A' * y_k
        delta_P = maximum([
            maximum(abs.(b_bar)),
            maximum(l_bar),
            maximum(-u_bar)
        ])
        # Question: (l + u)/2 or previuous l and u???? (original apparently)
        # maximum([c_bar[x_k .> (l + u) / 2] ; -c_bar[x_k .<= (l + u) / 2]])
        # Idea: change to max (c1 - c2) in just one line
        # println("c_bar: ", c_bar)
        delta_D = maximum([
            0,
            # maximum(abs.(c_bar)),
            maximum([
                c_bar[x_k .> (l + u) / 2] ; 
                -c_bar[x_k .<= (l + u) / 2]
                ])
        ])
        # delta_S = abs(
        #     -sum(l_bar[x_k .<= (l + u) / 2] .* c_bar[x_k .<= (l + u) / 2], init=0) +
        #     sum(u_bar[x_k .> (l + u) / 2] .* c_bar[x_k .> (l + u) / 2], init=0)
        # )

        println("delta_P: ", delta_P)
        println("delta_D: ", delta_D)
        # println("delta_S: ", delta_S)
        # println("c", c)
        # println("A'*y_k", A'*y_k)

        # Check the optimality condition for objective tolerance
        # (old from IR) if delta_P <= objective_tol && delta_D <= objective_tol && delta_S <= objective_tol || k >= max_iter || remaining_time <= 0
        if convergence_info.relative_optimality_gap <= objective_tol && convergence_info.relative_l_inf_primal_residual <= objective_tol && convergence_info.relative_l_inf_dual_residual <= objective_tol || k >= max_iter || remaining_time <= 0
            total_time = time() - total_time

            println("Optimality tolerance or max, iterations achieved in iteration: k=", k)

            # Update the output metrics
            out["final iteration"] = k
            out["total_time"] = total_time
            return out
            # break
        end

        # Compute the scaling factors
        Delta_P = 1 / maximum([delta_P, 1/(alpha * Delta_P)])
        Delta_D = 1 / maximum([delta_D, 1/(alpha * Delta_D)])

        # Build the new LP
        b_bar = b_bar * Delta_P
        l_bar = l_bar * Delta_P
        u_bar = u_bar * Delta_P
        c_bar = c_bar * Delta_D
        lp = cuPDLP.QuadraticProgrammingProblem(
            l_bar,
            u_bar,
            lp.objective_matrix,
            c_bar,
            0,
            lp.constraint_matrix,
            b_bar,
            lp.num_equalities
        )

        # Solve the new LP (blackbox PDLP algorithm)
        println("Iteration ", k + 1)
        remaining_time -= time() - t_start_k
        println("remaining_time: ", remaining_time)
        t_start_k = time()
        params, lp, output = call_pdlp(
            lp,
            objective_tol,
            remaining_time
        )
        t_pdlp_k = time() - t_start_k

        # Retrieve the solution to the original problem
        x_k = x_k + output.primal_solution / Delta_P
        y_k = y_k + output.dual_solution / Delta_D
        optimal_primal_cost = sum(c.*x_k)
        optimal_dual_cost = sum(b.*y_k)
        println("optimal_primal_cost on k=",k,": ", optimal_primal_cost)
        println("optimal_dual_cost on k=",k,": ", optimal_dual_cost)

        # Calculate the KKT error of the problem
        qp_cache = cuPDLP.cached_quadratic_program_info(lp_0_stand) # As in "optimize" function (line 462)
        lp_cuLPP = cuPDLP.qp_cpu_to_gpu(lp_0_stand)
        convergence_info = cuPDLP.compute_convergence_information(
            lp_cuLPP,#::CuLinearProgrammingProblem,
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
        push!(out["blackbox_time"], t_pdlp_k)
        push!(out["primal_objective"], convergence_info.primal_objective)
        push!(out["dual_objective"], convergence_info.dual_objective)
        push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
        push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
        push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
        push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
        push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
        k += 1
    end
end

function main(
    instance_index::Int,
)
        

    # instance_name = "2club200v15p5scn.mps.gz"
    instance_dir = "./MIPLIB/"
    # instance_path = instance_dir * instance_name 
    output_dir = "./MIPLIB_output/"
    tol_it_ref = 1e-3
    tol_objective = 1e-8
    time_sec_limit = 3600 
    max_iter = 1e3
    # Read all the files in the instance directory
    instance_files = readdir(instance_dir)

    # Iterate over the instances (test version)
    # for instance_name in reverse(instance_files[1:end-1])# ["ci-s4.mps.gz"]# reverse(instance_files)#[15:end]
    instance_name = instance_files[instance_index]
    for max_iter in [0,max_iter]
        println("-----------------------------------")
        println("Instance: ", instance_name, " max_iter: ", max_iter)
        instance_path = instance_dir * instance_name
        if max_iter == 0
            output = iterative_refinement(instance_path, tol_objective, tol_objective, time_sec_limit, max_iter)
        else
            output = iterative_refinement(instance_path, tol_it_ref, tol_objective, time_sec_limit, max_iter)
        end
        println(output)

        # Save the output in .json format
        output_path = output_dir * instance_name * "_output_k"*string(output["final iteration"])*"_tol12.json"
        open(output_path, "w") do io
            write(io, JSON3.write(output, allow_inf = true))
        end

        # # Plot the max_delta_feas_opt vs iteration
        # using Plots
        # plot(output["max_delta_feas_opt"], label="max_delta_feas_opt", title="Optimality gap vs iteration")
        # savefig(output_dir * instance_name * "_optimality_gap_vs_iteration.png")
    end
    # end
end
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
# output_path = output_dir * "/" * instance_name * "_out_k"*string(output["final iteration"])*".json"
# open(output_path, "w") do io
#     write(io, JSON3.write(output, allow_inf = true))
# end









# instance_name = "2club200v15p5scn.mps.gz"
# instance_dir = "./MIPLIB/"
# instance_path = instance_dir * instance_name 
# output_dir = "./MIPLIB_output/"
# tol_it_ref = 1e-3
# tol_objective = 1e-8
# time_sec_limit = 600 
# max_iter = 1e3
# Read all the files in the instance directory
# instance_files = readdir(instance_dir)

# # Iterate over the instances (test version)
# for instance_name in reverse(instance_files[1:end-1])# ["ci-s4.mps.gz"]# reverse(instance_files)#[15:end]
#     for max_iter in [0,5]
#         println("-----------------------------------")
#         println("Instance: ", instance_name, " max_iter: ", max_iter)
#         instance_path = instance_dir * instance_name
#         if max_iter == 0
#             output = iterative_refinement(instance_path, tol_objective, tol_objective, time_sec_limit, max_iter)
#         else
#             output = iterative_refinement(instance_path, tol_it_ref, tol_objective, time_sec_limit, max_iter)
#         end
#         println(output)

#         # Save the output in .json format
#         output_path = output_dir * instance_name * "_output_k"*string(output["final iteration"])*"_tol12.json"
#         open(output_path, "w") do io
#             write(io, JSON3.write(output, allow_inf = true))
#         end

#         # # Plot the max_delta_feas_opt vs iteration
#         # using Plots
#         # plot(output["max_delta_feas_opt"], label="max_delta_feas_opt", title="Optimality gap vs iteration")
#         # savefig(output_dir * instance_name * "_optimality_gap_vs_iteration.png")
#     end
# end
