import ArgParse
import GZip
import JSON3

import cuPDLP

# My imports
# import JSON3
# import Formatting
using LinearAlgebra
using SparseArrays
using Statistics 

using CUDA



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
    iterative_tol = 1e-3,
    objective_tol=1e-8,
    time_sec_limit=300,
    max_iter =100,
    alpha=1.1,
    bound=1e3
)

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
        "l_inf_primal_residual" => Float64[],
        "l_inf_dual_residual" => Float64[],
        "relative_l_inf_primal_residual" => Float64[], 
        "relative_l_inf_dual_residual" => Float64[],
        "relative_optimality_gap" => Float64[],
        "D1_condition_number" => Float64[],
        "D2_condition_number" => Float64[],
        "time_sec_limit" => time_sec_limit,
        "max_iter" => max_iter,
        "last_iteration" => 0,
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
        iterative_tol,
        time_sec_limit
    )
    lp = lp_0_stand
    # println(output.iteration_stats[3])#convergence_information)
    # println(output)
    # println("-------------------------------------------")
    # println(output.termination_reason)
    # println(output.iteration_stats[end])
    # println(output.iteration_stats[end].convergence_information[1])
    # exit()
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

    # KKT relative error de-normalizer [...*(1+norm(param))]
    c_kkt = c
    b_kkt = b
    l_kkt = l 
    u_kkt = u

    # Initial parameters
    Delta_1 = ones(length(b)) 
    Delta_2 = ones(length(c)) 
    # Delta_P_lu = ones(size(l))
    # Delta_2 = ones(size(c))
    # alpha = 1.1# + iterative_tol # (Think about this alpha > 1)


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
    push!(out["termination_reason"], output.termination_reason)
    push!(out["blackbox_time"], t_pdlp_0)
    push!(out["primal_objective"], convergence_info.primal_objective)
    push!(out["dual_objective"], convergence_info.dual_objective)
    push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
    push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
    push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
    push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
    push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
    push!(out["D1_condition_number"], maximum(Delta_1)/minimum(Delta_1)) # Normal matrix: cond_number = |delta_max|/|delta_min|
    push!(out["D2_condition_number"], maximum(Delta_2)/minimum(Delta_2)) # Normal matrix: cond_number = |delta_max|/|delta_min|

    # Algorithm loop
    k = 0
    while true
        # Compute the shifted parameters and optimality measures
        b_bar = b - A * x_k
        l_bar = l - x_k
        u_bar = u - x_k
        c_bar = c - A' * y_k
        A_bar = A .- 0.0

        # Get a vector of the maximums of the form max(abs(b_bar_i), l_bar_i, -u_bar_i)
        # println("min b_bar: ", abs(minimum(b_bar))^2)
        # println("max b_bar: ", abs(maximum(b_bar))^2)
        # # println("max b_bar: ", maximum(b_bar))
        # println("min Ax: ", minimum(A*x_k))
        # println("max Ax: ", maximum(A*x_k))
        # println("min b: ", minimum(b))
        # println("max b: ", maximum(b))
        

        delta_1 = abs.(b_bar) # Ignore for now
        println("max delta_1: ", maximum(delta_1))
        # simple correction
        # delta_1 = delta_1 / iterative_tol ^ (k+1)
        # # median correction
        # # println("median: ", median(delta_1[delta_1.>0]))
        # # delta_1 = delta_1 / median(delta_1[delta_1.>0])
        # # relative error correction
        # # delta_1 = delta_1 .* (1 + sqrt(b_kkt'*b_kkt))
        # println("max delta_1 post: ", maximum(delta_1))




        # CONTINUE FROM HERE
        c_bar_plus = Vector{Float64}(c_bar) # original
        # relative error correction
        # c_bar_plus = Vector{Float64}(c_bar) .* (1 + sqrt(c_kkt'*c_kkt)) # rel. correction
        c_bar_plus[x_k .<= (l + u) / 2] .= -Inf # original
        c_bar_minus = Vector{Float64}(c_bar) # original
        # relative error correction
        # c_bar_minus = Vector{Float64}(c_bar) .* (1 + sqrt(c_kkt'*c_kkt))
        c_bar_minus[x_k .> (l + u) / 2] .= Inf # original
        # println("c_bar_plus: ", c_bar_plus)
        # println("c_bar_minus: ", c_bar_minus)
        # println("l_bar rel: ", )
        # l_finite = l_kkt[l_kkt .> -Inf]
        # u_finite = u_kkt[u_kkt .< Inf]
        delta_2 = max.(
            zeros(size(c_bar)), # almost zero
            l_bar,
            # l_bar .* (1 + sqrt(l_finite'*l_finite)), # rel. correction (mine)
            -u_bar,
            # -u_bar .* (1 + sqrt(u_finite'*u_finite)), # rel. correction (mine)
            c_bar_plus,
            -c_bar_minus
        )   
        println("max delta_2: ", maximum(delta_2))   
        # simple correction
        # delta_2 = delta_2 / iterative_tol ^ (k+1)
        # # median correction
        # # println("median: ", median(delta_2[delta_2.>0]))
        # # delta_2 = delta_2 / median(delta_2[delta_2.>0])
        # println("max delta_2 post: ", maximum(delta_2))
        # # relative error correction


        # Check the optimality condition for objective tolerance
        # (old from IR) if delta_P <= objective_tol && delta_2 <= objective_tol && delta_S <= objective_tol || k >= max_iter || remaining_time <= 0
        if convergence_info.relative_optimality_gap <= objective_tol && convergence_info.relative_l_inf_primal_residual <= objective_tol && convergence_info.relative_l_inf_dual_residual <= objective_tol || k >= max_iter || remaining_time <= 0
            total_time = time() - total_time

            println("Optimality tolerance or max, iterations achieved in iteration: k=", k)

            # Update the output metrics
            out["last_iteration"] = k
            out["total_time"] = total_time
            return out
            # break
        end

        # Compute the scaling factors
        # Delta_1 = 1 ./ max.(delta_1, 1 ./ (alpha * Delta_1))
        # Delta_2 = 1 ./ max.(delta_2, 1 ./ (alpha * Delta_2))
        # Bounds on the value of delta (lower and upper)
        Delta_1 = min.(max.(1/alpha, 1 ./ delta_1), alpha)
        Delta_2 = min.(max.(1/alpha, 1 ./ delta_2), alpha)

        # Build the new LP
        D_1 = sparse(LinearAlgebra.I, length(Delta_1),length(Delta_1)).-0.0
        D_1[diagind(D_1)] = Delta_1
        # Print diagonal elements of D_1 which are different from 1.1
        println("D_1 lambda_min: ", minimum(D_1[diagind(D_1)]))
        println("D_1 lambda_max: ", maximum(D_1[diagind(D_1)]))
        # M_Delta_P_LU = sparse(LinearAlgebra.I, length(Delta_P_lu),length(Delta_P_lu)).-0.0
        # M_Delta_P_LU[diagind(M_Delta_P_LU)] = Delta_P_lu
        # println("M_Delta_P_LU: ", M_Delta_P_LU)
        D_2 = sparse(LinearAlgebra.I, length(Delta_2),length(Delta_2)).-0.0
        D_2_inv = sparse(LinearAlgebra.I, length(Delta_2),length(Delta_2)).-0.0
        D_2[diagind(D_2)] = Delta_2
        D_2_inv[diagind(D_2_inv)] = 1 ./ Delta_2
        println("D_2 lambda_min: ", minimum(D_2[diagind(D_2)]))
        println("D_2 lambda_max: ", maximum(D_2[diagind(D_2)]))

        b_bar = D_1 * b_bar  # Ignore for now
        l_bar = D_2_inv * l_bar # D_2 inverse is 1/D_2
        u_bar = D_2_inv * u_bar # D_2 inverse is 1/D_2
        c_bar = D_2 * c_bar # Ignore for now
        A_bar = D_1 * A_bar * D_2
        # println("b_bar: ", b_bar)
        # println("l_bar: ", l_bar)
        # println("u_bar: ", u_bar)
        # println("c_bar: ", c_bar)
        # println("A_bar: ", A_bar)

        lp = cuPDLP.QuadraticProgrammingProblem(
            l_bar,
            u_bar,
            lp.objective_matrix,
            c_bar,
            0,
            A_bar,
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
            iterative_tol,
            remaining_time
        )
        t_pdlp_k = time() - t_start_k

        # KKT de-normalizer update
        b_kkt = copy(b_bar)
        c_kkt = copy(c_bar)
        l_kkt = copy(l_bar)
        u_kkt = copy(u_bar)

        # Retrieve the solution to the original problem
        x_k =  x_k + D_2 * output.primal_solution
        y_k =  y_k + D_1 * output.dual_solution 
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
        push!(out["termination_reason"], output.termination_reason)
        push!(out["blackbox_time"], t_pdlp_k)
        push!(out["primal_objective"], convergence_info.primal_objective)
        push!(out["dual_objective"], convergence_info.dual_objective)
        push!(out["l_inf_primal_residual"], convergence_info.l_inf_primal_residual)
        push!(out["l_inf_dual_residual"], convergence_info.l_inf_dual_residual)
        push!(out["relative_l_inf_primal_residual"], convergence_info.relative_l_inf_primal_residual)
        push!(out["relative_l_inf_dual_residual"], convergence_info.relative_l_inf_dual_residual)
        push!(out["relative_optimality_gap"], convergence_info.relative_optimality_gap)
        push!(out["D1_condition_number"], maximum(Delta_1)/minimum(Delta_1)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        push!(out["D2_condition_number"], maximum(Delta_2)/minimum(Delta_2)) # Normal matrix: cond_number = |delta_max|/|delta_min|
        k += 1
    end
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
# output_path = output_dir * "/" * instance_name * "_out_k"*string(output["last_iteration"])*".json"
# open(output_path, "w") do io
#     write(io, JSON3.write(output, allow_inf = true))
# end




function main()

    

    # instance_name = "2club200v15p5scn.mps.gz"
    # instance_dir = "./MIPLIB/"
    instance_dir = "/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/"
    # instance_path = instance_dir * instance_name 
    output_dir = "./MIPLIB_output/test2"
    tol_it_ref = 1e-3
    tol_objective = 1e-8
    time_sec_limit = 600 # 10 mins per instance
    max_iter_limit = 1e3 # Infinite iterations
    # Read all the files in the instance directory
    # instance_files = readdir(instance_dir)

    alphas = [1.1,1e1,1e2]#[1e3, 1e4, 1e5]#[1+1e-3,1+1e-2, 1+1e-1,1.5]

    # Instances
    instances = Dict(
        "tiny_instances"=>String[ # "app1-2",
            "ns1456591",  "graph20-80-1rand", "blp-ic98", "piperout-d20", "ns1828997", 
            "neos-4292145-piako", "neos-960392", "d20200", "mushroom-best", "bppc6-02", "neos-1354092", 
            "neos-933638", "neos-4300652-rahue", "n2seq36q", "bppc6-06", "neos-933966", "ns1430538", 
            "neos-5195221-niemur", "neos-5193246-nerang", "germanrr", "ger50-17-trans-dfn-3t", 
            "ger50-17-trans-pop-3t", "neos-5196530-nuhaka", "neos-5266653-tugela", "stockholm", "neos-953928", 
            "dws008-03", "neos-1122047", "eva1aprime6x6opt", "supportcase23", "cmflsp50-24-8-8", "sorrell7", 
            "physiciansched5-3", "bab5", "pb-grow22", "gmut-76-40", "opm2-z8-s0", "neos-913984", "mzzv42z", 
            "neos-498623", "sct5", "ns930473", "iis-hc-cov", "neos-4954274-beardy", "neos-824661", "reblock420",
             "supportcase37", "chromaticindex512-7", "fhnw-binschedule2", "mzzv11", "neos-5013590-toitoi", 
             "neos-5188808-nattai", "brazil3", "t1722", "dws012-01", "neos-1171448", "leo1", "ci-s4", "neos-826224",
              "cmflsp40-24-10-7", "unitcal_7", "neos-4359986-taipa", "satellites2-60-fs", "shipsched", 
              "fhnw-schedule-paira200", "blp-ic97", "neos-4805882-barwon", "ns1631475", "neos-3372571-onahau",
               "neos-1593097", "rmatr200-p5", "neos-827175", "30n20b8", "sct32", "neos-932721", 
               "lr1dr04vc05v17a-t360", "ns1856153", "sct1", "rmatr200-p10", "2club200v15p5scn", "fiball",
                "supportcase40", "neos-950242", "v150d30-2hopcds", "momentum1", "ex1010-pi", "neos-578379", "neos-738098", "ns1830653"
            ],
        # "small_instances"=>String[
        #     "30n20b8", "hgms30", "var-smallemery-m6j6", "nursesched-medium-hint03", 
        #     "neos-2629914-sudost",
        #     "neos-4966258-blicks", "neos-3755335-nizao", "neos-3695882-vesdre", "neos6",
        #     "cmflsp40-24-10-7", 
        #     "triptim7", "neos-2746589-doon", "reblock420",
        #     "neos-872648", "neos-4760493-puerua", 
        #     "fhnw-schedule-pairb200", "sct1",
        #     "t1717", "iis-hc-cov", 
        #     "gmut-75-50", 
        #     "t1722", "ex1010-pi",
        #     "neos-5221106-oparau", "neos-1354092", "neos-827175",
        #      "radiationm40-10-02",
        #     "nw04", "neos-4359986-taipa", "neos-960392", "map18", "neos-932721",
        #     "gmut-76-40",
        # ],
        # "medium_instances"=>String[
        #     "ds-big", "neos-4647030-tutaki", "neos-5129192-manaia", "graph40-80-1rand",
        #     "neos-5049753-cuanza", "seqsolve1", "neos-5102383-irwell", "bab6",
        #     "neos-5123665-limmat", "shs1014", "shs1042", "wnq-n100-mw99-14",
        #     "fhnw-binschedule0", "fhnw-binschedule1", "neos-4321076-ruwer",
        #     "physiciansched3-3", "neos-5079731-flyers", "neos-3322547-alsek",
        #     "neos-4647027-thurso", "ns1644855", "datt256", "kosova1",
        #     "neos-4533806-waima", "neos-4647032-veleka", "z26", "neos-5118851-kowhai",
        #     "neos-4972437-bojana", "hgms62", "in", "zeil",
        # ]
    )
    

    # Iterate over the instances (test version)
    # for instance_name in reverse(instance_files[1:end-1])# ["ci-s4.mps.gz"]# reverse(instance_files)#[15:end]
    for instance_size in keys(instances)
        for instance_name in instances[instance_size]
            instance_name=instance_name*".mps.gz"
            for alpha in alphas
                for max_iter_limit in [0,max_iter_limit]
                    println("alhpa/max_iter: ", (alpha, max_iter_limit))
                    
                    println("-----------------------------------")
                    println("Instance: ", instance_name, " max_iter_limit: ", max_iter_limit)
                    instance_path = instance_dir * instance_name
                    if max_iter_limit > 0
                        output = iterative_refinement(instance_path, tol_it_ref, tol_objective, time_sec_limit, max_iter_limit, alpha)
                    elseif alpha == alphas[1] # Baseline
                        output = iterative_refinement(instance_path, tol_objective, tol_objective, time_sec_limit, max_iter_limit)
                    else
                        continue
                    end
                    println(output)

                    # Save the output in .json format
                    output_path = output_dir * "/" * instance_size * "/" *  instance_name * "_output_k"*string(output["last_iteration"])*"_alpha_"*string(alpha)*"_nocorrection.json"
                    open(output_path, "w") do io
                        write(io, JSON3.write(output, allow_inf = true))
                    end
                end
            end
        end
    end
end

main()