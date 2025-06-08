
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

# GPU conditional masking for large GPU arrays (in paralell)
function filter_positive_and_nonpositive(x, condition_vector)
    mask1 = condition_vector .> 0  # Boolean mask on GPU
    mask2 = condition_vector .<= 0
    indices1 = cumsum(Int.(mask1))  # Cumulative sum for indexing (positive)
    indices2 = cumsum(Int.(mask2))  # Cumulative sum for indexing (nonpositive)
    total1 = sum(Int.(mask1))  # Total count of positive values
    total2 = sum(Int.(mask2))  # Total count of nonpositive values
    out1 = CUDA.zeros(eltype(x), total1)  # Allocate output
    out2 = CUDA.zeros(eltype(x), total2)  # Allocate output

    config = launch_configuration(kernel.fun);
    threads = Base.min(arrayLength, config.threads);
    blocks = cld(arrayLength, threads);

    @cuda threads=length(x) kernel_filter!(x, mask1, indices1, out1)
    @cuda threads=length(x) kernel_filter!(x, mask2, indices2, out2)

    return out1, out2
end

function kernel_filter!(x, mask, indices, out)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(x) && mask[i]
        out[indices[i]] = x[i]
    end
    return nothing
end


# MY aux function
function cu_quad_to_quad_matrix(A_gpu::CUDA.CUSPARSE.CuSparseMatrixCSR)
    row_ptr = Array(A_gpu.rowPtr)
    col_ind = Array(A_gpu.colVal)
    values = Array(A_gpu.nzVal)

    # # 1-based indexing
    # row_ptr .+= 1
    # col_ind .+= 1
    
    # Convert CSR to CSC
    return SparseMatrixCSC(
        reverse(A_gpu.dims)...,  # Dimensions
        row_ptr, col_ind, values         # CSR data
    )'
end

# Triple matrix multiplication on CUDA
function triple_cudaMM(
    D1::CUDA.CUSPARSE.CuSparseMatrixCSR,
    A::CUDA.CUSPARSE.CuSparseMatrixCSR,
    D2::CUDA.CUSPARSE.CuSparseMatrixCSR,
)
    A_copy = deepcopy(A) # Copy: maybe erase it
    CUDA.CUSPARSE.gemm!(
             'N',   # 1st matrix is not transpose
             'N',   # 2nd matrix is not transpose
             1,     # 1*A*B
             D1, # A
             A_copy,     # B
             0,                            
             A_copy,     # C
             'O',   # Secret
        # CUDA.CUSPARSE.CUSPARSE_SPMM_CSR_ALG2, # determinstic algorithm(?)
        ) 
    CUDA.CUSPARSE.gemm!(
             'N',   # 1st matrix is not transpose
             'N',   # 2nd matrix is not transpose
             1,     # 1*A*B
             A_copy, # A
             D2,     # B
             0,                            
             A_copy,     # C
             'O',   # Secret
        ) 
    return A_copy
end

# Matrix-vector multiplication on CUDA
    # CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix, delta_primal, 0, delta_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    # algo 2 is deterministic / algo 1 is random
function cudaMV(
    A::CUDA.CUSPARSE.CuSparseMatrixCSR,
    x::CuVector,
)
    v = CuVector{Float64}(CUDA.zeros(size(A)[1]))
    CUDA.CUSPARSE.mv!(
        'N', 
        1, 
        A, 
        x, 
        0, 
        v, 
        'O', 
        CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2
    )
    return v
end


"""
    gpu_transpose_csr(A::CuSparseMatrixCSR)

Transpose a CuSparseMatrixCSR on the GPU, returning the result as a new CuSparseMatrixCSR.
No CPU communication is performed.
"""
function gpu_transpose_csr(A::CUDA.CUSPARSE.CuSparseMatrixCSR)
    m, n = size(A)
    nnz = A.nnz
    T = eltype(A)

    # Allocate output arrays for the CSC representation (which is the transpose in CSR)
    d_cscVal = similar(A.nzVal, nnz)
    d_cscRowInd = similar(A.rowPtr, nnz)
    d_cscColPtr = similar(A.colVal, n + 1)

    # Perform CSR to CSC conversion (which is equivalent to transposing)
    CUDA.CUSPARSE.csr2csc!(
        m, n, nnz,
        A.nzVal, A.rowPtr, A.colVal,
        d_cscVal, d_cscColPtr, d_cscRowInd,
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE
    )

    # Construct the transposed matrix in CSR format (by reinterpreting CSC as CSR)
    # Note: rowptr <-> colptr, colind <-> rowind
    return CUDA.CUSPARSE.CuSparseMatrixCSR(
        d_cscVal,
        d_cscColPtr,   # now rowptr for the transposed matrix
        d_cscRowInd,   # now colind for the transposed matrix
        n, m           # note the swapped dimensions
    )
end


# mutable struct QuadraticProgrammingProblem
#     """
#     The vector of variable lower bounds.
#     """
#     variable_lower_bound::Vector{Float64}
  
#     """
#     The vector of variable upper bounds.
#     """
#     variable_upper_bound::Vector{Float64}
  
#     """
#     The symmetric and positive semidefinite matrix that defines the quadratic
#     term in the objective.
#     """
#     objective_matrix::SparseMatrixCSC{Float64,Int64}
  
#     """
#     The linear coefficients of the objective function.
#     """
#     objective_vector::Vector{Float64}
  
#     """
#     The constant term of the objective function.
#     """
#     objective_constant::Float64
  
#     """
#     The matrix of coefficients in the linear constraints.
#     """
#     constraint_matrix::SparseMatrixCSC{Float64,Int64}
  
#     """
#     The vector of right-hand side values in the linear constraints.
#     """
#     right_hand_side::Vector{Float64}
  
#     """
#     The number of equalities in the problem. This value splits the rows of the
#     constraint_matrix between the equality and inequality parts.
#     """
#     num_equalities::Int64
#   end
  

# mutable struct CuLinearProgrammingProblem
#     num_variables::Int64
#     num_constraints::Int64
#     variable_lower_bound::CuVector{Float64}
#     variable_upper_bound::CuVector{Float64}
#     isfinite_variable_lower_bound::CuVector{Bool}
#     isfinite_variable_upper_bound::CuVector{Bool}
#     objective_vector::CuVector{Float64}
#     objective_constant::Float64
#     constraint_matrix::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
#     constraint_matrix_t::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
#     right_hand_side::CuVector{Float64}
#     num_equalities::Int64
# end

# mutable struct CuScaledQpProblem
#     original_qp::CuLinearProgrammingProblem
#     scaled_qp::CuLinearProgrammingProblem
#     constraint_rescaling::CuVector{Float64}
#     variable_rescaling::CuVector{Float64}
# end



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
    # println("NUMBER OF EQUALITIES: ", lp.num_equalities)
    n_inequalities = size(A, 1) - lp.num_equalities # num of slacks to add
    if n_inequalities > 0 
        # Identify matrix to add slack variables
        I = sparse(LinearAlgebra.I, n_inequalities, n_inequalities)

        # Add n_eq columns and rows of zeros to I
        Z_I = [
            spzeros(lp.num_equalities, n_inequalities); 
            -I
            ]
        # print("\nType of the A matrix: \n", typeof([A Z_I]))
        A_t = CUDA.CUSPARSE.CuSparseMatrixCSR([A Z_I]') # this one first!!
        A =  CUDA.CUSPARSE.CuSparseMatrixCSR([A Z_I]) 

        # Add slack variables to the objective function
        c = lp.objective_vector
        c = [c; CuVector{Float64}(spzeros(n_inequalities))]

        # Add slack variables to the upper bound
        u = lp.variable_upper_bound
        u = [u; CuVector{Float64}(Inf*(spzeros(n_inequalities).+1))]

        # Add slack variables to the lower bound
        l = lp.variable_lower_bound
        l = [l; CuVector{Float64}(spzeros(n_inequalities))]

        # # Update the LP
        # lp.constraint_matrix = A
        # lp.objective_vector = c
        # lp.variable_upper_bound = u
        # lp.variable_lower_bound = l
        # lp.num_equalities = size(A, 1)
        # # lp.objective_matrix =  sparse(Int64[], Int64[], Float64[], size(c, 1), size(c, 1))

        lp_new = CuLinearProgrammingProblem(
            size(c, 1),
            lp.num_constraints,
            l,
            u,
            isfinite.(l),
            isfinite.(u),
            c,
            lp.objective_constant,
            A,
            A_t,
            lp.right_hand_side,
            size(A, 1)
        )
    else
        lp_new = lp
    end
    return lp_new
end



function cuPDLP_KKT_buffer(lp, x_k, y_k)

    # Calculate the KKT error of the problem
    primal_size = size(x_k)[1]
    dual_size = size(y_k)[1]
    num_eq = lp.num_equalities

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
        

function compute_current_KKT(
    lp::Union{QuadraticProgrammingProblem, CuQuadraticProgrammingProblem, CuLinearProgrammingProblem},
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    qp_cache::Any,
)
    # Preliminars
    buffer_kkt = cuPDLP.cuPDLP_KKT_buffer(lp, current_primal_solution, current_dual_solution)
    # qp_cache = cuPDLP.cached_quadratic_program_info(lp) # As in "optimize" function (line 462)
    if lp isa CuLinearProgrammingProblem
        cuLPP = lp
    elseif lp isa QuadraticProgrammingProblem || lp isa CuQuadraticProgrammingProblem
        cuLPP = cuPDLP.qp_cpu_to_gpu(lp) # Already a cuLPP?
    end
    convergence_info = cuPDLP.compute_convergence_information(
        cuLPP, #        problem::CuLinearProgrammingProblem,
        qp_cache,#      qp_cache::CachedQuadraticProgramInfo,
        CuVector{Float64}(current_primal_solution),#    primal_iterate::CuVector{Float64},
        CuVector{Float64}(current_dual_solution),#    dual_iterate::CuVector{Float64},
        1.0, #          eps_ratio::Float64,
        cuPDLP.POINT_TYPE_AVERAGE_ITERATE,# candidate_type::PointType,
        CuVector{Float64}(cudaMV(lp.constraint_matrix,current_primal_solution)),#      primal_product::CuVector{Float64},
        CuVector{Float64}(lp.objective_vector - cudaMV(CUDA.CUSPARSE.CuSparseMatrixCSR(lp.constraint_matrix'),current_dual_solution)),#  primal_gradient::CuVector{Float64},
        buffer_kkt  #   buffer_kkt::BufferKKTState,
    )

    # KKT computation 
    current_kkt =norm([
        convergence_info.l_inf_primal_residual,
        convergence_info.l_inf_dual_residual,
        convergence_info.primal_objective - convergence_info.dual_objective
    ],Inf)
    current_rel_kkt = norm([
        convergence_info.relative_l_inf_primal_residual,
        convergence_info.relative_l_inf_dual_residual,
        convergence_info.relative_optimality_gap
    ],Inf)
    println("Relative KKT's:")
    println(convergence_info.relative_l_inf_primal_residual)
    println(convergence_info.relative_l_inf_dual_residual)
    println(convergence_info.relative_optimality_gap)

    return current_kkt, current_rel_kkt, convergence_info
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

function warm_up(lp::Union{QuadraticProgrammingProblem, CuQuadraticProgrammingProblem})
    println("WARM UP STARTING...")
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

    # println("Warm up optimizing...")
    cuPDLP.optimize(params_warmup, lp);
    # println("done")
end


function call_cuPDLP(
    qp::Union{QuadraticProgrammingProblem, CuQuadraticProgrammingProblem},
    # lp::CuLinearProgrammingProblem,
    tolerance::Float64,
    time_sec_limit::Union{Float64, Int64},
    iteration_limit::Union{Float64, Int64, Int32}
)

    # Initialization
    println("Initializing call_cuPDLP...")
    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(qp);
    redirect_stdout(oldstd)
    # println("done")

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
    println("Starting to optimize... (tol="*string(tolerance)*")")
    output = cuPDLP.optimize(
        params,
        qp,
        false, # IR instead of restart
        )
    # println("done")
    
    return output
end


# Input:    Violations
# Output:   New solution from scaled/shifted problem
function scalar_refinement(
    lp::Union{QuadraticProgrammingProblem, CuLinearProgrammingProblem},
    x_k::CuVector{Float64},
    y_k::CuVector{Float64},
    # problem_parameters::ProblemParams
    # shifted_parameters::ProblemParamse
    tolerance::Float64,
    alpha::Float64,
    Delta_P::Union{Float64, Int64},
    Delta_D::Union{Float64, Int64},
    iteration_limit::Int32=typemax(Int32),
    )

    # shifted parameters
    # println(typeof(lp.constraint_matrix))
    # println(size(lp.constraint_matrix))
    # println(size(x_k))
    b_bar = lp.right_hand_side - cudaMV(lp.constraint_matrix, x_k)# - lp.constraint_matrix * x_k
    l_bar = lp.variable_lower_bound - x_k
    u_bar = lp.variable_upper_bound - x_k
    c_bar = lp.objective_vector - cudaMV(lp.constraint_matrix_t, y_k) # - lp.constraint_matrix_t * y_k

    println("Computing constraint violations...")
    # Maximum primal/dual violation
    delta_P = maximum([
        maximum(abs.(b_bar)),
        maximum(l_bar),
        maximum(-u_bar)
    ])

    # println("Filtering c_bar in terms of condition...")
    c_bar_pos = Base.getindex(c_bar, Base.findall(x_k .> (lp.variable_lower_bound + lp.variable_upper_bound) / 2))
    c_bar_neg = Base.getindex(c_bar, Base.findall(x_k .<= (lp.variable_lower_bound + lp.variable_upper_bound) / 2))
    # println("done with that one")
    delta_D = maximum([
        # c_bar[x_k .> (lp.variable_lower_bound + lp.variable_upper_bound) / 2] ;
        maximum(c_bar_pos), 
        # -c_bar[x_k .<= (lp.variable_lower_bound + lp.variable_upper_bound) / 2]
        maximum(-c_bar_neg)
        ], 
        init=0 # at least 0
    )

    # Scaling factor
    # println("Computing scaling factors...")
    Delta_P = max(min( 1 / delta_P , alpha*Delta_P ), 1) 
    Delta_D = max(min( 1 / delta_D , alpha*Delta_D ), 1) 
    

    # Scaling the problem
    b_bar = b_bar * Delta_P
    l_bar = l_bar * Delta_P
    u_bar = u_bar * Delta_P
    c_bar = c_bar * Delta_D

    # Solve scaled-shifted problem
    # println("Creating the scaled programming problem...")

    # scaled_qp = cuPDLP.QuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     Vector{Float64}(l_bar),
    #     Vector{Float64}(u_bar),
    #     spzeros(length(x_k), length(x_k)), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     Vector{Float64}(c_bar),
    #     lp.objective_constant,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     cu_quad_to_quad_matrix(lp.constraint_matrix),
    #     Vector{Float64}(b_bar),
    #     lp.num_equalities
    # )
    # original_qp = cuPDLP.QuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     Vector{Float64}(lp.variable_lower_bound),
    #     Vector{Float64}(lp.variable_upper_bound),
    #     spzeros(length(x_k), length(x_k)), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     Vector{Float64}(lp.objective_vector),
    #     lp.objective_constant,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     cu_quad_to_quad_matrix(lp.constraint_matrix),
    #     Vector{Float64}(lp.right_hand_side),
    #     lp.num_equalities
    # )
    scaled_qp = cuPDLP.CuQuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
        l_bar,
        u_bar,
        CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(length(x_k), length(x_k))), # objective matrix. I guess it is Q: x'Qx + c'x + b
        c_bar,
        lp.objective_constant,  # Objective constant. I guess it is b: x'Qx + c'x + b
        lp.constraint_matrix,
        b_bar,
        lp.num_equalities
    )

    ### KKT computation ###
    # original_qp = cuPDLP.CuQuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     lp.variable_lower_bound,
    #     lp.variable_upper_bound,
    #     CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(length(x_k), length(x_k))), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     lp.objective_vector,
    #     lp.objective_constant,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     lp.constraint_matrix,
    #     lp.right_hand_side,
    #     lp.num_equalities
    # )
    # qp_cache = cached_quadratic_program_info(original_qp)
    # println("done")

    # println("Calling cuPDLP to optimize... (tol="*string(tolerance)*")")
    # Call cuPDLP to optimize new problem
    output = call_cuPDLP(
        scaled_qp,
        tolerance,
        600,#time_sec_limit,
        iteration_limit, #typemax(Int32)
    )

    post_ir_kkt = CUDA.norm(
        [output.iteration_stats[end].convergence_information[end].relative_l_inf_primal_residual,
        output.iteration_stats[end].convergence_information[end].relative_l_inf_dual_residual,
        output.iteration_stats[end].convergence_information[end].relative_optimality_gap],
        Inf
    )
    if post_ir_kkt <= 1e-2 # if enough progress
        # Update refined solution
        x_k = x_k + output.primal_solution / Delta_P
        y_k = y_k + output.dual_solution / Delta_D
    else
        x_k = nothing 
        y_k = nothing
    end


    @info "Post IR relative KKT: " post_ir_kkt



    ### KKT change check ###
    # current_kkt, current_rel_kkt = compute_current_KKT(
    #     lp,
    #     x_k,
    #     y_k,
    #     qp_cache
    # )

    # println("After IR KKT: ", current_kkt)
    # println("After IR rel KKT: ", current_rel_kkt)


    return x_k, y_k#, current_kkt, current_rel_kkt
end

# Input:    Violations
# Output:   New solution from scaled/shifted problem
function matrix_refinement(
    lp::Union{QuadraticProgrammingProblem, CuLinearProgrammingProblem},
    x_k::CuVector{Float64},
    y_k::CuVector{Float64},
    # problem_parameters::ProblemParams
    # shifted_parameters::ProblemParamse
    tolerance::Float64,
    alpha::Float64,
    Delta_1::Union{CuVector{Float64}, CuVector{Int64}},
    Delta_2::Union{CuVector{Float64}, CuVector{Int64}},
    Delta_3::Union{CuVector{Float64}, CuVector{Int64}},
    convergence_info::Any,
    )

    println("Entering the matrix refinement...")
    # Parameters to add
    scaling_type = "D2_D3_adaptive_v7"
    scaling_bound = 1e12

    # shifted parameters
    b_bar = lp.right_hand_side - cudaMV(lp.constraint_matrix, x_k)# - lp.constraint_matrix * x_k
    l_bar = lp.variable_lower_bound - x_k
    u_bar = lp.variable_upper_bound - x_k
    c_bar = lp.objective_vector - cudaMV(lp.constraint_matrix_t, y_k) # - lp.constraint_matrix_t * y_k


    # Primal eq violation
    delta_1 = abs.(b_bar)

    # Box constraint violation
    delta_2 = max.( # l/u constraint violation. To use in D2: D2 (u - x^*) >= x >= D2 (l - x^*)
    CUDA.zeros(size(l_bar)), # Safety card for only negative violations (which is the desired case)
    l_bar,
    # l_bar .* (1 + sqrt(l_finite'*l_finite)), # rel. correction (mine) [delte?]
    -u_bar,
    # -u_bar .* (1 + sqrt(u_finite'*u_finite)), # rel. correction (mine) [delte?]
    )   

    c_bar_sign = CuVector{Float64}(c_bar)
    c_bar_sign[x_k .<= (lp.variable_lower_bound + lp.variable_upper_bound) / 2] .= -CuVector{Float64}(c_bar[x_k .<= (lp.variable_lower_bound + lp.variable_upper_bound) / 2])
    # c_bar_pos = Base.getindex(c_bar, Base.findall(x_k .> (lp.variable_lower_bound + lp.variable_upper_bound) / 2))
    # c_bar_neg = Base.getindex(c_bar, Base.findall(x_k .<= (lp.variable_lower_bound + lp.variable_upper_bound) / 2))
    delta_3 = max.(
        CUDA.zeros(size(c_bar)), # In case there is no positive c_bar_sign
        c_bar_sign
    ) 

    # Scaling matrix
    Delta_1 = min.(1 ./delta_1, alpha * Delta_1 ) 


    if scaling_type == "D2_D3_adaptive_v7"
        # Adaptive method for nonconvergent methods
        primal_res = convergence_info.relative_l_inf_primal_residual
        dual_res = convergence_info.relative_l_inf_dual_residual

        if maximum(delta_1) > 0# If not zero violation
            # delta correction
            n_delta_1 = floor(log10(maximum(delta_1)))
            delta_1 = delta_1 * 10^(-n_delta_1)
            Delta_1 = max.( delta_1, alpha ) # indep of alpha
        else 
            Delta_1 = CUDA.ones(length(Delta_1))
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
                Delta_2 = CUDA.ones(length(Delta_2))
                Delta_3 = CUDA.ones(length(Delta_3))
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
                Delta_2 = CUDA.ones(length(Delta_2))
                Delta_3 = CUDA.ones(length(Delta_3)) 
            end
        else 
            # Identity matrices
            Delta_2 = CUDA.ones(length(Delta_2))
            Delta_3 = CUDA.ones(length(Delta_3)) 
        end
    end

    # println("Bounding the value of the scaling factors by: ", scaling_bound)
    # Scaling bound check
    Delta_1 = max.( min.(Delta_1, scaling_bound), 1/scaling_bound )
    Delta_2 = max.( min.(Delta_2, scaling_bound), 1/scaling_bound )
    Delta_3 = max.( min.(Delta_3, scaling_bound), 1/scaling_bound )

    # Create scaling matrices
    # S = CuSparseMatrixCSC(n, n, I, 1:1:n+1, V)
    # D_1 = CUDA.CUSPARSE.CuSparseMatrixCSC(CuVector(1:(length(Delta_1)+1)), CuVector(1:length(Delta_1)),  Delta_1, (length(Delta_1), length(Delta_1)))
    # D_2 = CUDA.CUSPARSE.CuSparseMatrixCSC(CuVector(1:(length(Delta_2)+1)), CuVector(1:length(Delta_2)),  Delta_2, (length(Delta_2), length(Delta_2)))
    # D_3 = CUDA.CUSPARSE.CuSparseMatrixCSC(CuVector(1:(length(Delta_3)+1)), CuVector(1:length(Delta_3)),  Delta_3, (length(Delta_3), length(Delta_3))) # inverse of D_2
    D_1 = CUDA.CUSPARSE.CuSparseMatrixCSR(spdiagm(Delta_1))
    D_2 = CUDA.CUSPARSE.CuSparseMatrixCSR(spdiagm(Delta_3))
    D_3 = CUDA.CUSPARSE.CuSparseMatrixCSR(spdiagm(Delta_3))



    # Scaling the problem
    b_bar = cudaMV(D_1, b_bar) # D_1 * b_bar # Must have this scaling
    l_bar = cudaMV(D_2, l_bar) # D_2 * l_bar 
    u_bar = cudaMV(D_2, u_bar) # D_2 * u_bar 
    c_bar = cudaMV(D_3, c_bar) # D_3 * c_bar   

    # Testing MM
    # A_bar = D_1 * lp.constraint_matrix * D_3
    A_bar = triple_cudaMM(D_1, lp.constraint_matrix, D_3)
    # A_bar = deepcopy(lp.constraint_matrix)
    # A : mxn => D1 mxm / D3 nxn


    # sub_mult = lp.constraint_matrix * D_3
    # A_bar = D_1 * sub_mult#(D_1 * CUDA.CUSPARSE.CuSparseMatrixCSC(lp.constraint_matrix)) * D_3

    # Solve scaled-shifted problem
    # println("Creating the scaled programming problem...")

    # scaled_qp = cuPDLP.QuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     Vector{Float64}(l_bar),
    #     Vector{Float64}(u_bar),
    #     spzeros(length(x_k), length(x_k)), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     Vector{Float64}(c_bar),
    #     0,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     cu_quad_to_quad_matrix(A_bar),
    #     Vector{Float64}(b_bar),
    #     lp.num_equalities
    # )
    # original_qp = cuPDLP.QuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     Vector{Float64}(lp.variable_lower_bound),
    #     Vector{Float64}(lp.variable_upper_bound),
    #     spzeros(length(x_k), length(x_k)), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     Vector{Float64}(lp.objective_vector),
    #     0,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     cu_quad_to_quad_matrix(lp.constraint_matrix),
    #     Vector{Float64}(lp.right_hand_side),
    #     lp.num_equalities
    # )



    scaled_qp = cuPDLP.CuQuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
        l_bar,
        u_bar,
        CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(length(x_k), length(x_k))), # objective matrix. I guess it is Q: x'Qx + c'x + b
        c_bar,
        0,  # Objective constant. I guess it is b: x'Qx + c'x + b
        A_bar, # A_bar
        b_bar,
        lp.num_equalities
    )
    original_qp = cuPDLP.CuQuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
        lp.variable_lower_bound,
        lp.variable_upper_bound,
        CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(length(x_k), length(x_k))), # objective matrix. I guess it is Q: x'Qx + c'x + b
        lp.objective_vector,
        0,  # Objective constant. I guess it is b: x'Qx + c'x + b
        lp.constraint_matrix,
        lp.right_hand_side,
        lp.num_equalities
    )

    qp_cache = cached_quadratic_program_info(original_qp)
    # println("done")

    # println("Calling cuPDLP to optimize... (tol="*string(tolerance)*")")
    # Call cuPDLP to optimize new problem
    output = call_cuPDLP(
        scaled_qp,
        tolerance,
        600,#time_sec_limit,
        typemax(Int32),#iteration_limit
    )
    # println("done")
    
    # New solutions
    # println("New solutions...")
    # println("Max sol.", maximum(CuVector{Float64}(output.primal_solution)))
    # println("Max sol.", maximum(CuVector{Float64}(output.dual_solution)))    
    # println("Min sol.", minimum(CuVector{Float64}(output.primal_solution)))
    # println("Min sol.", minimum(CuVector{Float64}(output.dual_solution)))

    x_k = x_k + cudaMV(D_3, output.primal_solution) # D_3 * output.primal_solution
    y_k = y_k + cudaMV(D_1, output.dual_solution)   # D_1 * output.dual_solution 

    current_kkt, current_rel_kkt = compute_current_KKT(
        lp,
        x_k,
        y_k,
        qp_cache
    )

    println("After IR KKT: ", current_kkt)
    println("After IR rel KKT: ", current_rel_kkt)
    # println("done")
    

    # # New violations
    # b_bar = lp.right_hand_side - lp.constraint_matrix * x_k
    # l_bar = lp.variable_lower_bound - x_k
    # u_bar = lp.variable_upper_bound - x_k
    # c_bar = lp.objective_vector - lp.constraint_matrix' * y_k
    # println("(NEW) Max b violation: ", maximum(abs.(b_bar)))
    # println("(NEW) Max l violation: ", maximum(l_bar))
    # println("(NEW) Max u violation: ", maximum(-u_bar))

    return x_k, y_k, current_kkt, current_rel_kkt
end



function iterative_refinement(
    problem::CuLinearProgrammingProblem,            # Potentially: Scaled problem of cuPDLP
    current_primal_solution::CuVector{Float64},       
    current_dual_solution::CuVector{Float64},
    # My inputs
    current_iteration_stats::Any,
    qp_cache::Any,
    tolerance::Float64,
    ir_tolerance_factor::Float64,
    alpha::Float64,
    scaling_type::String="scalar",
    scaling_name::String="scalar",
    max_iteration::Int64=1,
    iteration_limit::Int32=typemax(Int32)
    )

    ### KKT Computation (to check the method) ###
    # original_qp = cuPDLP.CuQuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     problem.variable_lower_bound,#Vector{Float64}(problem.variable_lower_bound),
    #     problem.variable_upper_bound,#Vector{Float64}(problem.variable_upper_bound),
    #     CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(length(current_primal_solution), length(current_primal_solution))),#spzeros(length(x_k), length(x_k)), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     problem.objective_vector,#Vector{Float64}(problem.objective_vector),
    #     problem.objective_constant,#problem.objective_constant,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     problem.constraint_matrix,#cu_quad_to_quad_matrix(problem.constraint_matrix),
    #     problem.right_hand_side, #Vector{Float64}(problem.right_hand_side),
    #     problem.num_equalities, #problem.num_equalities,
    # )
    # qp_cache = cached_quadratic_program_info(original_qp)
    # println("Computing initial KKT...")
    # current_kkt, current_rel_kkt, convergence_info = compute_current_KKT(
    # original_qp,#lp,
    # current_primal_solution,
    # current_dual_solution,
    # qp_cache
    # )

    
    convergence_info = current_iteration_stats.convergence_information[end]
    # KKT computation 
    # current_kkt =norm([
    #     convergence_info.l_inf_primal_residual,
    #     convergence_info.l_inf_dual_residual,
    #     convergence_info.primal_objective - convergence_info.dual_objective
    # ],Inf)
    current_rel_kkt = norm([
        convergence_info.relative_l_inf_primal_residual,
        convergence_info.relative_l_inf_dual_residual,
        convergence_info.relative_optimality_gap
    ],Inf)
    # @info "Before Initial KKT: ", current_kkt
    @info "Before Initial rel KKT: ", current_rel_kkt
    if log10(current_rel_kkt) - log10(tolerance) <= 1.6989700043360187
        return nothing
    end

    # Problem to quadratic form 
    lp = LP_to_quasi_standard_form(problem)



    # Initial solution
    # println("AE shape: ", size(problem.constraint_matrix[1:problem.num_equalities,:]))
    # println("AI shape: ", size(problem.constraint_matrix[problem.num_equalities+1:end,:]))
    # println(" x shape: ", size(current_primal_solution))

    # println("primal norm inside IR: ", norm(current_primal_solution))
    # println("dual norm inside IR:   ", norm(current_dual_solution))

    # Slack variables
    Z_I_s = cudaMV(problem.constraint_matrix, current_primal_solution) .- problem.right_hand_side # problem.constraint_matrix*current_primal_solution .- problem.right_hand_side # Zs = -(0; s) = b - Ax => s=Ax - b
    # println("Length 1: ", length(Z_I_s))
    s = Z_I_s[problem.num_equalities+1:end]
    # println("Norm of Z_I_s: ", norm(Z_I_s))
    # println("Norm of s: ", norm(s))
    # s_2 = Vector{Float64}(s)
    # println("any: ",any(s_2.<0))
    # slack_solution = CuVector{Float64}(problem.constraint_matrix[problem.num_equalities+1:end,:]*current_primal_solution - problem.right_hand_side[problem.num_equalities+1:end])  # s = AIx - bI
    x_k = [current_primal_solution; s] # (x;0) + (0;s)
    y_k = current_dual_solution

    # println("Length 1: ", length([current_primal_solution; s]))
    # println("Length 2: ", length(x_k))



    # ### Current KKT computation ### 
    # original_qp = cuPDLP.CuQuadraticProgrammingProblem( # .QuadraticProgrammingProblem( 
    #     lp.variable_lower_bound,#Vector{Float64}(lp.variable_lower_bound),
    #     lp.variable_upper_bound,#Vector{Float64}(lp.variable_upper_bound),
    #     CUDA.CUSPARSE.CuSparseMatrixCSR(spzeros(length(x_k), length(x_k))),#spzeros(length(x_k), length(x_k)), # objective matrix. I guess it is Q: x'Qx + c'x + b
    #     lp.objective_vector,#Vector{Float64}(lp.objective_vector),
    #     lp.objective_constant,#lp.objective_constant,  # Objective constant. I guess it is b: x'Qx + c'x + b
    #     lp.constraint_matrix,#cu_quad_to_quad_matrix(lp.constraint_matrix),
    #     lp.right_hand_side, #Vector{Float64}(lp.right_hand_side),
    #     lp.num_equalities, #lp.num_equalities,
    # )
    # qp_cache = cached_quadratic_program_info(original_qp)
    # # println("PRIMAL FEAS BY ME:")
    # # println("||A(x,s)-b||_2: ", norm(lp.constraint_matrix * x_k - lp.right_hand_side))
    # println("Computing initial KKT...")
    # current_kkt, current_rel_kkt, convergence_info = compute_current_KKT(
    # original_qp,#lp,
    # x_k,
    # y_k,
    # qp_cache
    # )

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
        Delta_1 = CUDA.ones(Float64, length(b)) 
        Delta_2 = CUDA.ones(Float64, length(l))
        Delta_3 = CUDA.ones(Float64, length(c))
    end

    println("Initializing the IR loop...")
    # Iterations of IR
    k = 0
    while true
        k+=1

        ir_tolerance = ir_tolerance_factor# max(min(current_rel_kkt*ir_tolerance_factor, 1e-1), 1e-3)

        # Refined solution
        if scaling_type == "scalar"
            @info "Calling scalar IR..., with tolerance: ", ir_tolerance
            # x_k, y_k, current_kkt, current_rel_kkt = scalar_refinement(
            x_k, y_k = scalar_refinement(
                lp,
                x_k,
                y_k,
                ir_tolerance,#10^(-ceil(-log10(current_rel_kkt))) * ir_tolerance_factor,
                alpha,
                Delta_P,
                Delta_D,
                iteration_limit, # iteration limit
            )
            println("done")
        elseif scaling_type == "matrix"
            @info "Calling matrix IR..., with tolerance: ", ir_tolerance
            x_k, y_k = matrix_refinement(
                lp,
                x_k,
                y_k,
                ir_tolerance,#10^(-ceil(-log10(current_rel_kkt))) * ir_tolerance_factor,
                alpha,
                Delta_1,
                Delta_2,
                Delta_3,
                convergence_info,
            )
            println("done")
        end

        if isnothing(x_k) || isnothing(y_k)
            return nothing
        end 



        # Termination criteria 
        if current_rel_kkt <= tolerance || k >= max_iteration
            new_primal_solution = x_k[1:length(current_primal_solution)]
            new_dual_solution = y_k
            new_primal_product = cudaMV(problem.constraint_matrix, new_primal_solution) # problem.constraint_matrix * new_primal_solution
            new_primal_gradient = cudaMV(problem.constraint_matrix_t, new_dual_solution) # problem.constraint_matrix'*new_dual_solution
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