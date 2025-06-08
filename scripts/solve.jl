import ArgParse
import GZip
import JSON3

# MINE
using NPZ

# import cuPDLP
include("/nfs/home2/nacevedo/RA/cuPDLP.jl/src/cuPDLP.jl")

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
    ir_instead_restart,
    ir_type::String="scalar",
    ir_iteration_threshold::Int64=1000,
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

        # IR instead restart
        output::cuPDLP.SaddlePointOutput = cuPDLP.optimize(parameters, lp, ir_instead_restart, ir_type, ir_iteration_threshold)
    
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

        @info "Saved all the others. Now saving my iterates..."

        # @info "primal iterates", output.primal_dual_iterates[1]
        primal_iterates = output.primal_dual_iterates[1]#zeros(size(output.primal_dual_iterates[1])) 
        # copyto!(primal_iterates, output.primal_dual_iterates[1])
        # @info "primal iterates 2", primal_iterates
        dual_iterates = output.primal_dual_iterates[2]
        # @info "4",output.primal_dual_iterates[4]
        # @info "5",output.primal_dual_iterates[5]
        pdhg_behavior = Dict(
            "primal_iterates_sample"=>primal_iterates,
            "dual_iterates_sample"=>dual_iterates,
            "primal_near_bounds_number"=>output.primal_dual_iterates[3],
            "primal_iterates_magnitudes"=>output.primal_dual_iterates[4],
            "primal_iterates_cosines"=>output.primal_dual_iterates[5],
            
        )
        # @info "pdhg behavior"
        # @info pdhg_behavior
        # primal_iterates_path = joinpath(output_dir, instance_name * "_primal_iterates.txt")
        # write_vector_to_file(primal_iterates_path, output.primal_dual_iterates[1])
        
        # dual_iterates_path = joinpath(output_dir, instance_name * "_dual_iterates.txt")
        # write_vector_to_file(dual_iterates_path, output.primal_dual_iterates[2])

        pdhg_behavior_path = joinpath(output_dir, instance_name * "_pdhg_behavior.json")
        open(pdhg_behavior_path, "w") do f
            JSON3.write(f, pdhg_behavior)
            println(f) # Add a newline for better readability
        end

        # write_vector_to_file(pdhg_behavior_path, JSON3.write(pdhg_behavior, allow_inf = true))
        


        
        @info "Done saving my iterates :)"
        # @info "See primal iterates on "*primal_iterates_path
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
        # required = true
        default = "instance/house_k0.01_d0.0001.mps"

        "--output_directory"
        help = "The directory for output files."
        arg_type = String
        # required = true
        default = "output/PDHG"

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-4

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0
    end

    return ArgParse.parse_args(arg_parse)
end


function main(instance_path,output_directory,tolerance, time_sec_limit, ir_instead_restart=false, ir_type::String="scalar", ir_iteration_threshold::Int64=1000)
    # parsed_args = parse_command_line()
    # instance_path = parsed_args["instance_path"]
    println("Instance path: ", instance_path)
    # tolerance = parsed_args["tolerance"]
    # time_sec_limit = parsed_args["time_sec_limit"]
    # output_directory = parsed_args["output_directory"]

    println("Trying to read instance from ", instance_path)
    lp = cuPDLP.qps_reader_to_standard_form(instance_path)
    println("Instance read successfully and transformed to 'standard form'")

    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(lp);
    redirect_stdout(oldstd)

    print("Passed through warm up")

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
        optimality_norm = cuPDLP.L_INF,#  L2,L_INF
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
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
        1,
        termination_params,
        restart_params,
        cuPDLP.AdaptiveStepsizeParams(0.3,0.6),  
    )

    solve_instance_and_output(
        params,
        output_directory,
        instance_path,
        ir_instead_restart,
        ir_type,
        ir_iteration_threshold,
    )

end

# main()
