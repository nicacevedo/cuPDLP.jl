using JSON3

include("iterative_refinement.jl")



# Read the command line
args = parse_command_line()
instance_dir = args["instance_path"]
instance_name = basename(instance_dir)
instance_name = replace(instance_name, r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "") # delete extension from instance_name
output_dir = args["output_directory"]
tol_it_ref = args["iter_tolerance"]
tol_objective = args["obj_tolerance"]
time_sec_limit = args["time_sec_limit"]
max_iter = args["max_iter"]



# Batch version
println("-----------------------------------")
println("Instance: ", instance_name, " max_iter: ", max_iter)
# instance_path = instance_dir * instance_name
# println("Instance path: ", )
if max_iter == 0
    output = iterative_refinement(instance_dir, tol_objective, tol_objective, time_sec_limit, max_iter)
else
    output = iterative_refinement(instance_dir, tol_it_ref, tol_objective, time_sec_limit, max_iter)
end

# Save the output in .json format
output_path = output_dir * "/" * instance_name * "_out_k"*string(output["final iteration"])*".json"
open(output_path, "w") do io
    write(io, JSON3.write(output, allow_inf = true))
end


# # instance_name = "2club200v15p5scn.mps.gz"
# instance_dir = "./MIPLIB/"
# # instance_path = instance_dir * instance_name 
# output_dir = "./MIPLIB_output/Matrix_output/"
# tol_it_ref = 1e-3
# tol_objective = 1e-8
# time_sec_limit = 3600 
# max_iter = 1e3
# # Read all the files in the instance directory
# instance_files = readdir(instance_dir)

# # Iterate over the instances (test version)
# # for instance_name in reverse(instance_files[1:end-1])# ["ci-s4.mps.gz"]# reverse(instance_files)#[15:end]
# for instance_name in instance_files[3:end]

#     for max_iter in [0,max_iter]
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
