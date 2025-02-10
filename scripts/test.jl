include("./solve.jl")
include("./iterative_refinement.jl")


function build_house_problem(kappa,delta)
    # Build the housing problem in .mps format
    base_mps = "NAME          PROBLEM
ROWS
 N  OBJ
 L  C1
 L  C2
COLUMNS
 Y1     OBJ      0
 Y1     C1       1
 Y1     C2      -1
 Y2     OBJ     -1
 Y2     C1      $(1/kappa)
 Y2     C2      $(1/kappa)
RHS
 RHS1   C1       1
 RHS1   C2       1
BOUNDS
 LO BND Y1      -1
 UP BND Y1       1
 LO BND Y2      -1
 UP BND Y2      $(kappa-delta)
ENDATA"
    # save string as a .mps file on "instance/house_k$(kappa)_d($delta).mps"
    instance_path = "instance/house_k$(kappa)_d$(delta).mps"
    open(instance_path, "w") do f
        write(f, base_mps)
    end
    return instance_path
end

# Apparently PDLP is minimizing, so we are solving min{-y_2} instead of max{y_2}, which gives
# us a negative primal optimal cost. 

function main()

    

    objective_tol = 1e-6
    iter_tol = 1e-3
    time_sec_limit=600
    data_source = "house_shaped"   #"MIPLIB" #
    max_iter = 1000


    # # # dummy
    # for kappa in 1:90#[1] # 90 tiny instances
    #     for delta in [1]

    # # Paper-like parameters for the housing problem
    for kappa in [0.1, 0.5, 0.99, 1] # 
        for delta in [0, 1e-3, 1] #
    #     # for delta in [1e-3] # special delta

            if data_source == "MIPLIB"
                data_size = "tiny_instances"
                if data_size == "tiny_instances"
                    MIPLIB_instances = ["ns1456591", "app1-2", "graph20-80-1rand", "blp-ic98", "piperout-d20", "ns1828997", "neos-4292145-piako", "neos-960392", "d20200", "mushroom-best", "bppc6-02", "neos-1354092", "neos-933638", "neos-4300652-rahue", "n2seq36q", "bppc6-06", "neos-933966", "ns1430538", "neos-5195221-niemur", "neos-5193246-nerang", "germanrr", "ger50-17-trans-dfn-3t", "ger50-17-trans-pop-3t", "neos-5196530-nuhaka", "neos-5266653-tugela", "stockholm", "neos-953928", "dws008-03", "neos-1122047", "eva1aprime6x6opt", "supportcase23", "cmflsp50-24-8-8", "sorrell7", "physiciansched5-3", "bab5", "pb-grow22", "gmut-76-40", "opm2-z8-s0", "neos-913984", "mzzv42z", "neos-498623", "sct5", "ns930473", "iis-hc-cov", "neos-4954274-beardy", "neos-824661", "reblock420", "supportcase37", "chromaticindex512-7", "fhnw-binschedule2", "mzzv11", "neos-5013590-toitoi", "neos-5188808-nattai", "brazil3", "t1722", "dws012-01", "neos-1171448", "leo1", "ci-s4", "neos-826224", "cmflsp40-24-10-7", "unitcal_7", "neos-4359986-taipa", "satellites2-60-fs", "shipsched", "fhnw-schedule-paira200", "blp-ic97", "neos-4805882-barwon", "ns1631475", "neos-3372571-onahau", "neos-1593097", "rmatr200-p5", "neos-827175", "30n20b8", "sct32", "neos-932721", "lr1dr04vc05v17a-t360", "ns1856153", "sct1", "rmatr200-p10", "2club200v15p5scn", "fiball", "supportcase40", "neos-950242", "v150d30-2hopcds", "momentum1", "ex1010-pi", "neos-578379", "neos-738098", "ns1830653"]
                
                "ns1456591", "app1-2", "graph20-80-1rand", "blp-ic98", "piperout-d20", "ns1828997", "neos-4292145-piako",
                end
            end 

            if delta <= kappa
                if data_source == "house_shaped"
                    # Create the Housing-Shaped instance with those parameters
                    println("\n###############################################")
                    println("Creating instance for (k,d)="*string((kappa,delta))*"...")
                    println("###############################################")
                    instance_dir = build_house_problem(kappa,delta)
                    output_dir = "./output/PDHG/IR"
                else
                    if data_source == "MIPLIB"
                        instance_dir = "/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/"
                        # files = readdir(instance_dir)
                        # instance_dir = instance_dir * rand(MIPLIB_instances) * ".mps.gz" # "fiball.mps.gz" # 
                        instance_dir = instance_dir * MIPLIB_instances[kappa] * ".mps.gz"
                        output_dir = "./output/PDHG/MIPLIB_test/" * data_size
                    end
                end

                # existence of out dir
                if !isdir(output_dir)
                    # print(output_dir, "algo")
                    mkdir(output_dir)
                end

                # A folder for each iteration
                instance_name = split(instance_dir, "/")[end]
                instance_name = replace(instance_name, ".mps"=>"")
                instance_name = replace(instance_name, ".gz"=>"")

                output_dir = output_dir * "/" * instance_name 
                if !isdir(output_dir)
                    mkdir(output_dir)
                end

                # # Solve the instance directly
                # println("\n###############################################")
                # println("Solving (PDHG) instance for (k,d)="*string((kappa,delta))*"...")
                # println("###############################################")
                # main(instance_dir, output_dir, objective_tol, time_sec_limit)
                # # println("Solved the .log files of direct method PDLP")
                # ir_out = iterative_refinement(
                #     instance_dir,
                #     output_dir,
                #     objective_tol, # Solving in 0 iter
                #     objective_tol,
                #     time_sec_limit,
                #     0,          # max_iter
                #     1,          # alpha (NOT USED HERE)
                #     false,      # save_log
                # )
                # println("Solved the k=0 iter version of PDLP")

                # # Save the output in .json format
                # output_dir_ir = output_dir * "/" * instance_name * "_PDLP_summary.json"
                # open(output_dir_ir, "w") do io
                #     write(io, JSON3.write(ir_out, allow_inf = true))
                # end

                # Iterative refinement
                # 1.1, 1.5, 1.9, 2, 
                for alpha in [1.1, 1.5, 1.9] # 1024 is P's condition number

                    # println("\n###############################################")
                    # println("Solving (IR) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                    # println("###############################################")
                    
                    # ir_out = iterative_refinement(
                    #     instance_dir,
                    #     output_dir,
                    #     iter_tol,
                    #     objective_tol,
                    #     time_sec_limit,
                    #     max_iter,       # max_iter
                    #     alpha,         # alpha
                    #     false,      # save_log
                    # )

                    # # Save the output in .json format
                    # output_dir_ir = output_dir * "/" * instance_name * "_IR_a"*string(alpha)*"_summary.json"
                    # open(output_dir_ir, "w") do io
                    #     write(io, JSON3.write(ir_out, allow_inf = true))
                    # end

                #     println("\n###############################################")
                #     println("Solving (M_IR D3=D2inv) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                #     println("###############################################")

                #     # D2 scaling version of matrix IR
                #     ir_out_MD2 = M_iterative_refinement(
                #         instance_dir,
                #         output_dir,
                #         iter_tol,
                #         objective_tol,
                #         time_sec_limit,
                #         max_iter,       # max_iter
                #         alpha,          # alpha
                #         false,          # save_log
                #         "D3_eq_D2inv"   # scaling_type
                #     )

                #     # Save the output in .json format
                #     output_dir_ir_MD2 = output_dir * "/" * instance_name * "_MD2_IR_a"*string(alpha)*"_summary.json"
                #     open(output_dir_ir_MD2, "w") do io
                #         write(io, JSON3.write(ir_out_MD2, allow_inf = true))
                #     end


                #     println("\n###############################################")
                #     println("Solving (M_IR D3=D2=I) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                #     println("###############################################")


                #     # D2=I scaling version of matrix IR
                #     ir_out_MI = M_iterative_refinement(
                #         instance_dir,
                #         output_dir,
                #         iter_tol,
                #         objective_tol,
                #         time_sec_limit,
                #         max_iter,       # max_iter
                #         alpha,          # alpha
                #         false,          # save_log
                #         "D3_eq_D2_eq_I" # scaling_type
                #     )


                #     # Save the output in .json format
                #     output_dir_ir_MI = output_dir * "/" * instance_name * "_MI_IR_a"*string(alpha)*"_summary.json"
                #     open(output_dir_ir_MI, "w") do io
                #         write(io, JSON3.write(ir_out_MI, allow_inf = true))
                #     end


                # #     # THIS ONE DOES NOT MAKE MUCH SENSE TO ME:
                # #     println("\n###############################################")
                # #     println("Solving (M_IR D3=D2 & D2=D2inv) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                # #     println("###############################################")


                # #    # D2=D2_inv scaling version of matrix IR
                # #     ir_out_MD3 = M_iterative_refinement(
                # #         instance_dir,
                # #         output_dir,
                # #         iter_tol,
                # #         objective_tol,
                # #         time_sec_limit,
                # #         max_iter,         # max_iter
                # #         alpha,            # alpha
                # #         false,            # save_log
                # #         "D3_eq_D2_and_swap"   # scaling_type
                # #     )

                # #     "D3_eq_D2inv", "D3_eq_D2_eq_I", "D3_eq_D2", "D3_dual_violation"

                # #     # Save the output in .json format
                # #     output_dir_ir_MD3 = output_dir * "/" * instance_name * "_MD3_IR_a"*string(alpha)*"_summary.json"
                # #     open(output_dir_ir_MD3, "w") do io
                # #         write(io, JSON3.write(ir_out_MD3, allow_inf = true))
                # #     end

                #     # # WARNING: Not clear how to recover the dual y (results are not promising in terms of KKT and p-d objectives)
                #     # println("\n###############################################")
                #     # println("Solving (M_IR D3(c-A'y) ) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                #     # println("###############################################")


                #     # # D2=I scaling version of matrix IR
                #     # ir_out_DV = M_iterative_refinement(
                #     #     instance_dir,
                #     #     output_dir,
                #     #     iter_tol,
                #     #     objective_tol,
                #     #     time_sec_limit,
                #     #     max_iter,       # max_iter
                #     #     alpha,          # alpha
                #     #     false,          # save_log
                #     #     "D3_dual_violation" # scaling_type
                #     # )

                #     # # Save the output in .json format
                #     # output_dir_ir_DV = output_dir * "/" * instance_name * "_DV_IR_a"*string(alpha)*"_summary.json"
                #     # open(output_dir_ir_DV, "w") do io
                #     #     write(io, JSON3.write(ir_out_DV, allow_inf = true))
                #     # end



                #     println("\n###############################################")
                #     println("Solving (M_IR D3(c-A'y) & D2=D3inv) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                #     println("###############################################")


                #     # D2=I scaling version of matrix IR
                #     ir_out_DVSw = M_iterative_refinement(
                #         instance_dir,
                #         output_dir,
                #         iter_tol,
                #         objective_tol,
                #         time_sec_limit,
                #         max_iter,       # max_iter
                #         alpha,          # alpha
                #         false,          # save_log
                #         "D3_dual_violation_swap" # scaling_type
                #     )

                #     # Save the output in .json format
                #     output_dir_ir_DVSw = output_dir * "/" * instance_name * "_DVSw_IR_a"*string(alpha)*"_summary.json"
                #     open(output_dir_ir_DVSw, "w") do io
                #         write(io, JSON3.write(ir_out_DVSw, allow_inf = true))
                #     end

                #     println("\n###############################################")
                #     println("Solving (M_IR swaping: D3<->D2) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                #     println("###############################################")


                #     # D2=I scaling version of matrix IR
                #     ir_out_D3D2Sw = M_iterative_refinement(
                #         instance_dir,
                #         output_dir,
                #         iter_tol,
                #         objective_tol,
                #         time_sec_limit,
                #         max_iter,       # max_iter
                #         alpha,          # alpha
                #         false,          # save_log
                #         "D3_D2_iterative_swap" # scaling_type
                #     )

                #     # Save the output in .json format
                #     output_dir_ir_D3D2Sw = output_dir * "/" * instance_name * "_D3D2Sw_IR_a"*string(alpha)*"_summary.json"
                #     open(output_dir_ir_D3D2Sw, "w") do io
                #         write(io, JSON3.write(ir_out_D3D2Sw, allow_inf = true))
                #     end


                    println("\n###############################################")
                    println("Solving (M_IR MIX: D3/D2) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                    println("###############################################")


                    # D2=I scaling version of matrix IR
                    ir_out_MIX = M_iterative_refinement(
                        instance_dir,
                        output_dir,
                        iter_tol,
                        objective_tol,
                        time_sec_limit,
                        max_iter,       # max_iter
                        alpha,          # alpha
                        false,          # save_log
                        "D3_D2_mix" # scaling_type
                    )

                    # Save the output in .json format
                    output_dir_ir_MIX = output_dir * "/" * instance_name * "_MIX_IR_a"*string(alpha)*"_summary.json"
                    open(output_dir_ir_MIX, "w") do io
                        write(io, JSON3.write(ir_out_MIX, allow_inf = true))
                    end


                    println("\n###############################################")
                    println("Solving (M_IR MIX P.: D3/D2) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                    println("###############################################")


                    # D2=I scaling version of matrix IR
                    ir_out_MIXP = M_iterative_refinement(
                        instance_dir,
                        output_dir,
                        iter_tol,
                        objective_tol,
                        time_sec_limit,
                        max_iter,       # max_iter
                        alpha,          # alpha
                        false,          # save_log
                        "D3_D2_mix_pure" # scaling_type
                    )

                    # Save the output in .json format
                    output_dir_ir_MIXP = output_dir * "/" * instance_name * "_MIXP_IR_a"*string(alpha)*"_summary.json"
                    open(output_dir_ir_MIXP, "w") do io
                        write(io, JSON3.write(ir_out_MIXP, allow_inf = true))
                    end



                    println("\n###############################################")
                    println("Solving (M_IR PURE.: D1-D2-D3) instance for (k,d,a)="*string((kappa,delta, alpha))*"...")
                    println("###############################################")


                    # D2=I scaling version of matrix IR
                    ir_out_PURE = M_iterative_refinement(
                        instance_dir,
                        output_dir,
                        iter_tol,
                        objective_tol,
                        time_sec_limit,
                        max_iter,       # max_iter
                        alpha,          # alpha
                        false,          # save_log
                        "D123_pure" # scaling_type
                    )

                    # Save the output in .json format
                    output_dir_ir_PURE = output_dir * "/" * instance_name * "_PURE_IR_a"*string(alpha)*"_summary.json"
                    open(output_dir_ir_PURE, "w") do io
                        write(io, JSON3.write(ir_out_PURE, allow_inf = true))
                    end
                end
            end
        end
    end
end
main()



