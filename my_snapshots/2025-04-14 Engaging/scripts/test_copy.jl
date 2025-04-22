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

function test()

    objective_tol = 1e-8 # 1e-12
    iter_tol = 1e3 # Now the initial one
    tol_decrease_factor=0.1
    scaling_bound = 1/objective_tol # if tol = 0, then 1/tol = infinity (?)
    time_sec_limit= 600#1200#3600
    data_source = "MIPLIB" #"house_shaped"  #"MIPLIB" # 
    data_size = "instances_"*string(objective_tol)# Only relevant for miplib
    max_iter = 1000
    # cuPDLP_max_iter_IR = 1000
    save_IR_full_log = false
    base_output_dir = "./output/PDHG_test1/"
    # Parameter combinations
    kappas = [0.5, 0.99, 1] # [0.1, 0.5, 0.99,1] # , 0.5, 0.99, 1
    deltas = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11] # [1e-3, 1]# , 1e-3, 1
    alphas = [1.1, 1.9]#[1.01, 1.1, 1.9]#[ 1.1, 1.9] # 1 + 1e-4,

    # solving method
    solving_methods = [
        # # # "direct_cuPDLP",
        "direct_cuPDLP_IR",
        "scalar", 
        # "scalar_no_alpha",  
        "D3_eq_D2_eq_I", 
        # "D3_eq_D2_eq_I_indep",      # NEW NEW ONE
        # "D3_D2_iterative_swap",     
        # "D3_D2_iterative_swap_indep", # NEW NEW ONE 
        
        # # # # "D3_eq_D2inv",      
        # # # # "D3_eq_D2_and_swap",
        # # # # "D3_dual_violation",

        # # # "D3_dual_violation_swap", # NOT USED

        # "D3_D2_mix", 
        # "D3_D2_mix_pure", 
        # "D3_D2_mix_max",        # NEW ONES
        # "D3_D2_mix_max_pure",   # NEW ONES
        # "D123_pure",
        # # "D123_pure_max"
        ]

    alpha_independent_methods = [
        "direct_cuPDLP",
        "direct_cuPDLP_IR",
        "scalar_no_alpha",  
        "D3_eq_D2_eq_I_indep",  # NEW NEW ONE
        "D3_D2_iterative_swap_indep", # NEW NEW ONE
        
        "D123_pure",
        "D123_pure_max"
    ]


    scaling_dict = Dict(
    # "direct_cuPDLP"=>"", # Does not need this
    "direct_cuPDLP_IR"=>"PDLP",
    "scalar"=>"IR",              # Original scalar IR
    "scalar_no_alpha"=>"IR_NO",  # Scalar scaling without alpha, and bounded 
    "D3_eq_D2inv"=> "MD2_IR",    # D2 is f(l-x), and D3 = 1 ./D2
    "D3_eq_D2_eq_I"=> "MI_IR",   # D3 = D2 = I (only D1 scaling)
    "D3_eq_D2_eq_I_indep"=>"MIP_IR", # D3 = D2 = I (only D1 scaling), indep of alpha
    "D3_dual_violation_swap"=>"DVSw_IR", # D3 is f(c-A'y), and D2 = 1 ./D3
    "D3_D2_iterative_swap"=>"D3D2Sw_IR", # even k=> D2 is f(l-x), and D3 = 1 ./D2 // odd k=> # D3 is f(c-A'y), and D2 = 1 ./D3
    "D3_D2_iterative_swap_indep"=>"D3D2SwP_IR", # even k=> D2 is f(l-x), and D3 = 1 ./D2 // odd k=> # D3 is f(c-A'y), and D2 = 1 ./D3
    "D3_D2_mix"=> "MIX_IR",       # D2 = min(1/d2 , d3, alpha*D2), and D3 = 1 ./D2 (Does this make sense??)
    "D3_D2_mix_pure"=> "MIXP_IR", # D2 = min(1/d2 , d3), and D3 = 1 ./D2 (Does this make sense??)
    "D3_D2_mix_max"=> "MIX_MAX_IR",       # NEW ONES 
    "D3_D2_mix_max_pure"=> "MIXP_MAX_IR", # NEW ONES
    "D123_pure"=>"PURE_IR",        # D2 = min(1/d2 , d3), D3 = 1 ./D2, and D1 = 1/d1
    "D123_pure_max"=>"PURE_MAX_IR"
    )


    if data_source == "MIPLIB"
        if data_size == "tiny_instances" || contains(data_size,"instances_")
            # Convergent instances
            instances = ["ns1456591"] #["glass4", 
            # Nonconvergent instances
            # instances = ["irish-electricity"]
            # Mix
            #["glass4","irish-electricity"]#["irish-electricity", "lr1dr12vc10v70b-t360", "lr2-22dr3-333vc4v17a-t60", "map06", "map10", "map18", "neos-4391920-timok", "proteindesign121pgb11p9", "dlr1", "neos-4332801-seret"] # Instances that fail to converge on certain tolerance
            # I was testing on ["ns1828997", "ns1456591", "app1-2", "neos-1354092", "bppc6-02", "blp-ic98"]#["ns1828997", "neos-4292145-piako"]#["neos-4292145-piako"] # Single instance to analize 
                    # MIPLIB_instances = ["ns1456591", "app1-2", "graph20-80-1rand", "blp-ic98", "piperout-d20", "ns1828997", "neos-4292145-piako", "neos-960392", "d20200", "mushroom-best",  # Estan todos
            # MIPLIB_instances = ["bppc6-02", "neos-1354092", "neos-933638", "neos-4300652-rahue", "n2seq36q", "bppc6-06", "neos-933966", "ns1430538"] # Faltan algunos
            # 30 mins:
            #  "neos-5195221-niemur", "neos-5193246-nerang",
            # 10 mins:
            # MIPLIB_instances = ["germanrr", "ger50-17-trans-dfn-3t", "ger50-17-trans-pop-3t", "neos-5196530-nuhaka", "neos-5266653-tugela", "stockholm", "neos-953928", "dws008-03", "neos-1122047", "eva1aprime6x6opt", "supportcase23", "cmflsp50-24-8-8", "sorrell7", "physiciansched5-3", "bab5", "pb-grow22", "gmut-76-40", "opm2-z8-s0", "neos-913984", "mzzv42z", "neos-498623", "sct5", "ns930473", "iis-hc-cov", "neos-4954274-beardy", "neos-824661", "reblock420", "supportcase37", "chromaticindex512-7", "fhnw-binschedule2", "mzzv11", "neos-5013590-toitoi", "neos-5188808-nattai", "brazil3", "t1722", "dws012-01", "neos-1171448", "leo1", "ci-s4", "neos-826224", "cmflsp40-24-10-7", "unitcal_7", "neos-4359986-taipa", "satellites2-60-fs", "shipsched", "fhnw-schedule-paira200", "blp-ic97", "neos-4805882-barwon", "ns1631475", "neos-3372571-onahau", "neos-1593097", "rmatr200-p5", "neos-827175", "30n20b8", "sct32", "neos-932721", "lr1dr04vc05v17a-t360", "ns1856153", "sct1", "rmatr200-p10", "2club200v15p5scn", "fiball", "supportcase40", "neos-950242", "v150d30-2hopcds", "momentum1", "ex1010-pi", "neos-578379", "neos-738098", "ns1830653"] # Falta todo
            # MIPLIB_instances ="ns1430538"

            instance_path = "/nfs/sloanlab007/projects/pdopt_proj/instances/lp/miplib2017/"
            output_dir = base_output_dir*"MIPLIB_test/"*data_size
            # instance_dir = instance_path * MIPLIB_instances[kappa] * ".mps.gz"
        end
    elseif data_source == "house_shaped"
        instances = []
        for kappa in kappas
            for delta in deltas
                if kappa>=delta
                    # Instance array}
                    instance_name = "house_k$(kappa)_d$(delta)"
                    push!(instances, instance_name)

                    # Create the Housing-Shaped instance with those parameters
                    println("\n###############################################")
                    println("Creating instance for (k,d)="*string((kappa,delta))*"...")
                    println("###############################################")
                    instance_dir = build_house_problem(kappa,delta)
                    instance_path = "./instance/"
                    output_dir = base_output_dir*"IR"
                end
            end
        end
    end

    println("INSTANCES: ",instances)

    # existence of out dir
    if !isdir(output_dir)
        # println(output_dir, "algo")
        mkdir(output_dir)
    end


    # Loop over the instances
    for (i,instance_name) in enumerate(instances)
        if data_source == "MIPLIB"
            instance_dir = instance_path * instance_name * ".mps.gz"
        elseif data_source == "house_shaped"
            instance_dir = instance_path * instance_name * ".mps"
        end

        # Existence of the output dir for the instance
        output_dir_instance = output_dir * "/" * instance_name 
        if !isdir(output_dir_instance)
            mkdir(output_dir_instance)
        end

        for solving_method in solving_methods

             
            if solving_method in alpha_independent_methods 
                real_alphas = [0]
            else 
                real_alphas = alphas 
            end

            for alpha in real_alphas
                println("ALPHA: ", alpha)
                # Solve the instance directly
                println("\n###############################################")
                println("Solving instance "*instance_name*" via "*solving_method*"...")
                println("###############################################")
    
                if solving_method == "direct_cuPDLP"
                    # println(instance_dir)
                    # println(output_dir_instance) 
                    # println(objective_tol) 
                    # println(time_sec_limit)
                    main(instance_dir, output_dir_instance, objective_tol, time_sec_limit)
                    println("Solved the .log files of direct method PDLP")
                    # exit()
                else
                    # Scalar version of IR
                    if solving_method in ["direct_cuPDLP_IR", "scalar", "scalar_no_alpha"]
        
                        # no_alpha_scaling = true if solving_method == "scalar_no_alpha" else false
                         
                        if solving_method != "direct_cuPDLP_IR" 
                            real_max_iter = max_iter
                            real_iter_tol = iter_tol
                            real_save_IR_full_log = save_IR_full_log
                        else 
                            real_max_iter = 0 
                            real_iter_tol = objective_tol
                            real_save_IR_full_log = true
                        end 
        
                        # println("Solved the .log files of direct method PDLP")
                        ir_out = iterative_refinement(
                            instance_dir,
                            output_dir_instance,
                            real_iter_tol,      
                            objective_tol,
                            time_sec_limit,
                            real_max_iter,      # max_iter
                            alpha,              # alpha 
                            real_save_IR_full_log,   # save_log
                            !(alpha > 0),        #no_alpha_scaling   
                            scaling_bound,       # scaling_bound  
                            tol_decrease_factor     # tolerance decreasing factor
                        )
                        # println("Solved the k=0 iter version of PDLP")

                    else    
                        # Matrix version of IR
                        ir_out = M_iterative_refinement(
                            instance_dir,
                            output_dir_instance,
                            iter_tol,
                            objective_tol,
                            time_sec_limit,
                            max_iter,           # max_iter
                            alpha,              # alpha
                            save_IR_full_log,   # save_log
                            solving_method,      # scaling_type
                            scaling_bound,       # scaling_bound
                            tol_decrease_factor     # tolerance decreasing factor
                        )
                    end
                    
                    # Save the output in .json format
                    # println("output_dir_instance", output_dir_instance)
                    # println("output_dir", output_dir)
                    # exit()
                    println("Saving results for "*instance_name*" solved via "*solving_method*"...")
                    # output_dir_ir = output_dir_instance * "/" * instance_name * "_PDLP_summary.json"
                    output_dir_ir = output_dir_instance * "/" * instance_name * "_"*scaling_dict[solving_method]*"_a"*string(alpha)*"_summary.json"
                    open(output_dir_ir, "w") do io
                        write(io, JSON3.write(ir_out, allow_inf = true))
                    end

                end

            end 
        end
    end
end
# end

test()
