include("./solve.jl")
# include("./iterative_refinement.jl")
include("./iterative_refinement2.jl")



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

    objective_tol = 1e-6 # 1e-12
    iter_tol = 1e3 # Now the initial one
    tol_decrease_factor=0.001 # 1e-3
    scaling_bound = 1/objective_tol # if tol = 0, then 1/tol = infinity (?)
    time_sec_limit= 3600#1200#3600
    data_source = "MIPLIB" #"house_shaped"  #"MIPLIB" # 
    data_size = "instances_"*string(objective_tol)# Only relevant for miplib
    max_iter = 1000
    # cuPDLP_max_iter_IR = 1000
    save_IR_full_log = false
    base_output_dir = "./output/PDHG_test34/"
    # Parameter combinations
    kappas = [0.5, 0.99, 1] # [0.1, 0.5, 0.99,1] # , 0.5, 0.99, 1
    deltas = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11] # [1e-3, 1]# , 1e-3, 1
    alphas = [1.01, 1.1, 1.9]#[1.01, 1.1, 1.9]#[ 1.1, 1.9] # 1 + 1e-4,

    # solving method
    solving_methods = [
        # # # "direct_cuPDLP",
        "scalar_no_alpha",  
        "direct_cuPDLP_IR",
        "scalar", 
        # "D3_eq_D2_eq_I",         
        "D2_D3_adaptive_v5",
        "D2_D3_adaptive_v6",
        "D2_D3_adaptive_v7",
        "D2_D3_adaptive_v8",
        
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
    "D123_pure_max"=>"PURE_MAX_IR",
    "D2_D3_adaptive_v5"=>"ADAPT_v5_IR",
    "D2_D3_adaptive_v6"=>"ADAPT_v6_IR",
    "D2_D3_adaptive_v7"=>"ADAPT_v7_IR", 
    "D2_D3_adaptive_v8"=>"ADAPT_v8_IR",

    )


    if data_source == "MIPLIB"
        if data_size == "tiny_instances" || contains(data_size,"instances_")
            # 1. Tiny
            # instances = ["ns1830653", "neos-738098", "neos-578379", "ex1010-pi", "momentum1", "v150d30-2hopcds", "neos-950242", "supportcase40", "fiball", "2club200v15p5scn", "rmatr200-p10", "sct1", "ns1856153", "lr1dr04vc05v17a-t360", "neos-932721", "sct32", "30n20b8", "neos-827175", "rmatr200-p5", "neos-1593097", "neos-3372571-onahau", "ns1631475", "neos-4805882-barwon", "blp-ic97", "fhnw-schedule-paira200", "shipsched", "satellites2-60-fs", "neos-4359986-taipa", "unitcal_7", "cmflsp40-24-10-7", "neos-826224", "ci-s4", "leo1", "neos-1171448", "dws012-01", "t1722", "brazil3", "neos-5188808-nattai", "neos-5013590-toitoi", "mzzv11", "fhnw-binschedule2", "chromaticindex512-7", "supportcase37", "reblock420", "neos-824661", "neos-4954274-beardy", "iis-hc-cov", "ns930473", "sct5", "neos-498623", "mzzv42z", "neos-913984", "opm2-z8-s0", "gmut-76-40", "pb-grow22", "bab5", "physiciansched5-3", "sorrell7", "cmflsp50-24-8-8", "supportcase23", "eva1aprime6x6opt", "neos-1122047", "dws008-03", "neos-953928", "stockholm", "neos-5266653-tugela", "neos-5196530-nuhaka", "ger50-17-trans-dfn-3t", "ger50-17-trans-pop-3t", "germanrr", "neos-5193246-nerang", "neos-5195221-niemur", "ns1430538", "neos-933966", "bppc6-06", "n2seq36q", "neos-4300652-rahue", "neos-933638", "neos-1354092", "bppc6-02", "mushroom-best", "d20200", "neos-960392", "neos-4292145-piako", "ns1828997", "piperout-d20", "blp-ic98", "graph20-80-1rand", "app1-2", "ns1456591"]
            # 2. Small
            # instances = ["neos-662469", "blp-ar98", "nursesched-sprint02", "neos-4409277-trave", "neos-2978205-isar", "nursesched-sprint-hidden09", "nursesched-sprint-late03", "neos-2629914-sudost", "supportcase33", "thor50dday", "uccase8", "tbfp-network", "leo2", "supportcase41", "nsrand-ipx", "piperout-d27", "graphdraw-opmanager", "neos-3237086-abava", "ns1905797", "rmine11", "neos9", "neos-5221106-oparau", "neos-885086", "neos6", "atlanta-ip", "neos-885524", "cdc7-4-3-2", "neos-1367061", "dws012-02", "ns1690781", "neos-4408804-prosna", "neos-4760493-puerua", "neos-4763324-toguru", "sing326", "academictimetablesmall", "chromaticindex1024-7", "neos-2746589-doon", "genus-sym-g62-2", "sing44", "satellites2-25", "satellites2-40", "sp97ar", "cryptanalysiskb128n5obj16", "neos-787933", "fhnw-schedule-pairb200", "neos-4360552-sangro", "rocII-5-11", "circ10-3", "neos-4722843-widden", "neos8", "eilC76-2", "sp98ic", "sp97ic", "s55", "neos-3759587-noosa", "t1717", "tanglegram4", "uccase9", "ns1952667", "uccase7", "sorrell3", "n3div36", "graph40-20-1rand", "neos-872648", "neos-873061", "opm2-z10-s4", "neos-4355351-swalm", "vpphard", "neos-4724674-aorere", "ns1904248", "neos-860300", "neos-5083528-gimone", "cmflsp40-36-2-10", "dws012-03", "rail01", "supportcase39", "radiationm40-10-02", "sing5", "fast0507", "ns1116954", "adult-max5features", "adult-regularized", "uccase12", "ns2122698", "sp98ar", "ns2124243", "hypothyroid-k1", "lr2-22dr3-333vc4v17a-t60", "supportcase42", "neos-3209462-rhin", "dc1l", "neos-4322846-ryton", "neos-4797081-pakoka", "snp-02-004-104", "rail507", "gmut-76-50", "snip10x10-35r1budget17", "sing11", "fhnw-schedule-paira400", "physiciansched6-2", "neos-956971", "rocII-8-11", "neos-941313", "rmine13", "van", "neos-780889", "neos-957143", "neos-957323", "neos-5149806-wieprz", "proteindesign122trx11p8", "allcolor58", "triptim1", "ex9", "triptim4", "triptim2", "triptim7", "triptim8", "irish-electricity", "proteindesign121pgb11p9", "neos-948346", "physiciansched6-1", "neos-3755335-nizao", "map06", "map10", "map14860-20", "map16715-04", "map18", "neos-4295773-pissa", "supportcase10", "ns1111636", "gmut-75-50", "neos-2987310-joes", "cmflsp60-36-2-6", "supportcase6", "neos-4391920-timok", "lr1dr12vc10v70b-t360", "neos-3695882-vesdre", "rocII-10-11", "graphdraw-grafo2", "netdiversion", "nursesched-medium04", "nursesched-medium-hint03", "proteindesign121hz512p9", "nw04", "neos-2669500-cust", "neos-4306827-ravan", "vpphard2", "stp3d", "neos-876808", "neos-5100895-inster", "satellites3-25", "neos-4260495-otere", "fastxgemm-n3r21s3t6", "opm2-z12-s8", "fastxgemm-n3r22s4t6", "rail02", "neos-4413714-turia", "ns1849932", "neos-5093327-huahum", "fastxgemm-n3r23s5t6", "uccase10", "buildingenergy", "neos-3555904-turama", "satellites4-25", "neos-631710", "neos-954925", "hgms30", "var-smallemery-m6j6", "rd-rplusc-21", "bab1", "neos-3352863-ancoa", "roi2alpha3n4", "rmine15", "square23", "physiciansched3-4", "eilA101-2", "neos-4966258-blicks", "rvb-sub", "scpj4scip"]
            # 3. Medium
            # instances = ["scpk4", "sorrell4", "ds-big", "seqsolve1", "seqsolve2short4288", "seqsolve3short4288excess384", "supportcase43", "shs1023", "shs1014", "physiciansched3-3", "neos-5076235-embley", "neos-5079731-flyers", "neos-848589", "shs1042", "ex10", "fhnw-schedule-pairb400", "sing17", "neos-5102383-irwell", "gfd-schedulen180f7d50m30k18", "neos-4533806-waima", "graph40-40-1rand", "neos-4555749-wards", "neos-4321076-ruwer", "bab6", "s250r10", "wnq-n100-mw99-14", "neos-5041822-cockle", "fhnw-binschedule0", "neos-5049753-cuanza", "neos-4972461-bolong", "datt256", "neos-525149", "academictimetablebig", "highschool1-aigio", "pb-fit2d", "neos-5129192-manaia", "zeil", "k1mushroom", "z26", "neos-4972437-bojana", "neos-4976951-bunnoo", "hgms62", "splice1k1", "savsched1", "s100", "ns1760995", "neos-5273874-yomtsa", "neos-5251015-ogosta", "co-100", "scpl4", "neos-3322547-alsek", "neos-4562542-watut", "bab2", "neos-3402294-bobin", "neos-5104907-jarama", "ns1644855", "neos-5223573-tarwin", "ivu52", "supportcase12", "roi5alpha10n8", "woodlands09", "neos-5106984-jizera", "neos-5118834-korana", "supportcase7", "neos-5116085-kenana", "n3seq24", "bab3", "neos-5138690-middle", "kosova1", "rmine21", "neos-5108386-kalang", "neos-5118851-kowhai", "graph40-80-1rand", "tpl-tub-ss16", "neos-5123665-limmat", "square31", "neos-4647027-thurso", "neos-4647030-tutaki", "neos-4647032-veleka", "neos-5114902-kasavu", "supportcase19", "rwth-timetable", "ds", "tpl-tub-ws1617", "neos-5052403-cygnet", "neos-3354841-apure", "scpm1", "in", "rmine25", "supportcase2", "ivu06", "fhnw-binschedule1", "neos-3025225-shelon", "square37"]
            # 4. Large
            instances = ["neos-4545615-waita", "scpn2", "kottenpark09", "square41", "neos-2991472-kalu", "usafa", "dlr1", "a2864-99blp", "ivu06-big", "neos-3208254-reiu", "nucorsav", "neos-3740487-motru", "square47", "ivu59", "t11nonreg", "neos-4332801-seret", "neos-4332810-sesia", "neos-3229051-yass", "neos-3230511-yuna", "neos-4535459-waipa"]
            # Convergent instances
            # instances = ["ns1456591"] #["glass4", 
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

            # SKIP if scalar-no-alpha
            if solving_method == "scalar_no_alpha" 
                if instance_name == instances[1]
                    ir_out = iterative_refinement(
                        instance_dir,
                        output_dir_instance,
                        1,      
                        1,
                        10,
                        1,      # max_iter
                        0,              # alpha 
                        false,   # save_log
                        true,#!(alpha > 0),        #no_alpha_scaling   
                        scaling_bound,       # scaling_bound  
                        tol_decrease_factor     # tolerance decreasing factor
                    )
                else 
                    continue
                end
            end 


            # After skipping 
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
                            real_save_IR_full_log = false # true BENCHMARK: No log
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
