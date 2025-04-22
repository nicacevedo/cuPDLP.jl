include("./solve.jl")
# include("./iterative_refinement.jl")
# include("./iterative_refinement2.jl")
include("/nfs/home2/nacevedo/RA/cuPDLP.jl/src/cuPDLP.jl")



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
    # Inside cuPDLP
    ir_instead_restart = true
    ir_type="matrix" #"matrix", "scalar" 
    ir_iteration_threshold=1

    # Inside and outside cuPDLP
    objective_tol = 1e-6 # 1e-12
    
    tol_decrease_factor=0.001 # 1e-3
    iter_tol = 1e3 # Now the initial one
    scaling_bound = 1/objective_tol # if tol = 0, then 1/tol = infinity (?)
    time_sec_limit= 600#1200#3600
    data_source = "MIPLIB" #"house_shaped"  #"MIPLIB" # 
    data_size = "instances_"*string(objective_tol)# Only relevant for miplib
    max_iter = 1000
    # cuPDLP_max_iter_IR = 1000
    save_IR_full_log = false

    base_output_dir = "./output/PDHG_test36_matrix/"
    # Parameter combinations
    kappas = [0.5, 0.99, 1] # [0.1, 0.5, 0.99,1] # , 0.5, 0.99, 1
    deltas = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11] # [1e-3, 1]# , 1e-3, 1
    alphas = [1.01, 1.1, 1.9]#[1.01, 1.1, 1.9]#[ 1.1, 1.9] # 1 + 1e-4,

    # solving method
    solving_methods = [
        # "scalar_no_alpha",  
        "direct_cuPDLP",
        # "direct_cuPDLP_IR",
        # "scalar", 
        # "D3_eq_D2_eq_I",         
        # "D2_D3_adaptive_v5",
        # "D2_D3_adaptive_v3", #v3 is v6
        # "D2_D3_adaptive_v7",
        # "D2_D3_adaptive_v8",
        
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
    "D3_dual_violation_swap"=>"DVSw_IR", # D3 is f(c-A"y), and D2 = 1 ./D3
    "D3_D2_iterative_swap"=>"D3D2Sw_IR", # even k=> D2 is f(l-x), and D3 = 1 ./D2 // odd k=> # D3 is f(c-A"y), and D2 = 1 ./D3
    "D3_D2_iterative_swap_indep"=>"D3D2SwP_IR", # even k=> D2 is f(l-x), and D3 = 1 ./D2 // odd k=> # D3 is f(c-A"y), and D2 = 1 ./D3
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
            # Hand-made
            # 1. Fast instances where cuPDLP wins#["fhnw-schedule-pairb400", "ex10"]#["irish-electricity"]#
            # instances =  ["adult-regularized"]#["bppc6-02", "glass4", "ns1456591" ] # ["2club200v15p5scn", "a2864-99blp", "ex10"]#, "ex1010-pi", "fhnw-schedule-pairb400", "fiball", "neos-1593097", "neos-2629914-sudost"]

            # missed instances
            # New order of instances according to execution time <=600 + iterations >=1000
            # too large & infeasible: "neos-3229051-yass", "neos-3230511-yuna"
            # too large & OOM: 
            # Tiny (<160s)
            # instances = ["neos-2629914-sudost", "sorrell4", "cdc7-4-3-2", "sorrell3", "sorrell7", "ns1904248", "neos-885524", "neos-3237086-abava", "neos-913984", "ns1856153", "genus-sym-g62-2", "neos-2991472-kalu", "dws012-02", "chromaticindex512-7", "chromaticindex1024-7", "fhnw-schedule-paira200", "fhnw-schedule-paira400", "neos-3759587-noosa", "ns1828997", "neos-5221106-oparau", "dws008-03", "dws012-03", "neos-5083528-gimone", "neos-5149806-wieprz", "fhnw-schedule-pairb200", "supportcase43", "graph40-20-1rand", "graph20-80-1rand", "neos-1171448", "ex9", "fhnw-schedule-pairb400", "ex10", "circ10-3", "neos-5195221-niemur", "neos-885086", "neos-5196530-nuhaka", "neos-956971", "neos-3755335-nizao", "neos-5193246-nerang", "graph40-40-1rand", "neos-933966", "dws012-01", "graphdraw-opmanager", "neos-525149", "neos-4409277-trave", "neos-932721", "2club200v15p5scn", "tanglegram4", "supportcase2", "neos-5188808-nattai", "neos-5049753-cuanza", "neos-4360552-sangro", "neos-2978205-isar", "neos-933638", "savsched1", "allcolor58", "neos-950242", "graph40-80-1rand", "neos9", "ns1111636", "ns1116954", "neos-957143", "neos-4300652-rahue", "neos-5118851-kowhai", "neos-4322846-ryton", "neos-5114902-kasavu", "seqsolve2short4288", "highschool1-aigio", "neos-5116085-kenana", "neos-578379", "n2seq36q", "neos-787933", "nursesched-sprint02", "neos-4355351-swalm", "neos-5106984-jizera", "kosova1", "neos-948346", "neos-4359986-taipa", "z26", "neos-1593097", "cryptanalysiskb128n5obj16", "fiball", "vpphard", "neos-738098", "neos-3740487-motru", "neos-4295773-pissa", "academictimetablesmall", "supportcase40", "n3div36", "neos-5079731-flyers", "ns1952667", "v150d30-2hopcds", "academictimetablebig", "neos-953928", "snip10x10-35r1budget17", "datt256", "neos-5076235-embley", "neos-3402294-bobin", "ns930473", "neos-3209462-rhin", "neos-5102383-irwell", "woodlands09", "neos-780889", "k1mushroom", "supportcase23", "neos-848589", "ns1830653", "nsrand-ipx", "n3seq24", "supportcase41", "neos-1367061", "in", "neos-5251015-ogosta", "roi5alpha10n8", "neos6", "ns2124243", "neos-5223573-tarwin", "neos-960392", "seqsolve1", "neos-5273874-yomtsa", "neos-5266653-tugela", "iis-hc-cov", "reblock420", "scpk4", "nursesched-sprint-hidden09", "ns1905797", "nursesched-sprint-late03", "map16715-04", "rmatr200-p10", "shipsched", "neos-3354841-apure", "neos-957323", "thor50dday", "neos-4321076-ruwer", "mushroom-best", "tbfp-network", "neos-1354092", "nucorsav", "gfd-schedulen180f7d50m30k18", "rocII-8-11", "splice1k1", "usafa", "eilC76-2", "neos-2746589-doon", "neos-941313", "supportcase39", "neos-3025225-shelon", "brazil3", "wnq-n100-mw99-14", "neos-3322547-alsek", "ex1010-pi", "neos-662469", "eva1aprime6x6opt", "bppc6-02", "dc1l", "ns2122698", "rocII-5-11", "neos-498623", "scpn2", "uccase8", "bppc6-06", "blp-ar98", "scpm1", "blp-ic98", "cmflsp50-24-8-8", "adult-regularized", "neos-1122047", "neos8", "supportcase12", "neos-4954274-beardy", "neos-4972437-bojana", "rmatr200-p5", "neos-4292145-piako", "seqsolve3short4288excess384", "blp-ic97", "t1722", "satellites2-40", "ns1456591", "neos-4647030-tutaki", "neos-826224", "ns1644855", "neos-860300", "hypothyroid-k1", "neos-4972461-bolong"]
            # Small (<300s)
            # instances = ["uccase7", "satellites2-25", "scpl4", "neos-4408804-prosna", "rail507", "neos-4647032-veleka", "neos-5129192-manaia", "sct5", "neos-4647027-thurso", "supportcase10", "cmflsp40-36-2-10", "neos-873061", "tpl-tub-ss16", "uccase9", "sp97ic", "neos-872648", "bab5", "ivu52", "ns1430538", "van", "satellites2-60-fs", "neos-3372571-onahau", "neos-5118834-korana", "neos-5108386-kalang", "rmine13", "opm2-z8-s0"]
            # # Medium (<480)
            # instances = ["neos-5052403-cygnet", "opm2-z10-s4", "pb-grow22", "piperout-d27", "ds", "neos-4976951-bunnoo", "cmflsp40-24-10-7", "t1717", "ger50-17-trans-dfn-3t", "adult-max5features", "fast0507", "sp97ar", "atlanta-ip", "neos-827175", "uccase12", "fhnw-binschedule2"]
            # # Large (>= 480)
            instances = ["neos-4562542-watut", "square31", "neos-4724674-aorere", "s55", "neos-5104907-jarama", "neos-4555749-wards", "neos-5138690-middle", "stockholm", "s100", "ivu06", "s250r10", "sp98ic"]


            # Original
            # 1. Tiny
            # instances = ["ns1830653", "neos-738098", "neos-578379", "ex1010-pi", "momentum1", "v150d30-2hopcds", "neos-950242", "supportcase40", "fiball", "2club200v15p5scn", "rmatr200-p10", "sct1", "ns1856153", "lr1dr04vc05v17a-t360", "neos-932721", "sct32", "30n20b8", "neos-827175", "rmatr200-p5", "neos-1593097", "neos-3372571-onahau", "ns1631475", "neos-4805882-barwon", "blp-ic97", "fhnw-schedule-paira200", "shipsched", "satellites2-60-fs", "neos-4359986-taipa", "unitcal_7", "cmflsp40-24-10-7", "neos-826224", "ci-s4", "leo1", "neos-1171448", "dws012-01", "t1722", "brazil3", "neos-5188808-nattai", "neos-5013590-toitoi", "mzzv11", "fhnw-binschedule2", "chromaticindex512-7", "supportcase37", "reblock420", "neos-824661", "neos-4954274-beardy", "iis-hc-cov", "ns930473", "sct5", "neos-498623", "mzzv42z", "neos-913984", "opm2-z8-s0", "gmut-76-40", "pb-grow22", "bab5", "physiciansched5-3", "sorrell7", "cmflsp50-24-8-8", "supportcase23", "eva1aprime6x6opt", "neos-1122047", "dws008-03", "neos-953928", "stockholm", "neos-5266653-tugela", "neos-5196530-nuhaka", "ger50-17-trans-dfn-3t", "ger50-17-trans-pop-3t", "germanrr", "neos-5193246-nerang", "neos-5195221-niemur", "ns1430538", "neos-933966", "bppc6-06", "n2seq36q", "neos-4300652-rahue", "neos-933638", "neos-1354092", "bppc6-02", "mushroom-best", "d20200", "neos-960392", "neos-4292145-piako", "ns1828997", "piperout-d20", "blp-ic98", "graph20-80-1rand", "app1-2", "ns1456591"]
            # 2. Small
            # instances = ["neos-662469", "blp-ar98", "nursesched-sprint02", "neos-4409277-trave", "neos-2978205-isar", "nursesched-sprint-hidden09", "nursesched-sprint-late03", "neos-2629914-sudost", "supportcase33", "thor50dday", "uccase8", "tbfp-network", "leo2", "supportcase41", "nsrand-ipx", "piperout-d27", "graphdraw-opmanager", "neos-3237086-abava", "ns1905797", "rmine11", "neos9", "neos-5221106-oparau", "neos-885086", "neos6", "atlanta-ip", "neos-885524", "cdc7-4-3-2", "neos-1367061", "dws012-02", "ns1690781", "neos-4408804-prosna", "neos-4760493-puerua", "neos-4763324-toguru", "sing326", "academictimetablesmall", "chromaticindex1024-7", "neos-2746589-doon", "genus-sym-g62-2", "sing44", "satellites2-25", "satellites2-40", "sp97ar", "cryptanalysiskb128n5obj16", "neos-787933", "fhnw-schedule-pairb200", "neos-4360552-sangro", "rocII-5-11", "circ10-3", "neos-4722843-widden", "neos8", "eilC76-2", "sp98ic", "sp97ic", "s55", "neos-3759587-noosa", "t1717", "tanglegram4", "uccase9", "ns1952667", "uccase7", "sorrell3", "n3div36", "graph40-20-1rand", "neos-872648", "neos-873061", "opm2-z10-s4", "neos-4355351-swalm", "vpphard", "neos-4724674-aorere", "ns1904248", "neos-860300", "neos-5083528-gimone", "cmflsp40-36-2-10", "dws012-03", "rail01", "supportcase39", "radiationm40-10-02", "sing5", "fast0507", "ns1116954", "adult-max5features", "adult-regularized", "uccase12", "ns2122698", "sp98ar", "ns2124243", "hypothyroid-k1", "lr2-22dr3-333vc4v17a-t60", "supportcase42", "neos-3209462-rhin", "dc1l", "neos-4322846-ryton", "neos-4797081-pakoka", "snp-02-004-104", "rail507", "gmut-76-50", "snip10x10-35r1budget17", "sing11", "fhnw-schedule-paira400", "physiciansched6-2", "neos-956971", "rocII-8-11", "neos-941313", "rmine13", "van", "neos-780889", "neos-957143", "neos-957323", "neos-5149806-wieprz", "proteindesign122trx11p8", "allcolor58", "triptim1", "ex9", "triptim4", "triptim2", "triptim7", "triptim8", "irish-electricity", "proteindesign121pgb11p9", "neos-948346", "physiciansched6-1", "neos-3755335-nizao", "map06", "map10", "map14860-20", "map16715-04", "map18", "neos-4295773-pissa", "supportcase10", "ns1111636", "gmut-75-50", "neos-2987310-joes", "cmflsp60-36-2-6", "supportcase6", "neos-4391920-timok", "lr1dr12vc10v70b-t360", "neos-3695882-vesdre", "rocII-10-11", "graphdraw-grafo2", "netdiversion", "nursesched-medium04", "nursesched-medium-hint03", "proteindesign121hz512p9", "nw04", "neos-2669500-cust", "neos-4306827-ravan", "vpphard2", "stp3d", "neos-876808", "neos-5100895-inster", "satellites3-25", "neos-4260495-otere", "fastxgemm-n3r21s3t6", "opm2-z12-s8", "fastxgemm-n3r22s4t6", "rail02", "neos-4413714-turia", "ns1849932", "neos-5093327-huahum", "fastxgemm-n3r23s5t6", "uccase10", "buildingenergy", "neos-3555904-turama", "satellites4-25", "neos-631710", "neos-954925", "hgms30", "var-smallemery-m6j6", "rd-rplusc-21", "bab1", "neos-3352863-ancoa", "roi2alpha3n4", "rmine15", "square23", "physiciansched3-4", "eilA101-2", "neos-4966258-blicks", "rvb-sub", "scpj4scip"]
            # 3. Medium 
            # instances = ["scpk4", "sorrell4", "ds-big", "seqsolve1", "seqsolve2short4288", "seqsolve3short4288excess384", "supportcase43", "shs1023", "shs1014", "physiciansched3-3", "neos-5076235-embley", "neos-5079731-flyers", "neos-848589", "shs1042", "ex10", "fhnw-schedule-pairb400", "sing17", "neos-5102383-irwell", "gfd-schedulen180f7d50m30k18", "neos-4533806-waima", "graph40-40-1rand", "neos-4555749-wards", "neos-4321076-ruwer", "bab6", "s250r10", "wnq-n100-mw99-14", "neos-5041822-cockle", "fhnw-binschedule0", "neos-5049753-cuanza", "neos-4972461-bolong", "datt256", "neos-525149", "academictimetablebig", "highschool1-aigio", "pb-fit2d", "neos-5129192-manaia", "zeil", "k1mushroom", "z26", "neos-4972437-bojana", "neos-4976951-bunnoo", "hgms62", "splice1k1", "savsched1", "s100", "ns1760995", "neos-5273874-yomtsa", "neos-5251015-ogosta", "co-100", "scpl4", "neos-3322547-alsek", "neos-4562542-watut", "bab2", "neos-3402294-bobin", "neos-5104907-jarama", "ns1644855", "neos-5223573-tarwin", "ivu52", "supportcase12", "roi5alpha10n8", "woodlands09", "neos-5106984-jizera", "neos-5118834-korana", "supportcase7", "neos-5116085-kenana", "n3seq24", "bab3", "neos-5138690-middle", "kosova1", "rmine21", "neos-5108386-kalang", "neos-5118851-kowhai", "graph40-80-1rand", "tpl-tub-ss16", "neos-5123665-limmat", "square31", "neos-4647027-thurso", "neos-4647030-tutaki", "neos-4647032-veleka", "neos-5114902-kasavu", "supportcase19", "rwth-timetable", "ds", "tpl-tub-ws1617", "neos-5052403-cygnet", "neos-3354841-apure", "scpm1", "in", "rmine25", "supportcase2", "ivu06", "fhnw-binschedule1", "neos-3025225-shelon", "square37"]
            # 4. Large
            # instances = ["neos-4545615-waita", "scpn2", "kottenpark09", "square41", "neos-2991472-kalu", "usafa", "dlr1", "a2864-99blp", "ivu06-big", "neos-3208254-reiu", "nucorsav", "neos-3740487-motru", "square47", "ivu59", "t11nonreg", "neos-4332801-seret", "neos-4332810-sesia", "neos-3229051-yass", "neos-3230511-yuna", "neos-4535459-waipa"]
            # Convergent instances
            # instances = ["ns1456591"] #["glass4", 
            # Nonconvergent instances
            # instances = ["irish-electricity"]

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

        println("\nINSTANCE: "*instance_name*"\n")

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


            println("\n SOLVING METHOD: "*solving_method*"\n")

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
                    main(instance_dir, output_dir_instance, objective_tol, time_sec_limit, ir_instead_restart, ir_type, ir_iteration_threshold) # NEW VERSION
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
