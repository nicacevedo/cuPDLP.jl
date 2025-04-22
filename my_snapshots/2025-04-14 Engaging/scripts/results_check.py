import json 
import gzip
import numpy as np
import matplotlib.pyplot as plt

l2_norm = lambda x: np.sqrt( np.array(x) @ np.array(x) )


plt.figure(figsize=(8,6))
# MINE
# for kappa in [0.01, 0.1, 0.99]:
#     for delta in [kappa/100, kappa/10, kappa]:

# # MINE 2: Rational kappa, and easy delta (to prove is not that, and it is near-deg)
# for kappa in [0.987654321987654321, 0.987654321, 0.98]:
#     for delta in [0.987, 0.985654321, 0.90987654321987654321]:

# # # Paper (more or less)
# for kappa in [1.0]:
for kappa in [0.1, 0.99, 1.0]:
    for delta in [0.0, 1e-3, 1.0]:
#     # for delta in [1.0]:
        if delta <= kappa:

            # Example output to test
            output_directory = f"output/PDHG/IR/house_k{kappa}_d{delta}"
            output_file = f"house_k{kappa}_d{delta}_full_log.json.gz"
            # output_file_2 = f"house_k{kappa}_d{delta}_iter_stats.json.gz"



            # Load the output file
            with gzip.open(f"{output_directory}/{output_file}", "rt") as f:
                output = json.load(f)
            # with gzip.open(f"{output_directory}/{output_file_2}", "rt") as f:
            #     output_2 = json.load(f)
            

            instance = output["instance_name"]
            n_iters = output["iteration_count"]
            print(output)

            kkt_keys = [
                "relative_l_inf_primal_residual", # primal feas.
                "relative_l_inf_dual_residual", # dual feas. 
                "relative_optimality_gap" # optimality gap between primal and dual (comp. slackness implicitly)
                ]


            kkt_values = []
            force_continue = False
            for k in range(int(n_iters)+1):
                # print(f"iteration {k}")
                iter_kkt_values = []
                for key in kkt_keys:
                    # print(f"{key}: {output['iteration_stats'][k]['convergence_information'][0][key]}")
                    try:
                        iter_kkt_values.append(output["iteration_stats"][k]['convergence_information'][0][key])
                    except Exception as e:
                        print("error in:",kappa,delta,k)
                        print(e)
                        print(len(output["iteration_stats"]))
                        # print(len(output_2))
                        break
                        


                kkt_values.append(l2_norm(iter_kkt_values))
                # print("iter kkt vals.:", iter_kkt_values)
                # print("kkt:", )
                
                # output["iteration_stats"][k]
                # print(f"{output['iteration_stats'][k]}")


            # Plot the kkt evolution through the iterations
            k_iters = [k+1 for k in range(n_iters+1)]
            plt.plot(k_iters, kkt_values, label=fr"($\kappa$,$\delta$)={(kappa,delta)}")
plt.legend( bbox_to_anchor=(1, 0.8))
plt.yscale('log')
plt.xscale('log')
plt.title("cuPDLP on House-Shaped problem")
plt.ylabel("KKT residual")
plt.xlabel("Iterations")
plt.show()

            
