import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario
from itertools import product

# measures
from scipy.stats import kendalltau, describe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Corras.Evaluation.evaluation import ndcg_at_k, compute_relevance_scores_unit_interval

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

scenario_path = "./aslib_data-aslib-v4.0/"
evaluations_path = "./evaluations/"

figures_path = "../Masters_Thesis/Thesis/latex-thesis-template/gfx/plots/hinge_nn/"
figures_path = "../Masters_Thesis/masters-thesis/gfx/plots/hinge_nn/"
scenarios = ["MIP-2016", "SAT11-HAND", "CSP-2010"]
# scenarios = ["CSP-2010", "SAT11-HAND", "SAT11-INDU", "SAT11-RAND"]
lambda_values = [0.0, 0.1, 0.5, 0.9, 1.0]
epsilon_values = [1.0]
max_pairs_per_instance = 5
maxiter = 1000
seeds = [15]

learning_rates = [0.01]
batch_sizes = [128]
es_patiences = [64]
es_intervals = [8]
es_val_ratios = [0.3]

splits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

params = [scenarios, learning_rates, seeds,
          batch_sizes, es_patiences, es_intervals, es_val_ratios]

param_product = list(product(*params))
name_map = {
    "ndcg": "NDCG",
    "tau_corr": "Kendall $\\tau_b$",
    "tau_p": "Kendall $\\tau_b$ p-value",
    "mae": "MAE",
    "mse": "MSE",
    "par10": "PAR10",
    "absolute_distance_to_vbs": "MP",
    "success_rate": "SR"
}

measures = ["tau_corr", "ndcg", "mae", "mse"]

for measure in measures:
    plt.clf()
    fig, axes = plt.subplots(1, 3)
    for index, (scenario_name, learning_rate, seed, batch_size, es_patience, es_interval, es_val_ratio) in enumerate(param_product):

        ax = axes[index]
        df_baseline_lr = None
        df_baseline_rf = None
        try:
            df_baseline_lr = pd.read_csv(
                evaluations_path + "baseline-evaluation-linear-regression" +
                scenario_name + ".csv")
            df_baseline_rf = pd.read_csv(evaluations_path +
                                         "baseline-evaluation-random_forest" +
                                         scenario_name + ".csv")
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")

        params_string = "-".join([scenario_name,
            str(learning_rate), str(batch_size), str(es_patience), str(es_interval), str(es_val_ratio)])
    

        # continue
        try:
            # df_corras = pd.read_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")
            corras = pd.read_csv(
                evaluations_path + "corras-hinge-nn-" + scenario_name + ".csv")
        except:
            print("Scenario " + scenario_name +
                  " not found in corras evaluation data!")
            continue
        current_frame = corras.loc[(corras["seed"] == seed) & (
            corras["learning_rate"] == learning_rate) & (
            corras["batch_size"] == batch_size) & (corras["es_patience"] == es_patience) & (corras["es_interval"] == es_interval) & (corras["es_val_ratio"] == es_val_ratio)]
        print(len(current_frame))
        print(current_frame.head())

        if measure in ["mae", "mse"]:
            current_frame = current_frame.loc[(current_frame["lambda"] <=
                                               0.99)]
        lp = sns.lineplot(x="lambda",
                          y=measure,
                          marker="o",
                          markersize=8,
                          data=current_frame,
                          ax=ax,
                          legend=None)
        if df_baseline_rf is not None:
            lp.axes.axhline(df_baseline_rf[measure].mean(),
                            c="g",
                            ls="--",
                            label="rf-baseline-mean")
        if df_baseline_lr is not None:
            lp.axes.axhline(df_baseline_lr[measure].mean(),
                            c="m",
                            ls="--",
                            label="lr-baseline-mean")
        ax.set_title(scenario_name)
        ax.set_ylabel(name_map[measure])
        ax.set_xlabel("$\\lambda$")
        # ax.set_aspect(5.5, adjustable="box")
        # ax.legend()
#         plt.savefig(figures_path + scenario_name + "-" + params_string.replace(".","_") + "-" + measure + "-lineplot-mi.pdf")
    fig.set_size_inches(10.5, 3.0)
    # plt.subplots_adjust(right=0.85)
    fig.tight_layout()
    labels = ["Hinge-NN", "Random Forest", "Linear Regression"]
    legend = fig.legend(list(axes),
                        labels=labels,
                        loc="lower center",
                        ncol=len(labels),
                        bbox_to_anchor=(0.5, -0.02))
    plt.savefig(fname=figures_path + "-".join(scenarios) + "-" +
                params_string.replace(".", "_") + "-" + measure + ".pdf",
                bbox_extra_artists=(legend, ),
                bbox_inches="tight")
    # plt.show()
