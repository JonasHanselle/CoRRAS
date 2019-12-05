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

sns.set_style("darkgrid")

scenario_path = "./aslib_data-aslib-v4.0/"
results_path_corras = "./results/results-hinge/"
results_path_baseline = "./results/results-rf/"
figures_path = "./figures/"
scenario_names = ["SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND"]

def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(true_performances)
    return result

for scenario_name in scenario_names:

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)
    scenario.compute_rankings(False)
    relevance_scores = compute_relevance_scores_unit_interval(scenario)

    try:
        baseline = pd.read_csv(results_path_baseline + "rf-" + scenario_name + ".csv")
    except:
        print("Scenario " + scenario_name + " not found in baseline result data!")
        continue
    try:
        corras = pd.read_csv(results_path_corras + "corras-linhinge-" + scenario_name + ".csv")
    except:
        print("Scenario " + scenario_name + " not found in corras result data!")
        continue
    baseline.set_index("problem_instance", inplace=True)
    corras.set_index("problem_instance", inplace=True)
    performance_indices = [x for x in corras.columns if x.endswith("_performance")]

    baseline_rankings = baseline[performance_indices].rank(axis=1, method="min").fillna(-1).astype("int16")
    # corras_rankings = corras[performance_indices].rank(axis=1, method="min").fillna(-1).astype("int16")

    lambda_values = pd.unique(corras["lambda"])
    epsilon_values = pd.unique(corras["epsilon"])

    baseline_measures = []
    corras_measures = []

    print(scenario.performance_data)
    print(relevance_scores)

    for problem_instance, performances in scenario.performance_data.iterrows():
        tau_corr = 0
        tau_p = 0
        ndcg = 0
        mse = 0
        mae = 0
        abs_vbs_distance = 0
        par10 = 0
        true_performances = scenario.performance_data.loc[problem_instance].astype("float64").to_numpy()
        true_ranking = scenario.performance_rankings.loc[problem_instance].astype("float64").to_numpy()
        baseline_performances = baseline[performance_indices].loc[problem_instance].astype("float64").to_numpy()
        baseline_ranking = baseline[performance_indices].loc[problem_instance].astype("float64").rank(method="min").astype("int16").to_numpy()
        mse = mean_squared_error(true_performances, baseline_performances)
        mae = mean_absolute_error(true_performances, baseline_performances)
        tau_corr, tau_p = kendalltau(true_ranking, baseline_ranking)
        abs_vbs_distance = compute_distance_to_vbs(baseline_performances, true_performances)
        ndcg = ndcg_at_k(baseline_ranking,relevance_scores.loc[problem_instance].to_numpy(), len(scenario.algorithms))
        par10 = true_performances[np.argmin(baseline_performances)]
        baseline_measures.append([problem_instance,tau_corr,tau_p,ndcg,mse,mae,abs_vbs_distance,par10])
        # print(corras.loc[problem_instance])
        for lambda_value, epsilon_value in  product(lambda_values,epsilon_values):
            tau_corr = 0
            tau_p = 0
            ndcg = 0
            mse = 0
            mae = 0
            abs_vbs_distance = 0
            par10 = 0
            # print(corras)
            corras_performances = corras.loc[(corras["lambda"] == lambda_value) & (corras["epsilon"] == epsilon_value)].loc[problem_instance][performance_indices].astype("float64").to_numpy()
            # print(corras.loc[problem_instance])
            corras_ranking = corras.loc[(corras["lambda"] == lambda_value) & (corras["epsilon"] == epsilon_value)].loc[problem_instance][performance_indices].astype("float64").rank(method="min").fillna(-1).astype("int16").to_numpy()
            if np.isinf(corras_performances).any():
                # print("Warning, NaN in performance prediction for " + problem_instance + "!")
                continue
            # print(true_ranking, corras_ranking)
            tau_corr, tau_p = kendalltau(true_ranking, corras_ranking)
            mse = mean_squared_error(true_performances, corras_performances)
            mae = mean_absolute_error(true_performances, corras_performances)
            abs_vbs_distance = compute_distance_to_vbs(corras_performances, true_performances)
            ndcg = ndcg_at_k(corras_ranking,relevance_scores.loc[problem_instance].to_numpy(), len(scenario.algorithms))
            par10 = true_performances[np.argmin(corras_performances)]
            corras_measures.append([problem_instance,lambda_value,epsilon_value,tau_corr,tau_p,ndcg,mse,mae,abs_vbs_distance,par10])

    df_baseline = pd.DataFrame(data=baseline_measures,columns=["problem_instance", "tau_corr", "tau_p", "ndcg", "mse", "mae","abs_distance_to_vbs", "par10"])
    df_corras = pd.DataFrame(data=corras_measures,columns=["problem_instance", "lambda", "epsilon", "tau_corr", "tau_p", "ndcg", "mse", "mae","abs_distance_to_vbs", "par10"])

    # print(df_baseline["par10"].median())
    # print(df_corras["par10"].median())

    # plot average kendall tau
    plt.clf()
    bp = sns.boxplot(x="lambda", y="tau_corr", hue="epsilon", data=df_corras)
    bp.axes.axhline(df_baseline["tau_corr"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["tau_corr"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-tau-boxplot.pdf")

    # plot average mse
    plt.clf()
    bp = sns.boxplot(x="lambda", y="mse", data=df_corras, hue="epsilon", showfliers=False)
    bp.axes.axhline(df_baseline["mse"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["mse"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-mse-boxplot.pdf")

    # plot average mae
    plt.clf()
    bp = sns.boxplot(x="lambda", y="mae", data=df_corras, hue="epsilon")
    bp.axes.axhline(df_baseline["mae"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["mae"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-mae-boxplot.pdf")

    # plot absolute distance to vbs
    plt.clf()
    bp = sns.boxplot(x="lambda", y="abs_distance_to_vbs", data=df_corras, hue="epsilon", showfliers=False)
    bp.axes.axhline(df_baseline["abs_distance_to_vbs"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["abs_distance_to_vbs"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-vbs-boxplot.pdf")

    # plot absolute distance to vbs
    plt.clf()
    bp = sns.boxplot(x="lambda", y="ndcg", data=df_corras, hue="epsilon", showfliers=False)
    bp.axes.axhline(df_baseline["ndcg"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["ndcg"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-ndcg-boxplot.pdf")

    plt.clf()
    bp = sns.boxplot(x="lambda", y="par10", data=df_corras, hue="epsilon", showfliers=False)
    bp.axes.axhline(df_baseline["par10"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["par10"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-par10-boxplot.pdf")

    # plot lineplots for tau and ndcg of lambda
    plt.clf()
    bp = sns.lineplot(x="lambda", y="tau_corr", hue="epsilon", data=df_corras)
    bp.axes.axhline(df_baseline["tau_corr"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["tau_corr"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-tau-lineplot.pdf")

    plt.clf()
    bp = sns.lineplot(x="lambda", y="ndcg", hue="epsilon", data=df_corras)
    bp.axes.axhline(df_baseline["ndcg"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["ndcg"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-ndcg-lineplot.pdf")

    plt.clf()
    bp = sns.lineplot(x="lambda", y="abs_distance_to_vbs", hue="epsilon", data=df_corras)
    bp.axes.axhline(df_baseline["abs_distance_to_vbs"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["abs_distance_to_vbs"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-vbs-lineplot.pdf")


    plt.clf()
    bp = sns.lineplot(x="lambda", y="par10", hue="epsilon", data=df_corras)
    bp.axes.axhline(df_baseline["par10"].mean(), c="g", ls="--", label="rf-baseline-mean")
    # bp.axes.axhline(df_baseline["par10"].median(), ls="--", color="r", label="rf-baseline-median")
    plt.title(scenario_name)
    plt.legend()
    plt.savefig(figures_path+scenario_name+"-par10-lineplot.pdf")