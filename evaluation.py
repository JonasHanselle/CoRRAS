import numpy as np
import pandas as pd
from Corras.Scenario.aslib_ranking_scenario import ASRankingScenario

# measures
from scipy.stats import kendalltau, describe
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Corras.Evaluation.evaluation import ndcg_at_k, compute_relevance_scores_equi_width 

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

scenario_name = "CPMP-2015"
scenario_path = "./aslib_data-aslib-v4.0/"
results_path = "./results/"
figures_path = "./figures/"

def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(true_performances)
    return result

scenario = ASRankingScenario()
scenario.read_scenario(scenario_path + scenario_name)
scenario.compute_rankings(False)
relevance_scores = compute_relevance_scores_equi_width(scenario)

print(scenario.performance_data)
print(relevance_scores)

baseline = pd.read_csv(results_path + "rf-" + scenario_name + ".csv")
corras = pd.read_csv(results_path + "corras-" + scenario_name + ".csv")
baseline.set_index("problem_instance", inplace=True)
corras.set_index("problem_instance", inplace=True)
performance_indices = [x for x in corras.columns if x.endswith("_performance")]

baseline_rankings = baseline[performance_indices].rank(axis=1, method="min").fillna(-1).astype("int16")
# corras_rankings = corras[performance_indices].rank(axis=1, method="min").fillna(-1).astype("int16")

lambda_values = pd.unique(corras["lambda"])

baseline_measures = []
corras_measures = []

for problem_instance, performances in scenario.performance_data.iterrows():
    tau_corr = 0
    tau_p = 0
    ndcg = 0
    mse = 0
    mae = 0
    abs_vbs_distance = 0
    true_performances = scenario.performance_data.loc[problem_instance].astype("float64").to_numpy()
    true_ranking = scenario.performance_rankings.loc[problem_instance].astype("float64").to_numpy()
    baseline_performances = baseline[performance_indices].loc[problem_instance].astype("float64").to_numpy()
    baseline_ranking = baseline[performance_indices].loc[problem_instance].astype("float64").rank(method="min").astype("int16").to_numpy()
    mse = mean_squared_error(true_performances, baseline_performances)
    mae = mean_absolute_error(true_performances, baseline_performances)
    tau_corr, tau_p = kendalltau(true_ranking, baseline_ranking)
    abs_vbs_distance = compute_distance_to_vbs(baseline_performances, true_performances)
    ndcg = ndcg_at_k(baseline_ranking,relevance_scores.loc[problem_instance].to_numpy(), len(scenario.algorithms))
    baseline_measures.append([problem_instance,tau_corr,tau_p,ndcg,mse,mae,abs_vbs_distance])
    for lambda_value in lambda_values:
        tau_corr = 0
        tau_p = 0
        ndcg = 0
        mse = 0
        mae = 0
        abs_vbs_distance = 0
        corras_performances = corras.loc[(corras["lambda"] == lambda_value)].loc[problem_instance][performance_indices].astype("float64").to_numpy()
        corras_ranking = corras.loc[(corras["lambda"] == lambda_value)].loc[problem_instance][performance_indices].astype("float64").rank(method="min").fillna(-1).astype("int16").to_numpy()
        if np.isinf(corras_performances).any():
            print("Warning, NaN in performance prediction for " + problem_instance + "!")
            continue
        tau_corr, tau_p = kendalltau(true_ranking, corras_ranking)
        mse = mean_squared_error(true_performances, corras_performances)
        mae = mean_absolute_error(true_performances, corras_performances)
        abs_vbs_distance = compute_distance_to_vbs(corras_performances, true_performances)
        ndcg = ndcg_at_k(corras_ranking,relevance_scores.loc[problem_instance].to_numpy(), len(scenario.algorithms))
        corras_measures.append([problem_instance,lambda_value,tau_corr,tau_p,ndcg,mse,mae,abs_vbs_distance])

df_baseline = pd.DataFrame(data=baseline_measures,columns=["problem_instance", "tau_corr", "tau_p", "ndcg", "mse", "mae","abs_distance_to_vbs"])
df_corras = pd.DataFrame(data=corras_measures,columns=["problem_instance", "lambda", "tau_corr", "tau_p", "ndcg", "mse", "mae","abs_distance_to_vbs"])

# plot average kendall tau
plt.clf()
bp = sns.boxplot(x="lambda", y="tau_corr", data=df_corras)
bp.axes.axhline(df_baseline["tau_corr"].mean(), ls="--", label="rf-baseline")
plt.title(scenario_name)
plt.legend()
plt.savefig(figures_path+scenario_name+"-tau-boxplot.pdf")

# plot average mse
plt.clf()
bp = sns.boxplot(x="lambda", y="mse", data=df_corras, showfliers=False)
bp.axes.axhline(df_baseline["mse"].mean(), ls="--", label="rf-baseline")
plt.title(scenario_name)
plt.legend()
plt.savefig(figures_path+scenario_name+"-mse-boxplot.pdf")

# plot average mae
plt.clf()
bp = sns.boxplot(x="lambda", y="mae", data=df_corras)
bp.axes.axhline(df_baseline["mae"].mean(), ls="--", label="rf-baseline")
plt.title(scenario_name)
plt.legend()
plt.savefig(figures_path+scenario_name+"-mae-boxplot.pdf")

# plit absolute distance to vbs
plt.clf()
bp = sns.boxplot(x="lambda", y="abs_distance_to_vbs", data=df_corras, showfliers=False)
bp.axes.axhline(df_baseline["abs_distance_to_vbs"].mean(), ls="--", label="rf-baseline")
plt.title(scenario_name)
plt.legend()
plt.savefig(figures_path+scenario_name+"-vbs-boxplot.pdf")

# plit absolute distance to vbs
plt.clf()
bp = sns.boxplot(x="lambda", y="ndcg", data=df_corras, showfliers=False)
bp.axes.axhline(df_baseline["ndcg"].mean(), ls="--", label="rf-baseline")
plt.title(scenario_name)
plt.legend()
plt.savefig(figures_path+scenario_name+"-ndcg.pdf")