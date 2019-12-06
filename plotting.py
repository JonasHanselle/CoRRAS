
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
evaluations_path = "./evaluations/"

figures_path = "./figures/"
scenario_names = ["SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND"]

for scenario_name in scenario_names:
    df_baseline = None
    try:
        df_baseline = pd.read_csv(evaluations_path + "baseline-evaluation-" + scenario_name + ".csv")
    except:
        print("Scenario " + scenario_name + " not found in corras evaluation data!")
        # continue
    try:
        df_corras = pd.read_csv(evaluations_path + "corras-linhinge-evaluation-" + scenario_name + ".csv")
    except:
        print("Scenario " + scenario_name + " not found in corras evaluation data!")
        continue
    # plot average kendall tau

    for measure in df_corras.columns[4:]:
        plt.clf()
        bp = sns.boxplot(x="lambda", y=measure, hue="epsilon", data=df_corras)
        if df_baseline is not None:
            bp.axes.axhline(df_baseline[measure].mean(), c="g", ls="--", label="rf-baseline-mean")
        plt.title(scenario_name)
        plt.legend()
        plt.savefig(figures_path+scenario_name+"-" + measure +"-boxplot.pdf")

    for measure in df_corras.columns[4:]:
        plt.clf()
        bp = sns.lineplot(x="lambda", y=measure, hue="epsilon", data=df_corras, palette=sns.color_palette("Set1", len(pd.unique(df_corras["epsilon"]))))
        if df_baseline is not None:
            bp.axes.axhline(df_baseline[measure].mean(), c="g", ls="--", label="rf-baseline-mean")
        plt.title(scenario_name)
        plt.legend()
        plt.savefig(figures_path+scenario_name+"-" + measure +"-lineplot.pdf")