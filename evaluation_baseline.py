import sys

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

# Database
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import exists, select, and_, or_
import urllib
sns.set_style("darkgrid")

# DB data
db_url = sys.argv[1]
db_user = sys.argv[2]
db_pw = urllib.parse.quote_plus(sys.argv[3])
db_db = sys.argv[4]

scenario_path = "./aslib_data-aslib-v4.0/"
results_path_baseline = "./results/results-rf/"
evaluations_path = "./evaluations/"
figures_path = "./figures/"
scenario_names = ["QBF-2016"]
baselines = ["linear_regression", "random_forest"]


def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(
        predicted_performances)] - np.min(true_performances)
    return result


for scenario_name, baseline in product(scenario_names, baselines):

    scenario = ASRankingScenario()
    scenario.read_scenario(scenario_path + scenario_name)
    scenario.compute_rankings(False)
    relevance_scores = compute_relevance_scores_unit_interval(scenario)
    table_name = "baseline_" + baseline + "-" + scenario_name
    try:
        engine = sql.create_engine("mysql://" + db_user +
                                   ":" + db_pw + "@" + db_url + "/" + db_db, echo=False)
        connection = engine.connect()
        baseline_df = pd.read_sql_table(
            table_name=table_name, con=connection)
        connection.close()
    except:
        print("Scenario " + scenario_name +
              " not found in baseline result data!", table_name)
        continue
    baseline_df.set_index("problem_instance", inplace=True)
    performance_indices = [
        x for x in baseline_df.columns if x.endswith("_performance")]

    baseline_measures = []

    for problem_instance, performances in scenario.performance_data.iterrows():
        filepath = evaluations_path + "baseline-evaluation-" + \
            baseline + scenario_name + ".csv"
        feature_cost = 0
        # we use all features, so we sum up the individual costs
        if scenario.feature_cost_data is not None:
            feature_cost = scenario.feature_cost_data.loc[problem_instance].sum(
            )
        tau_corr = 0
        tau_p = 0
        ndcg = 0
        mse = 0
        mae = 0
        abs_vbs_distance = 0
        par10 = 0
        par10_with_feature_cost = 0
        true_performances = scenario.performance_data.loc[problem_instance].astype(
            "float64").to_numpy()
        true_ranking = scenario.performance_rankings.loc[problem_instance].astype(
            "float64").to_numpy()
        baseline_performances = baseline_df[performance_indices].loc[problem_instance].astype(
            "float64").to_numpy()
        baseline_ranking = baseline_df[performance_indices].loc[problem_instance].astype(
            "float64").rank(method="min").astype("int16").to_numpy()
        run_stati = scenario.runstatus_data.loc[problem_instance]
        mse = mean_squared_error(true_performances, baseline_performances)
        mae = mean_absolute_error(true_performances, baseline_performances)
        tau_corr, tau_p = kendalltau(true_ranking, baseline_ranking)
        abs_vbs_distance = compute_distance_to_vbs(
            baseline_performances, true_performances)
        ndcg = ndcg_at_k(baseline_ranking, relevance_scores.loc[problem_instance].to_numpy(
        ), len(scenario.algorithms))
        par10 = true_performances[np.argmin(baseline_performances)]
        par10_with_feature_cost = par10 + feature_cost
        run_status = run_stati.iloc[np.argmin(baseline_performances)]
        baseline_measures.append(
            [problem_instance, tau_corr, tau_p, ndcg, mse, mae, abs_vbs_distance, par10, par10_with_feature_cost, run_status])

    df_baseline = pd.DataFrame(data=baseline_measures, columns=[
                               "problem_instance", "tau_corr", "tau_p", "ndcg", "mse", "mae", "abs_distance_to_vbs", "par10", "par10_with_feature_cost", "run_status"])
    print(df_baseline)
    df_baseline.to_csv(filepath)
