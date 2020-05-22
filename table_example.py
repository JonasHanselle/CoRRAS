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


def create_latex_min(df: pd.DataFrame, decimal_format="{:10.3f}"):
    result = "\\begin{tabular}{" + "l" + "r" * (len(df.columns)) + "} \n"
    result += "\\toprule \n"
    result += "Instance &" +  " & ".join(df.columns) + " \\\\ \n"
    result += "\\midrule \n"
    for index, row in df.iterrows():
        instance = index.split(".")[-2].split("/")[-1]

        result += str(instance) + " & " + " & ".join([
            "\\textbf{" + decimal_format.format(x) +
            "}" if x == np.nanmin(row) else decimal_format.format(x)
            for x in row
        ]) + " \\\\ \n"
    result += "\\toprule \n"
    result += "\\end{tabular} \n"
    result = result.replace("nan", "-")
    return result

def compute_distance_to_vbs(predicted_performances, true_performances):
    result = true_performances[np.argmin(predicted_performances)] - np.min(
        true_performances)
    return result


scenario_path = "./aslib_data-aslib-v4.0/"
figures_path = "./figures/"

scenario_name = "SAT11-INDU"

scenario = ASRankingScenario()
scenario.read_scenario(scenario_path + scenario_name)

print(",".join(scenario.performance_data.columns[2:7]).replace("_","\_"))

# table = create_latex_min(scenario.performance_data[scenario.performance_data.columns[2:7]].sample(n=15))

# print(table.replace("_","\_"))