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

scenario_names = ["SAT11-RAND", "MIP-2016", "CSP-2010", "SAT11-INDU", "SAT11-HAND"]
scenario_path = "./aslib_data-aslib-v4.0/"
loss_hists_path = "./loss-hists/"
figures_path = "../Masters_Thesis/Thesis/latex-thesis-template/gfx/plots/pl/losses"
lambda_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
epsilon_values = [0.2,0.4,0.6,0.8,1]
split = 4
seed = 15

# /home/jonas/Documents/CoRRAS/loss-hists/loss-hist-hinge-SAT11-RAND-1.0-0.6-4-15.csv

for lambda_value, epsilon_value, scenario_name in product(lambda_values,epsilon_values, scenario_names):
    figure_name = "loss-hist-hinge" + "-" + scenario_name + "-" + str(lambda_value) + "-" + str(epsilon_value) +  "-" + str(split) + "-" + str(seed)
    filename = loss_hists_path + figure_name
    figure_file = figure_name + ".pdf"
    filename += ".csv"
    df = pd.read_csv(filename)
    df = df.rename(columns={"call":"iteration"})
    df["$\lambda$ MSE"] = lambda_value * df["MEAN_SQUARED_ERROR"]
    df["$(1 - \lambda)$ SQH"] = (1 - lambda_value) * df["SQUARED_HINGE"]
    df["TOTAL_LOSS"] = df["$\lambda$ MSE"] + df["$(1 - \lambda)$ SQH"]
    df = df.melt(id_vars=["iteration"])
    print(df.head())
    text = "$\lambda$ = " + str(lambda_value) + ", "
    text += "$\epsilon$ = " + str(epsilon_value) + ", "
    text += "split = " + str(split) + ", "
    text += "seed = " + str(seed) + " "
    plt.clf()
    # plt.tight_layout()
    plt.annotate(text,(0,0), (0,-40), xycoords="axes fraction", textcoords="offset points", va="top")
    lp = sns.lineplot(x="iteration", y="value", hue="variable", data=df)
    plt.title(scenario_name)
    plt.savefig(figures_path+figure_file, bbox_inches="tight")