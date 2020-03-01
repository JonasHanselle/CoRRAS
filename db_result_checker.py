import sys

import autograd.numpy as np
import pandas as pd

from itertools import product

# evaluation stuff
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import kendalltau

# Corras
from Corras.Scenario import aslib_ranking_scenario
from Corras.Util import ranking_util as util

# Database
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, MetaData
from sqlalchemy.sql import exists, select, and_, or_
import urllib

db_url = sys.argv[1]
db_user = sys.argv[2]
db_pw = urllib.parse.quote_plus(sys.argv[3])
db_db = sys.argv[4]

scenario_names = ["MIP-2016", "CSP-2010", "SAT11-HAND",
                  "SAT11-INDU", "SAT11-RAND", "CPMP-2015"]

# for scenario_name in scenario_names:

# scenario = aslib_ranking_scenario.ASRankingScenario()
# scenario.read_scenario("aslib_data-aslib-v4.0/"+scenario_name)
splits = [1,2,3,4,5,6,7,8,9,10]
lambda_values = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                 0.7, 0.8, 0.9, 0.99, 0.999, 1.0]
epsilon_values = [0.1, 0.2, 0.3]
# lambda_values = [0.5]
max_pairs_per_instance = 5
maxiter = 100
seeds = [15]
use_quadratic_transform_values = [True, False]
use_max_inverse_transform_values = ["none", "max_cutoff", "max_par10"]
scale_target_to_unit_interval_values = [True, False]

params = [splits, lambda_values, epsilon_values, seeds, use_quadratic_transform_values,
          use_max_inverse_transform_values, scale_target_to_unit_interval_values]

param_product = list(product(*params))

# database_db = None

table_name = "linear-squared-hinge-" + scenario_names[0]

engine = sql.create_engine("mysql://" + db_user +
                           ":" + db_pw + "@" + db_url + "/" + db_db, echo=False)

if not engine.dialect.has_table(engine, table_name):
    print("Table not present")

# meta = MetaData(engine)
# experiments = Table(table_name, meta, autoload=True, autoload_with=engine)

# connection = engine.connect()
# for split, lambda_value, epsilon_value, seed, use_quadratic_transform_value, use_max_inverse_transform_value, scale_target_to_unit_interval_value in param_product: 

#     slct = experiments.select(and_(experiments.columns["split"] == split,
#                                 experiments.columns["lambda"] == lambda_value,
#                                 experiments.columns["epsilon"] == epsilon_value,
#                                 experiments.columns["use_quadratic_transform"] == use_quadratic_transform_value,
#                                 experiments.columns["use_max_inverse_transform"] == use_max_inverse_transform_value,
#                                 experiments.columns["scale_target_to_unit_interval"] == scale_target_to_unit_interval_value
#                                 ))
#     rs = connection.execute(slct)
#     if rs.first() == None:
#         print("no result")
#     else:
#         print("result")
# connection.close()




# connection.execute(ins, [{"id": 999, "split": 874, "problem_instance": "kaese",
#                         "astar-symmulgt-transmul_performance": 777,
#                         "astar-symmullt-transmul_performance": 999,
#                         "idastar-symmulgt-transmul_performance": 555,
#                         "idastar-symmullt-transmul_performance": 111}])

# for lambda_value, seed, use_quadratic_transform_values, use_max_inverse_transform_value, scale_target_to_unit_interval_value in param_product:


# connection.close()
