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
import urllib

# result_path = "./results-lr/"

scenario_names = ["QBF-2016", "SAT12-ALL", "SAT03-16_INDU", "TTP-2016", "MAXSAT-WPMS-2016", "MAXSAT-PMS-2016", "CSP-Minizinc-Time-2016"]
# scenario_names = ["CSP-Minizinc-Time-2016"]

for scenario_name in scenario_names:
    try:
        scenario = aslib_ranking_scenario.ASRankingScenario()
        scenario.read_scenario("aslib_data-aslib-v4.0/"+scenario_name)

        db_url = sys.argv[1]
        db_user = sys.argv[2]
        db_pw = urllib.parse.quote_plus(sys.argv[3])
        db_db = sys.argv[4]
        # database_db = None

        engine = sql.create_engine("mysql://" + db_user + ":" + db_pw + "@" + db_url + "/" + db_db)

        filepath = "results-pl/pl_log_linear-" + scenario_name + ".csv"
        dataframe = None
        try:
            dataframe = pd.read_csv(filepath)
        except:
            print("File " + filepath + " not found!")
            continue
        del dataframe["predicted_ranking"]
        connection = engine.connect()
        try: 
            dataframe.to_sql("linear-plackett-luce-" + scenario.scenario, connection)
        except:
            print("Table already exists")

        connection.close()
    except:
        print("Problem orccured")