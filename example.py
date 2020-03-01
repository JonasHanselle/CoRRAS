import numpy as np
import pandas as pd
import itertools as it
import random
from aslib_scenario import aslib_scenario
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def construct_numpy_representation_only_performances(
    features: pd.DataFrame, performances: pd.DataFrame):
    """Get numpy representation of features, performances

    Arguments:
        features {pd.DataFrame} -- Feature values
        performances {pd.DataFrame} -- Performances of algorithms

    Returns:
        [type] -- Triple of numpy ndarrays, first stores the feature
        values, the second stores the algirhtm performances and the
        third stores the algorithm rankings
    """

    joined = features.join(performances)
    np_features = joined[features.columns.values].values
    np_performances = joined[[x for x in performances.columns]].values
    return np_features, np_performances

def main():
    
    # read scneario
    scenario = aslib_scenario.ASlibScenario()
    scenario.read_scenario("aslib_data-aslib-v4.0/MIP-2016")
    
    # split scenario
    scenario.create_cv_splits(n_folds=10)
    train_scenario, test_scenario = scenario.get_split(indx=1)
    
    # create numpy representation of scenarios
    test_features, test_performances = construct_numpy_representation_only_performances(
        features=test_scenario.feature_data, performances=test_scenario.performance_data)
    train_features, train_performances = construct_numpy_representation_only_performances(
        features=train_scenario.feature_data, performances=train_scenario.performance_data)
    
    # impute missing values
    imputer = SimpleImputer()
    imputer.fit_transform(train_features)
    imputer.transform(test_features)

    # standardize feature values
    standard_scaler = StandardScaler()
    standard_scaler.fit_transform(train_features)
    standard_scaler.transform(test_features)

    # fit one linear model per label
    model = MultiOutputRegressor(estimator=LinearRegression())
    model.fit(train_features, train_performances)

    # make predictions for test data
    predicted_performances = model.predict(test_features)

    # compute mse on test data
    mse = mean_squared_error(y_true=test_performances, y_pred=predicted_performances)
    print("The model achieved a mean squared error of " + str(mse))

if __name__ == "__main__":
    main()