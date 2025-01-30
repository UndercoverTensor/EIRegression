import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter


def execute(n_buckets=3, bucketing_method="quantile"):
    # Load dataframe
    data = pd.read_csv("examples/datasets/insurance.csv")
    target = "charges"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    # Creation of EI Regression with Gradient Boosting
    eiReg = EmbeddedInterpreter(GradientBoostingRegressor,
                                n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                reg_args={"loss": "absolute_error"},
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True,
                                force_precompute=True, device="cuda")
    eiReg.fit(X_train, y_train,
              reg_args={},
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
    y_pred = eiReg.predict(X_test)

    print("R2: ", r2_score(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    eiReg.print_most_important_rules()

    results = {"R2": r2_score(y_test, y_pred),
               "MAE": mean_absolute_error(y_test, y_pred)}
    eiReg.rules_to_txt(
        "examples/results/insurance_results.txt", results=results)
