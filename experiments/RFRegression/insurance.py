import sys
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Get the absolute path of the current script
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

# Dynamically append the project path to sys.path and set the working directory
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.model_optimizer import ModelOptimizer

def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", single_rules_breaks=3):
    # Load dataframe
    dataset_path = os.path.join(PROJECT_ROOT, "experiments/datasets/insurance.csv")
    data = pd.read_csv(dataset_path)

    target = "charges"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    regressor_hp_grid = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__bootstrap': [True, False]
    }

    regressor_default_args = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True
    }

    regressor_optimizer = ModelOptimizer(search_method="grid")

    # Creation of EI Regression with Random Forest
    eiReg = EmbeddedInterpreter(regressor=RandomForestRegressor,
                                model_optimizer=regressor_optimizer,
                                model_preprocessor=None,
                                n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                reg_default_args=regressor_default_args,
                                reg_hp_args=regressor_hp_grid,
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True,
                                force_precompute=True, device="cpu")

    eiReg.fit(X_train, y_train,
              add_single_rules=True, single_rules_breaks=single_rules_breaks, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
    buck_pred, y_pred = eiReg.predict(X_test, return_buckets=True)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    acc, f1, cm = eiReg.evaluate_classifier(X_test, y_test)

    top_uncertainties = eiReg.get_top_uncertainties()

    results = {"R2": r2,
               "MAE": mae,
               "MSE": mse,
               "Accuracy": acc,
               "F1": f1,
               "Confusion Matrix": cm.tolist(),
               "Uncertainties": top_uncertainties}

    save_results = os.path.join(save_dir, "rules")
    os.makedirs(save_results, exist_ok=True)
    eiReg.rules_to_txt(os.path.join(save_results, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
                       results=results)

    return results


def run_multiple_executions(save_dir, num_buckets, num_iterations, single_rules_breaks=3):
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(save_dir, f"results_{num_buckets}_buckets_{num_iterations}_iterations.json")

    all_results = {}
    # Check if the consolidated results file exists and load it
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(1, num_buckets + 1):
        bucket_results = all_results.get(f"{n_buckets}_buckets", [])

        for iteration in range(1, num_iterations + 1):
            # Construct the expected path for the results of this iteration
            expected_result_path = os.path.join(save_dir, "rules",
                                                f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
                results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration, single_rules_breaks=single_rules_breaks)
                bucket_results.append(results)
                all_results[f"{n_buckets}_buckets"] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)


if __name__ == '__main__':
    save_dir = os.path.join(PROJECT_ROOT, "experiments/RFRegression/insurance")
    
    run_multiple_executions(
        save_dir=save_dir,
        num_buckets=10,
        num_iterations=25
    )