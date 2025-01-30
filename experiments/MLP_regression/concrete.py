from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
import pandas as pd
import os
import json

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from EIRegressor.model_optimizer import ModelOptimizer


def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile"):
    # Load dataframe
    data = pd.read_csv("/home/edgar.davtyan/projects/recla_v1/examples/datasets/concrete_data.csv")
    # data = pd.read_csv("/Users/eddavtyan/Documents/XAI/Projects/EIRegression/experiments/datasets/concrete_data.csv")
    target = "concrete_compressive_strength"
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    regressor_hp_grid = {
        'regressor__hidden_layer_sizes': [(100,), (150,), (100, 100)],
        'regressor__activation': ['relu', 'tanh', 'logistic'],
        'regressor__solver': ['sgd', 'adam'],
        'regressor__learning_rate_init': [0.001, 0.01, 0.0001],
        'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
        'regressor__batch_size': [64, 128, 256],  # Consider 'auto' if applicable
        'regressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'regressor__max_iter': [200, 300, 400]
    }

    regressor_default_args = {
        "hidden_layer_sizes": (100,),  # One hidden layer with 100 neurons
        "activation": 'relu',  # ReLU activation function, f(x) = max(0, x)
        "learning_rate_init": 0.001,  # Initial learning rate
    }

    regressor_optimizer = ModelOptimizer(search_method="random", n_iter=450)

    # Creation of EI Regression with XGBoost
    eiReg = EmbeddedInterpreter(regressor=MLPRegressor,
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
              add_single_rules=True, single_rules_breaks=3, add_multi_rules=True,
              column_names=data.drop(target, axis=1).columns)
    buck_pred, y_pred = eiReg.predict(X_test, return_buckets=True)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    acc, f1 = eiReg.evaluate_classifier(X_test, y_test)

    results = {"R2": r2,
               "MAE": mae,
               "MSE": mse,
               "Accuracy": acc,
               "F1": f1}

    save_results = os.path.join(save_dir, "rules")
    os.makedirs(save_results, exist_ok=True)
    eiReg.rules_to_txt(os.path.join(save_results, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
                       results=results)

    return results


def run_multiple_executions(save_dir, num_buckets, num_iterations):
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(save_dir,
                                         f"results_{num_buckets}_buckets_{num_iterations}_iterations.json")

    all_results = {}
    # Check if the consolidated results file exists and load it
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(1, num_buckets + 1):
        bucket_results = all_results.get(f"{n_buckets}_buckets", [])

        for iteration in range(1, num_iterations + 1):
            # Construct the expected path for the results of this iteration
            expected_result_path = os.path.join(save_dir, "rules", f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
                results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration)
                bucket_results.append(results)
                all_results[f"{n_buckets}_buckets"] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)


if __name__ == '__main__':
    save_dir = "/Users/eddavtyan/Documents/XAI/Projects/EIRegression/examples/MLP_regression/results/concrete"
    run_multiple_executions(save_dir, 3, 3)
