# delta_elevators.py

import os
import sys
import json
import pandas as pd
import warnings

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


def load_delta_elevators_data(dataset_path=None):
    """
    Loads and preprocesses the training and testing data for the delta_elevators dataset.

    Parameters:
    - dataset_path (str): Path to the dataset directory containing data and domain files.

    Returns:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing target.
    - column_names (list): List of feature column names.
    """
    if dataset_path is None:
        dataset_path = os.path.join(PROJECT_ROOT, "experiments/datasets/Elevators")

    # Load domain file for column names
    domain_file = os.path.join(dataset_path, "delta_elevators.domain")
    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain file not found at {domain_file}")

    domain_data = pd.read_csv(
        domain_file,
        delim_whitespace=True, header=None
    )
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Load data file
    data_file = os.path.join(dataset_path, "delta_elevators.data")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at {data_file}")

    data = pd.read_csv(
        data_file,
        delim_whitespace=True, header=None
    )
    data.columns = column_names

    # Set the target column
    target = "Se"

    # Data Preprocessing
    data = data.apply(pd.to_numeric, errors='ignore')

    # Separate target from features
    y = data[target]
    X = pd.get_dummies(data.drop(columns=[target]), drop_first=True)

    # Align the features if needed (not splitting here as original code uses train_test_split)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.33, random_state=42
    )

    feature_columns = X.columns.tolist()

    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Testing samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")

    return X_train, y_train, X_test, y_test, feature_columns


def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", single_rules_breaks=3):
    """
    Executes a single Random Forest regression experiment with specified parameters.

    Parameters:
    - save_dir (str): Directory to save the results.
    - n_buckets (int): Number of buckets for bucketing method.
    - i (int): Iteration number.
    - bucketing_method (str): Method for bucketing.
    - single_rules_breaks (int): Number of breaks for single rules.

    Returns:
    - dict: Dictionary containing evaluation metrics and uncertainties.
    """
    # Load and preprocess data
    X_train, y_train, X_test, y_test, feature_columns = load_delta_elevators_data()

    # Define hyperparameter grid for Random Forest
    regressor_hp_grid = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__bootstrap': [True, False],
        'regressor__n_jobs': [-1]
    }

    # Define default hyperparameters for Random Forest, including n_jobs=-1 to utilize all cores
    regressor_default_args = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True
    }

    # Initialize the model optimizer with grid search
    regressor_optimizer = ModelOptimizer(search_method="grid")

    # Initialize the Embedded Interpreter for Random Forest Regression
    eiReg = EmbeddedInterpreter(
        regressor=RandomForestRegressor,
        model_optimizer=regressor_optimizer,
        model_preprocessor=None,
        n_buckets=n_buckets,
        bucketing_method=bucketing_method,
        reg_default_args=regressor_default_args,
        reg_hp_args=regressor_hp_grid,
        max_iter=4000,
        lossfn="MSE",
        min_dloss=0.0001,
        lr=0.005,
        precompute_rules=True,
        force_precompute=True,
        device="cuda",
        verbose=True
    )

    # Fit the model
    eiReg.fit(
        X_train, y_train,
        add_single_rules=True,
        single_rules_breaks=single_rules_breaks,
        add_multi_rules=True,
        column_names=feature_columns
    )

    # Predict on test data
    y_pred = eiReg.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    acc, f1, cm = eiReg.evaluate_classifier(X_test, y_test)

    # Get top uncertainties
    top_uncertainties = eiReg.get_top_uncertainties()

    # Compile results
    results = {
        "R2": r2,
        "MAE": mae,
        "MSE": mse,
        "Accuracy": acc,
        "F1": f1,
        "Confusion Matrix": cm.tolist(),
        "Uncertainties": top_uncertainties
    }

    # Save the rules and results
    save_results = os.path.join(save_dir, "rules")
    os.makedirs(save_results, exist_ok=True)
    rule_filename = f"rule_results_{n_buckets}_buckets_{i}_iterations.txt" if i is not None else f"rule_results_{n_buckets}_buckets.txt"
    eiReg.rules_to_txt(
        os.path.join(save_results, rule_filename),
        results=results
    )

    return results


def run_multiple_executions(save_dir, num_buckets, num_iterations, dataset_name='delta_elevators', single_rules_breaks=3):
    """
    Runs multiple Random Forest regression experiments across different bucket numbers and iterations.

    Parameters:
    - save_dir (str): Directory to save all results.
    - num_buckets (int): Maximum number of buckets to iterate through.
    - num_iterations (int): Number of iterations per bucket.
    - dataset_name (str): Name of the dataset (used for result file naming).
    - single_rules_breaks (int): Number of breaks for single rules.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(
        save_dir, f"{dataset_name}_results_{num_buckets}_buckets_{num_iterations}_iterations.json"
    )

    all_results = {}
    # Load existing results if available
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(1, num_buckets + 1):
        bucket_key = f"{n_buckets}_buckets"
        bucket_results = all_results.get(bucket_key, [])

        for iteration in range(1, num_iterations + 1):
            # Define the path for the current iteration's results
            expected_result_path = os.path.join(
                save_dir, "rules",
                f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt"
            )

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} bucket(s), iteration {iteration}")
                results = execute(
                    save_dir=save_dir,
                    n_buckets=n_buckets,
                    i=iteration,
                    single_rules_breaks=single_rules_breaks
                )
                bucket_results.append(results)
                all_results[bucket_key] = bucket_results

                # Save the consolidated results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)

if __name__ == '__main__':
    dataset_name = "delta_elevators_3_breaks"    
    save_dir = os.path.join(PROJECT_ROOT, "experiments/RFRegression/results_debugging/", dataset_name)

    save_dir = os.path.join(
        "examples/RFRegression/results_debugging/",
        dataset_name
    )

    run_multiple_executions(
        save_dir=save_dir,
        num_buckets=10,
        num_iterations=15,
        dataset_name=dataset_name
    )

    print("\n" + "="*47)
    print("✨🎉   Thank you for using this program!   🎉✨")
    print("        🚀 Program executed successfully 🚀")
    print("="*47 + "\n")