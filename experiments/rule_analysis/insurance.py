# insurance.py

import os
import sys
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix

# Get the absolute path of the current script
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

# Dynamically append the project path to sys.path and set the working directory
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from EIRegressor.EmbeddedInterpreter import EmbeddedInterpreter
from EIRegressor.utils import compute_weighted_accuracy
from rule_analysis.compare_rules import process_rule_similarities
from utils import (
    compute_and_save_average_similarity_matrix,
    update_results_with_weighted_accuracy,
    execute
)


def load_and_preprocess_insurance():
    # Load dataset
    dataset_path = os.path.join(PROJECT_ROOT, "experiments/datasets/insurance.csv")
    data = pd.read_csv(dataset_path)
    target = "charges"

    # Data Preprocessing
    data = pd.get_dummies(data, drop_first=True)
    data = data[data[target].notna()]

    # Ensure all data is numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna(axis=1, how='all')

    # Separate features and target
    X = data.drop(target, axis=1).values.astype(np.float64)
    y = data[target].values.astype(np.float64)
    column_names = data.drop(target, axis=1).columns.tolist()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )

    return X_train, X_test, y_train, y_test, column_names



def run_multiple_executions(save_dir, num_buckets, num_iterations, dataset_name):
    """
    Run multiple executions of the EI Regression model across different bucket counts and iterations.

    Parameters:
    - save_dir (str): Directory to save all results.
    - num_buckets (int): Maximum number of buckets to experiment with.
    - num_iterations (int): Number of iterations per bucket count.
    - dataset_name (str): Name of the dataset being experimented on.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(
        save_dir,
        f"{dataset_name}_results_{num_buckets}_buckets_{num_iterations}_iterations.json"
    )

    all_results = {}
    # Check if the consolidated results file exists and load it
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            try:
                all_results = json.load(json_file)
            except json.JSONDecodeError:
                print(f"Warning: JSON file {all_results_file_path} is corrupted. Starting fresh.")
                all_results = {}

    # Load and preprocess data
    X_train, X_test, y_train, y_test, column_names = load_and_preprocess_insurance()

    for n_buckets in range(2, num_buckets + 1):
        bucket_key = f"{n_buckets}_buckets"
        bucket_results = all_results.get(bucket_key, [])

        for iteration in range(1, num_iterations + 1):
            # Construct the expected path for the results of this iteration
            expected_result_path = os.path.join(
                save_dir, "rules",
                f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt"
            )

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
                try:
                    results = execute(
                        save_dir=save_dir,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        column_names=column_names,
                        n_buckets=n_buckets,
                        i=iteration,
                        bucketing_method="quantile",
                        statistic="median",
                        threshold=0.2
                    )
                    bucket_results.append(results)
                    # Update the all_results dict
                    all_results[bucket_key] = bucket_results
                    # Save to JSON after each iteration
                    with open(all_results_file_path, 'w') as json_file:
                        json.dump(all_results, json_file, indent=4)
                except Exception as e:
                    print(f"Error during execution for {n_buckets} buckets, iteration {iteration}: {e}")

        # After all iterations for the current n_buckets, compute weighted accuracy and average similarity matrix
        completed_iterations = len(bucket_results)
        if completed_iterations == num_iterations:
            print(f"Processing results for {n_buckets} buckets.")

            try:
                # Compute weighted accuracy and update results
                # update_results_with_weighted_accuracy(
                #     save_dir=save_dir,
                #     bucket_results=bucket_results,
                #     n_buckets=n_buckets
                # )

                # Compute average similarity matrix over all iterations
                compute_and_save_average_similarity_matrix(
                    save_dir=save_dir,
                    bucket_results=bucket_results,
                    n_buckets=n_buckets
                )

                # Update the all_results dict after post-processing
                all_results[bucket_key] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)
            except Exception as e:
                print(f"Error during post-processing for {n_buckets} buckets: {e}")
        else:
            print(f"Warning: Not all iterations completed for {n_buckets} buckets. Only {completed_iterations}/{num_iterations} completed.")

    print("All experiments completed.")


if __name__ == '__main__':
    dataset_name = "insurance_3_breaks"

    save_dir = os.path.join(
        PROJECT_ROOT,
        "experiments/rule_analysis/results",
        dataset_name
    )
    
    run_multiple_executions(
        save_dir=save_dir,
        num_buckets=5,
        num_iterations=1,
        dataset_name=dataset_name
    )