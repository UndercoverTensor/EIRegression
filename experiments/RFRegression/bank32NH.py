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


def load_bank32NH_data(dataset_path="examples/datasets/bank32NH"):
    """
    Loads and preprocesses the training and testing data.

    Parameters:
    - dataset_path (str): Path to the dataset directory containing data and domain files.

    Returns:
    - X_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target.
    - X_test (np.ndarray): Testing features.
    - y_test (np.ndarray): Testing target.
    - column_names (list): List of feature column names.
    """
    # Load dataframes
    train_data = pd.read_csv(
        os.path.join(dataset_path, "bank32nh.data"),
        delim_whitespace=True, header=None)
    test_data = pd.read_csv(
        os.path.join(dataset_path, "bank32nh.test"),
        delim_whitespace=True, header=None)

    # Load domain file for column names
    domain_data = pd.read_csv(
        os.path.join(dataset_path, "bank32nh.domain"),
        delim_whitespace=True, header=None)
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Apply column names to the train and test dataframes
    train_data.columns = column_names
    test_data.columns = column_names

    # Set the target column
    target = "rej"

    # Data Preprocessing
    train_data = train_data.apply(pd.to_numeric, errors='ignore')
    test_data = test_data.apply(pd.to_numeric, errors='ignore')

    # Separate target from features
    y_train = train_data[target]
    y_test = test_data[target]

    # Drop target from features before get_dummies
    train_features = pd.get_dummies(train_data.drop(columns=[target]), drop_first=True)
    test_features = pd.get_dummies(test_data.drop(columns=[target]), drop_first=True)

    # Align the train and test features by columns
    train_features, test_features = train_features.align(test_features, join='inner', axis=1)

    # Reattach target
    train_features[target] = y_train[train_features.index]
    test_features[target] = y_test[test_features.index]

    # Remove rows where target is NA
    train_features = train_features[train_features[target].notna()]
    test_features = test_features[test_features[target].notna()]

    # Separate features and target
    X_train = train_features.drop(target, axis=1).values
    y_train = train_features[target].values
    X_test = test_features.drop(target, axis=1).values
    y_test = test_features[target].values

    feature_columns = train_features.drop(target, axis=1).columns.tolist()

    # After aligning and reattaching target
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
    X_train, y_train, X_test, y_test, feature_columns = load_bank32NH_data()

    # Define hyperparameter grid for Random Forest
    regressor_hp_grid = {
        'regressor__n_estimators': [50, 100, 150],
        'regressor__max_depth': [10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__bootstrap': [True, False],
        'regressor__n_jobs': [-1]
    }

    # Define default hyperparameters for Random Forest
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
    eiReg.rules_to_txt(
        os.path.join(save_results, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
        results=results
    )

    return results


def run_multiple_executions(save_dir, num_buckets, num_iterations, dataset_name, single_rules_breaks=3):
    """
    Runs multiple Random Forest regression experiments across different bucket numbers and iterations.

    Parameters:
    - save_dir (str): Directory to save all results.
    - num_buckets (int): Maximum number of buckets to iterate through.
    - num_iterations (int): Number of iterations per bucket.
    - single_rules_breaks (int): Number of breaks for single rules.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(
        save_dir, 
        f"{dataset_name}_results_{num_buckets}_buckets_{num_iterations}_iterations.json"
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

            # Skip if results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
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
    dataset_name = "bank32NH_3_breaks"

    save_dir = os.path.join(
        PROJECT_ROOT, "experiments/RFRegression/results_debugging/", dataset_name
    )

    run_multiple_executions(
        save_dir=save_dir,
        num_buckets=10,
        num_iterations=25,
        dataset_name=dataset_name
    )

    print("\n" + "="*47)
    print("✨🎉   Thank you for using this program!   🎉✨")
    print("        🚀 Program executed successfully 🚀")
    print("="*47 + "\n")