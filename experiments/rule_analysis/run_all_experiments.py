# run_all_experiments.py

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
# Assuming EIRegressor.utils and rule_analysis.compare_rules are available
from EIRegressor.utils import compute_weighted_accuracy
from rule_analysis.compare_rules import process_rule_similarities

# Import the execute function and other utilities from utils.py
from utils import (
    compute_and_save_average_similarity_matrix,
    execute
)

# Import the run_multiple_executions functions from your dataset scripts
from bank32NH import run_multiple_executions as run_bank32NH
from concrete import run_multiple_executions as run_concrete
from delta_elevators import run_multiple_executions as run_delta_elevators
from house_16 import run_multiple_executions as run_house_16
from housing import run_multiple_executions as run_housing
from insurance import run_multiple_executions as run_insurance
from movies import run_multiple_executions as run_movies

NUM_BUCKETS=10
NUM_ITERATIONS=25
SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments/rule_analysis_experiments/sim_def_V3_2/results_fixed")

def main():
    # Bank32NH Dataset
    # print("Starting experiments for bank32NH dataset")
    # dataset_name = "bank32NH_3_breaks"
    # save_dir = os.path.join(
    #     "examples/rule_analysis_experiments/sim_def_V3_2/results",
    #     dataset_name
    # )
    # run_bank32NH(
    #     save_dir=save_dir,
    #     num_buckets=NUM_BUCKETS,
    #     num_iterations=NUM_ITERATIONS,
    #     dataset_name=dataset_name
    # )
    # print("Finished experiments for bank32NH dataset\n")

    # # Concrete Dataset
    # print("Starting experiments for concrete dataset")
    # dataset_name = "concrete_3_breaks"
    # save_dir = os.path.join(
    #     "examples/rule_analysis_experiments/sim_def_V3_2/results",
    #     dataset_name
    # )
    # run_concrete(
    #     save_dir=save_dir,
    #     num_buckets=NUM_BUCKETS,
    #     num_iterations=NUM_ITERATIONS,
    #     dataset_name=dataset_name
    # )
    # print("Finished experiments for concrete dataset\n")

    # # Delta Elevators Dataset
    # print("Starting experiments for delta_elevators dataset")
    # dataset_name = "delta_elevators_3_breaks"
    # save_dir = os.path.join(
    #     "examples/rule_analysis_experiments/sim_def_V3_2/results",
    #     dataset_name
    # )
    # run_delta_elevators(
    #     save_dir=save_dir,
    #     num_buckets=NUM_BUCKETS,
    #     num_iterations=NUM_ITERATIONS,
    #     dataset_name=dataset_name
    # )
    # print("Finished experiments for delta_elevators dataset\n")

    # House_16 Dataset
    # print("Starting experiments for house_16 dataset")
    # dataset_name = "house_16_3_breaks"
    # save_dir = os.path.join(
    #     "examples/rule_analysis_experiments/sim_def_V3_2/results",
    #     dataset_name
    # )
    # run_house_16(
    #     save_dir=save_dir,
    #     num_buckets=NUM_BUCKETS,
    #     num_iterations=NUM_ITERATIONS,
    #     dataset_name=dataset_name
    # )
    # print("Finished experiments for house_16 dataset\n")

    # Housing Dataset
    print("Starting experiments for housing dataset")
    dataset_name = "housing_3_breaks"
    save_dir = os.path.join(
        SAVE_DIR,
        dataset_name
    )
    run_housing(
        save_dir=save_dir,
        num_buckets=NUM_BUCKETS,
        num_iterations=NUM_ITERATIONS,
        dataset_name=dataset_name
    )
    print("Finished experiments for housing dataset\n")

    # Insurance Dataset
    print("Starting experiments for insurance dataset")
    dataset_name = "insurance_3_breaks"
    save_dir = os.path.join(
        SAVE_DIR,
        dataset_name
    )
    run_insurance(
        save_dir=save_dir,
        num_buckets=NUM_BUCKETS,
        num_iterations=NUM_ITERATIONS,
        dataset_name=dataset_name
    )
    print("Finished experiments for insurance dataset\n")

    # Movies Dataset
    print("Starting experiments for movies dataset")
    dataset_name = 'movies_3_breaks'
    save_dir = os.path.join(
        SAVE_DIR,
        dataset_name
    )
    run_movies(
        save_dir=save_dir,
        num_buckets=NUM_BUCKETS,
        num_iterations=NUM_ITERATIONS,
        dataset_name=dataset_name
    )
    print("Finished experiments for movies dataset\n")

    print("All experiments completed.")

if __name__ == '__main__':
    main()