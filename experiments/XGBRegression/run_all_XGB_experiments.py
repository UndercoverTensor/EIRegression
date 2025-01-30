import sys
import os

# Get the absolute path of the current script
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

# Dynamically append the project path to sys.path and set the working directory
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from movies import run_multiple_executions as movies_example
from housing import run_multiple_executions as housing_example
from concrete import run_multiple_executions as concrete_example
from insurance import run_multiple_executions as insurance_example
from house_16H import run_multiple_executions as house_16_example
from bank32NH import run_multiple_executions as bank_32_example
from delta_elevators import run_multiple_executions as delta_elevators_example  # Newly added

def main():
    BUCKETING = "quantile"

    # Define the base directory for saving results dynamically
    save_dir = os.path.join(PROJECT_ROOT, "examples/XGBRegression/results")

    print("Running all experiments...")

    print("movies_3_breaks")
    movies_example(
        num_buckets=10,
        num_iterations=3,
        save_dir=os.path.join(save_dir, "movies_3_breaks"),
        single_rules_breaks=3
    )

    print("housing_3_breaks")
    housing_example(
        num_buckets=10,
        num_iterations=3,
        save_dir=os.path.join(save_dir, "housing_3_breaks"),
        single_rules_breaks=3
    )

    print("concrete_3_breaks")
    concrete_example(
        num_buckets=10,
        num_iterations=3,
        save_dir=os.path.join(save_dir, "concrete_3_breaks"),
        single_rules_breaks=3
    )

    print("insurance_3_breaks")
    insurance_example(
        num_buckets=10,
        num_iterations=3,
        save_dir=os.path.join(save_dir, "insurance_3_breaks"),
        single_rules_breaks=3
    )

    print("house_16H_3_breaks")
    house_16_example(
        num_buckets=10,
        num_iterations=3,
        save_dir=os.path.join(save_dir, "house_16H_3_breaks"),
        single_rules_breaks=3
    )

    print("bank32NH_3_breaks")
    bank_32_example(
        num_buckets=10,
        num_iterations=1,
        save_dir=os.path.join(save_dir, "bank32NH_3_breaks"),
        single_rules_breaks=3
    )

    print("delta_elevators_3_breaks")  # Newly added
    delta_elevators_example(
        num_buckets=10,
        num_iterations=3,
        save_dir=os.path.join(save_dir, "delta_elevators_3_breaks"),
        single_rules_breaks=3
    )

    print("\n" + "=" * 50)
    print("✅ All XGB experiments (3_breaks) have been successfully executed!")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()