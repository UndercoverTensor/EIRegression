import sys
import os

from movies import run_multiple_executions as movies_example
from housing import run_multiple_executions as housing_example
from concrete import run_multiple_executions as concrete_example


def main():
    BUCKETING = "quantile"

    save_dir = "/home/edgar.davtyan/projects/recla_v1/examples/MLP_regression/results_fine_tuned2/"
    # save_dir = "/Users/eddavtyan/Documents/XAI/Projects/EIRegression/examples/MLP_regression/results/"
    # print("Must be called as: python examples.py --<example_name> <n_buckets> <bucketing_method>")

    # Run all three examples one after the other
    print("movies")
    movies_example(num_buckets=15,
                   num_iterations=50,
                   save_dir=os.path.join(save_dir, "movies_no_tuning"))
    # print("housing")
    # housing_example(num_buckets=15,
    #                 num_iterations=10,
    #                 save_dir=os.path.join(save_dir, "housing"))
    # print("concrete")
    # concrete_example(num_buckets=N_BUCKETS,
    #                  num_iterations=ITTERATIONS,
    #                  save_dir=os.path.join(save_dir, "concrete"))


if __name__ == '__main__':
    main()
