import os
import re
import pandas as pd
import numpy as np

os.chdir('/home/davtyan.edd/projects/EIRegression/')

def compute_weighted_accuracy(actual_values, predicted_buckets, bins, n_buckets, similarity_matrices_dir):
    """
    Computes the Weighted Accuracy metric using the similarity matrices.
    :param actual_values: Actual target values (y_test)
    :param predicted_buckets: Predicted bucket indices from the classifier
    :param bins: Bins used for bucketing the data
    :param n_buckets: Number of buckets
    :param similarity_matrices_dir: Directory where similarity matrices are stored
    :return: Weighted Accuracy score
    """
    # Assign buckets to actual_values
    min_value, max_value = actual_values.min(), actual_values.max()
    extended_bins = [min(min_value, bins[0])] + list(bins[1:-1]) + [max(max_value, bins[-1])]
    actual_buckets = pd.cut(actual_values, bins=extended_bins, labels=False, include_lowest=True)

    # Define a pattern to match only the n_buckets part of the filename
    pattern = re.compile(rf".*_{n_buckets}_buckets\.npy$")

    # Search for the file in the similarity_matrices_dir directory
    similarity_matrix_filename = next(
        (f for f in os.listdir(similarity_matrices_dir) if pattern.match(f)), None
    )
    
    if not similarity_matrix_filename:
        raise FileNotFoundError(
            f"No similarity matrix found for {n_buckets} buckets in {similarity_matrices_dir}"
        )
    
    similarity_matrix_path = os.path.join(similarity_matrices_dir, similarity_matrix_filename)
    similarity_matrix = np.load(similarity_matrix_path)

    # Ensure buckets are integers starting from 0
    actual_buckets = np.array(actual_buckets).astype(int)
    predicted_buckets = np.array(predicted_buckets).astype(int)

    # Compute similarity scores for each sample
    similarity_scores = []
    for true_bucket, pred_bucket in zip(actual_buckets, predicted_buckets):
        similarity = similarity_matrix[true_bucket, pred_bucket]
        similarity_scores.append(similarity)

    # Compute Weighted Accuracy
    weighted_accuracy = np.mean(similarity_scores)
    return weighted_accuracy


def replace_nan_median(matrix: np.ndarray, medians: np.ndarray = None) -> np.ndarray:
    """
    Replace nan values with the median of the columns in the dataset. If medians is None, it will be calculated and returned.
    If medians is not None, it will be used to replace nan values.
    :param matrix: matrix to replace nan values
    :param medians: medians to use to replace nan values
    :return: medians used to replace nan values
    """
    if medians is None:
        num_samples, num_features = matrix.shape
        medians = np.zeros(num_features)
        for i, col in enumerate(matrix.T):
            medians[i] = np.nanmedian(col)
            if np.isnan(medians[i]):
                print(f'error at feature {i}')
                medians[i] = 0.0
    for i, col in enumerate(matrix.T):
        matrix.T[i][np.isnan(col)] = medians[i]
    return medians


def bucketing(data, bins, type):
    """
    Bucketing the data into bins

    :param data: array to bucket
    :param bins: number of bins
    :param type: type of bucketing('ranged'/'quantile'/'max_score')
    :return: (array of buckets, bins)
    """
    if type == "ranged":
        return pd.cut(data, bins=bins,
                      labels=False, retbins=True, duplicates="raise")
    elif type == "quantile":
        return pd.qcut(data, q=bins,
                       labels=False, retbins=True, duplicates="raise")
    elif type == "max_score":
        sorted_array = np.sort(np.array(data))
        total = sorted_array.sum()
        jump = total/bins
        count = 0
        group_number = 0
        sorted_groups = {}
        bins = [min(data)]
        groups = np.zeros_like(data, dtype=np.int8)
        for i in range(len(sorted_array)):
            if count > jump*(group_number+1):
                group_number += 1
                bins += [sorted_array[i]]
            sorted_groups[sorted_array[i]] = group_number
            count += sorted_array[i]
        bins += [max(data)]
        for i in range(len(groups)):
            groups[i] = sorted_groups[data[i]]
        return (groups, np.array(bins))
    else:
        print("type must be 'ranged', 'quantile' or 'max_score'")
        return ([], [])
