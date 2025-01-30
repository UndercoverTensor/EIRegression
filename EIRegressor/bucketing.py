# bucketing.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def bucketing(labels, bins, type, features=None):
    """
    Bucketing the labels into bins

    :param labels: array to bucket
    :param bins: number of bins
    :param type: type of bucketing('ranged'/'quantile'/'max_score'/'kmeans')
    :return: (array of buckets, bins)
    """
    if type == "ranged":
        return pd.cut(labels, bins=bins,
                      labels=False, retbins=True, duplicates="raise")
    elif type == "quantile":
        return pd.qcut(labels, q=bins,
                       labels=False, retbins=True, duplicates="drop")
    elif type == "max_score":
        sorted_array = np.sort(np.array(labels))
        total = sorted_array.sum()
        jump = total/bins
        count = 0
        group_number = 0
        sorted_groups = {}
        bins = [min(labels)]
        groups = np.zeros_like(labels, dtype=np.int8)
        for i in range(len(sorted_array)):
            if count > jump*(group_number+1):
                group_number += 1
                bins += [sorted_array[i]]
            sorted_groups[sorted_array[i]] = group_number
            count += sorted_array[i]
        bins += [max(labels)]
        for i in range(len(groups)):
            groups[i] = sorted_groups[labels[i]]
        return (groups, np.array(bins))
    elif type == "kmeans" and features is not None:
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=bins).fit(features)
        # Assign each original label to a cluster based on its corresponding feature's cluster
        clusters = kmeans.labels_
        # Create bins based on the labels sorted by the cluster assignment
        sorted_labels = sorted(zip(clusters, labels), key=lambda x: x[0])
        labels_sorted = [label for _, label in sorted_labels]
        unique_sorted_labels = np.unique(labels_sorted)
        if len(unique_sorted_labels) > bins:
            # Create bin edges ensuring they are unique and covering the entire range of labels
            bin_edges = np.percentile(unique_sorted_labels, np.linspace(0, 100, bins + 1))
        else:
            bin_edges = unique_sorted_labels.tolist() + [max(unique_sorted_labels) + 1]
        bin_edges[-1] = max(labels) + 1  # Ensure the last bin captures the maximum label
        # Use digitize to assign labels to bins
        bin_assignment = np.digitize(labels, bin_edges, right=False) - 1  # Subtract 1 to make bins start from 0
        return (bin_assignment, bin_edges)
    else:
        print("type must be 'ranged', 'quantile', 'kmeans', or 'max_score'")
        return ([], [])
