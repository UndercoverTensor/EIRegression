import numpy as np

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