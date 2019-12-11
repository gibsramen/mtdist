import numbers

import numpy as np
import pandas as pd


def gower_distances(X, Y, weights=None):
    """Calculate Gower distance (Gower 1971) between observations

    From daisy documentation:

     ---------------------------------------------------------------
    | d_ij = \frac{\sum_{k=1}^p w_k \delta_{ij}^{(k)} d_{ij}^{(k)}} |
    |             {\sum_{k=1}^p w_k \delta_{ij}^{(k)}}              |
     ---------------------------------------------------------------

    w_k: weight of k-th variable of p
    \delta_{ij}^{(i)}: 0 when k-th variable is missing in at least one
        row, 1 otherwise
    d_{ij}^{(k)}: 1 when k-th variable matches between the two rows,
        1 otherwise

    References:
        - Gower, J. C. (1971) A general coefficient of similarity and
          some of its properties, Biometrics 27, 857--874.

    Parameters:
    -----------
        X: array-like with shape (n_samples_X, n_features)
        Y: array-like with shape (n_samples_Y, n_features)
        weights: list of length n_features with feature weights

    Returns:
    --------
        distances: array with shape (n_samples_X, n_samples_Y)
    """
    n, p = X.shape
    if Y.shape != (n, p):
        raise ValueError("X and Y dimensions do not match!")

    if weights is None:
        weights = [1] * p

    return


def _check_numeric_features(X):
    numeric_types = 'biuf'  # boolean, signed/unsigned int, float
    feat_is_numeric = {
        name: (dt.kind in numeric_types) for name, dt in X.dtypes.iteritems()
    }
    return feat_is_numeric


def _feature_similarity(row1, row2, feat_is_numeric_list=None, weights=None):
    feat_distances = np.zeros(row1.shape) - 1
    for i, this_feat_is_numeric in enumerate(feat_is_numeric_list):
        if this_feat_is_numeric:
            pass
        else:
            feat_distances[i] = row1.iloc[0, i] == row2.iloc[0, i]
        pass
    return
