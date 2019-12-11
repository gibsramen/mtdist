import numbers

import numpy as np
import pandas as pd


def gower_distances(X, weights=None):
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
        weights: list of length n_features with feature weights

    Returns:
    --------
        distances: array with shape (n_samples_X, n_samples_X)
    """
    n, p = X.shape

    if weights is None:
        weights = [1] * p

    feat_is_numeric_dict = _check_numeric_features(X)
    feat_ranges = _check_feature_ranges(X)

    return


def _check_numeric_features(X):
    numeric_types = 'biuf'  # boolean, signed/unsigned int, float
    feat_is_numeric = {
        name: (dt.kind in numeric_types) for name, dt in X.dtypes.iteritems()
    }
    return feat_is_numeric


def _check_feature_ranges(X, feat_is_numeric_dict):
    feat_ranges = {
        feat: X[feat].max() - X[feat].min() for feat in
        X.columns if feat_is_numeric_dict[feat]
    }
    return feat_ranges


def _feature_similarity(row1, row2, feat_is_numeric_dict=None, weights=None):
    feat_distances = list()
    for feat, is_numeric in feat_is_numeric_dict.items():
        if is_numeric:
            pass
        else:
            feat_distances.append(row1[feat] == row2[feat])
            pass
        pass
    return
