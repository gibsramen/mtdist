import numpy as np
import pandas as pd

from ._datatypes import check_numeric_features


def gower_distances(X: pd.DataFrame, weights: dict = None) -> np.ndarray:
    """Calculate Gower distance (Gower 1971) between observations

    From daisy documentation:

     ------------------------------------------------------------------
    | d_ij = \\frac{\\sum_{k=1}^p w_k \\delta_{ij}^{(k)} d_{ij}^{(k)}} |
    |             {\\sum_{k=1}^p w_k \\delta_{ij}^{(k)}}               |
     ------------------------------------------------------------------

    w_k: weight of k-th variable of p
    \\delta_{ij}^{(k)}: 0 when k-th variable is missing in at least one
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
        weights = {feat: 1 for feat in X.columns}

    feat_is_numeric_dict = check_numeric_features(X)
    feat_ranges = _check_feature_ranges(X, feat_is_numeric_dict)

    D = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = _feature_similarity(
                X.iloc[i, :],
                X.iloc[j, :],
                feat_is_numeric_dict=feat_is_numeric_dict,
                feature_ranges=feat_ranges,
                weights=weights,
            )
            D[i, j] = d
            D[j, i] = d
    return D


def _check_feature_ranges(X, feat_is_numeric_dict):
    """Gower distance requires range of numeric features"""
    feat_ranges = {
        feat: X[feat].max() - X[feat].min() for feat in
        X.columns if feat_is_numeric_dict[feat]
    }
    return feat_ranges


def _feature_similarity(
    row1,
    row2,
    feat_is_numeric_dict=None,
    feature_ranges=None,
    weights=None,
):
    """Compute Gower distance between two observations"""
    num = 0
    denom = 0
    # can probably be cleaned up with nested dictionaries
    # maybe even a namedtuple
    for i, item in enumerate(feat_is_numeric_dict.items()):
        feat = item[0]
        is_numeric = item[1]

        val1 = row1[feat]
        val2 = row2[feat]
        weight = weights[feat]

        # delta is 0 if one of the two feature values is missing/NaN
        delta = not int(pd.isna(val1) or pd.isna(val2))
        if delta == 0:
            continue

        if is_numeric:
            dist = np.abs(val1 - val2) / feature_ranges[feat]
        else:
            dist = int(val1 != val2)

        dist = dist * weight

        num += dist
        denom = denom + weight
    return num/denom
