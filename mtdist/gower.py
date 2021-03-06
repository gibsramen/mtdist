import numpy as np
import pandas as pd

from ._datatypes import get_feature_types


def gower_distances(
    X: pd.DataFrame,
    weights: dict = None,
    feature_types: dict = None,
) -> np.ndarray:
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
        weights: dict of format {feature: weight}
        feature_types: dict of format {feature: weight}

    Returns:
    --------
        distances: array with shape (n_samples_X, n_samples_X)
    """
    n, p = X.shape

    if weights is None:
        weights = {feat: 1 for feat in X.columns}

    feature_types_dict = get_feature_types(X)
    if feature_types is not None:
        feature_types_dict.update(feature_types)

    feat_ranges = _check_feature_ranges(X, feature_types_dict)

    D = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = _feature_similarity(
                X.iloc[i, :],
                X.iloc[j, :],
                feature_types_dict=feature_types_dict,
                feature_ranges=feat_ranges,
                weights=weights,
            )
            D[i, j] = d
            D[j, i] = d
    return D


def _check_feature_ranges(X, feature_types_dict):
    """Gower distance requires range of numeric features"""
    feat_ranges = {
        feat: X[feat].max() - X[feat].min() for feat in
        X.columns if feature_types_dict[feat] == "numeric"
    }
    return feat_ranges


def _feature_similarity(
    row1,
    row2,
    feature_types_dict=None,
    feature_ranges=None,
    weights=None,
):
    """Compute Gower distance between two observations"""
    num = 0
    denom = 0
    for feat, feat_type in feature_types_dict.items():

        val1 = row1[feat]
        val2 = row2[feat]
        weight = weights.get(feat, 1)

        # delta is 0 if one of the two feature values is missing/NaN
        # if feature is asymmetric binary, delta is 0 if both observations
        #  are False, 1 otherwise
        delta = not int(pd.isna(val1) or pd.isna(val2))
        if feat_type == "asymmbin":
            delta = val1 or val2

        if delta == 0:
            continue

        if feat_type == "numeric":
            dist = np.abs(val1 - val2) / feature_ranges[feat]
        else:
            dist = int(val1 != val2)

        dist = dist * weight

        num += dist
        denom = denom + weight
    return num/denom
