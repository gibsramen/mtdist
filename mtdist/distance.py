import numpy as np
import pandas as pd


def gower_distances(X, weights=None):
    """Calculate Gower distance (Gower 1971) between observations

    From daisy documentation:

     ------------------------------------------------------------------
    | d_ij = \\frac{\\sum_{k=1}^p w_k \\delta_{ij}^{(k)} d_{ij}^{(k)}} |
    |             {\\sum_{k=1}^p w_k \\delta_{ij}^{(k)}}               |
     ------------------------------------------------------------------

    w_k: weight of k-th variable of p
    \\delta_{ij}^{(i)}: 0 when k-th variable is missing in at least one
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
    feat_ranges = _check_feature_ranges(X, feat_is_numeric_dict)

    x = _feature_similarity(
        X.iloc[0, :],
        X.iloc[1, :],
        feat_is_numeric_dict=feat_is_numeric_dict,
        feature_ranges=feat_ranges,
        weights=weights,
    )

    return x


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


def _feature_similarity(
    row1,
    row2,
    feat_is_numeric_dict=None,
    feature_ranges=None,
    weights=None,
):
    num = 0
    denom = 0
    for i, item in enumerate(feat_is_numeric_dict.items()):
        feat = item[0]
        is_numeric = item[1]

        if pd.isna(row1[feat]) or pd.isna(row2[feat]):
            delta = 0
        else:
            delta = 1

        if is_numeric:
            dist = np.abs(row1[feat] - row2[feat]) / feature_ranges[feat]
        else:
            dist = int(row1[feat] != row2[feat])

        dist = dist * weights[i] * delta
        num += dist
        denom = denom + weights[i] * delta
    return num/denom
