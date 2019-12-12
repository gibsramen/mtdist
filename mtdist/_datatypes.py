def check_numeric_features(X):
    """Check which features of data are numeric"""
    numeric_types = 'biuf'  # boolean, signed/unsigned int, float
    feat_is_numeric = {
        name: (dt.kind in numeric_types) for name, dt in X.dtypes.iteritems()
    }
    return feat_is_numeric
