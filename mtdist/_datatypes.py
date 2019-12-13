def get_feature_types(X):
    feat_type_dict = {name: dt.kind for name, dt in X.dtypes.iteritems()}
    type_map = {
        "i": "numeric",
        "u": "numeric",
        "f": "numeric",
        "b": "categorical",
        "O": "categorical",
    }
    return {name: type_map[_type] for name, _type in feat_type_dict.items()}
