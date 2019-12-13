#!/usr/bin/env python

import numpy as np
import pandas as pd

from .. import _datatypes


class TestMixedDataHandling():

    def test_get_feature_types(self):
        df = pd.DataFrame(np.random.randint(1, 10, (5, 5)))
        df.columns = list("abcde")
        df["f"] = ["hi"] * 5
        d = _datatypes.get_feature_types(df)
        target_dict = {x: "numeric" for x in df.columns}
        target_dict["f"] = "categorical"
        assert d == target_dict

    def test_binary_features(self):
        df = pd.DataFrame(np.random.randint(1, 10, (5, 5)))
        df.columns = list("abcde")
        df["f"] = [True, False, True, True, False]
        d = _datatypes.get_feature_types(df)
        target_dict = {x: "numeric" for x in df.columns}
        target_dict["f"] = "categorical"
        assert d == target_dict

    def test_only_numeric_features(self):
        df = pd.DataFrame(np.random.randint(1, 10, (5, 5)))
        df.columns = list("abcde")
        d = _datatypes.get_feature_types(df)
        target_dict = {x: "numeric" for x in df.columns}
        assert d == target_dict
