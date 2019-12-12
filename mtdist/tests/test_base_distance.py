#!/usr/bin/env python

import numpy as np
import pandas as pd

from .. import _datatypes


class TestMixedDataHandling():

    def test_check_numeric_features(self):
        df = pd.DataFrame(np.random.randint(1, 10, (5, 5)))
        df.columns = list("abcde")
        df["f"] = ["hi"] * 5
        d = _datatypes.check_numeric_features(df)
        target_dict = {x: True for x in df.columns}
        target_dict["f"] = False
        assert d == target_dict
