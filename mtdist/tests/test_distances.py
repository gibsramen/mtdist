#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest

from .. import distance


class TestMixedDataHandling():

    def test_check_numeric_features(self):
        df = pd.DataFrame(np.random.randint(1, 10, (5, 5)))
        df.columns = list("abcde")
        df["f"] = ["hi"] * 5
        d = distance._check_numeric_features(df)
        target_dict = {x: True for x in df.columns}
        target_dict["f"] = False
        assert d == target_dict

    def test_check_feature_ranges(self):
        mat = np.array([
            [1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8],
            [3, 6, 9, 12, 15],
            [2, 2, 2, 2, 2],
        ])
        df = pd.DataFrame(mat, columns=list("abcde"))
        df["f"] = ["hi"] * 4

        target_ranges = [3, 4, 7, 10, 13]
        target_ranges = {x: y for x, y in zip("abcde", target_ranges)}

        feat_is_numeric_dict = distance._check_numeric_features(df)
        r = distance._check_feature_ranges(df, feat_is_numeric_dict)
        assert r == target_ranges


class TestGower():

    def test_gower_1(self):
        """Test that Gower distance can handle categorical data

         ----------------------
        | 1  2  3  4  5  "hi"  |
        | 0  2  4  6  8  "bye" |
        | 3  6  9  12 15 "hi"  |
        | 2  2  2  2  2  "bye" |
         ----------------------

        Dist between row1 and row2:
            (1/3 + 0/4 + 1/7 + 2/10 + 3/13 + 1)/(6)
            = 0.31782661782661786
        """
        target_dist = 0.31782661782661786
        mat = np.array([
            [1, 2, 3, 4, 5],
            [0, 2, 4, 6, 8],
            [3, 6, 9, 12, 15],
            [2, 2, 2, 2, 2],
        ])
        df = pd.DataFrame(mat)
        df.columns = list("abcde")
        df["f"] = ["hi", "bye"] * 2
        gower_dist = distance.gower_distances(df)
        diff = target_dist - gower_dist
        assert diff == pytest.approx(0)
