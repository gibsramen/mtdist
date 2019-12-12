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

         -------------------------
        |    a  b  c   d  e     f |
        | r1 1  2  3  4  5  "hi"  |
        | r2 0  2  4  6  8  "bye" |
        | r3 3  6  9  12 15 "hi"  |
        | r4 2  2  2  2  2  "bye" |
         -------------------------

        Distances from daisy:

         --------------------------------------------
        |           r1        r2        r3        r4 |
        | r1 0.0000000 0.3178266 0.6821734 0.3178266 |
        | r2 0.3178266 0.0000000 0.8087912 0.3023199 |
        | r3 0.6821734 0.8087912 0.0000000 0.8888889 |
        | r4 0.3178266 0.3023199 0.8888889 0.0000000 |
         --------------------------------------------
        """
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
        target_dist = np.array([
            [0.0000000, 0.3178266, 0.6821734, 0.3178266],
            [0.3178266, 0.0000000, 0.8087912, 0.3023199],
            [0.6821734, 0.8087912, 0.0000000, 0.8888889],
            [0.3178266, 0.3023199, 0.8888889, 0.0000000],
        ])
        np.testing.assert_allclose(gower_dist, target_dist)
