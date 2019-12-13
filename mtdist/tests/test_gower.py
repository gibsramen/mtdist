#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest

from .. import gower


@pytest.fixture(scope="function")
def dataset1():
    """Testing dataset 1

         -------------------------
        |    a  b  c   d  e     f |
        | r1 1  2  3  4  5  "hi"  |
        | r2 0  2  4  6  8  "bye" |
        | r3 3  6  9  12 15 "hi"  |
        | r4 2  2  2  2  2  "bye" |
         -------------------------
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

    return df


def test_gower_1(dataset1):
    """Test that Gower distance can handle categorical data

    Distances from daisy:

         --------------------------------------------
        |           r1        r2        r3        r4 |
        | r1 0.0000000 0.3178266 0.6821734 0.3178266 |
        | r2 0.3178266 0.0000000 0.8087912 0.3023199 |
        | r3 0.6821734 0.8087912 0.0000000 0.8888889 |
        | r4 0.3178266 0.3023199 0.8888889 0.0000000 |
         --------------------------------------------
    """
    gower_dist = gower.gower_distances(dataset1)
    target_dist = np.array([
        [0.0000000, 0.3178266, 0.6821734, 0.3178266],
        [0.3178266, 0.0000000, 0.8087912, 0.3023199],
        [0.6821734, 0.8087912, 0.0000000, 0.8888889],
        [0.3178266, 0.3023199, 0.8888889, 0.0000000],
    ])
    np.testing.assert_allclose(gower_dist, target_dist, rtol=1e-05)


def test_weights(dataset1):
    """Test that weights are integrated properly

    With weights = [1, 1, 1, 2, 2, 1]

    Distances from daisy:

         --------------------------------------------
        |           r1        r2        r3        r4 |
        | r1 0.0000000 0.2922161 0.7077839 0.2922161 |
        | r2 0.2922161 0.0000000 0.7489011 0.3344322 |
        | r3 0.7077839 0.7489011 0.0000000 0.9166667 |
        | r4 0.2922161 0.3344322 0.9166667 0.0000000 |
         --------------------------------------------
    """
    weights = [1, 1, 1, 2, 2, 1]
    weights = {feat: w for feat, w in zip(dataset1.columns, weights)}

    gower_dist = gower.gower_distances(
        dataset1,
        weights=weights,
    )
    target_dist = np.array([
        [0.0000000, 0.2922161, 0.7077839, 0.2922161],
        [0.2922161, 0.0000000, 0.7489011, 0.3344322],
        [0.7077839, 0.7489011, 0.0000000, 0.9166667],
        [0.2922161, 0.3344322, 0.9166667, 0.0000000],
    ])
    np.testing.assert_allclose(gower_dist, target_dist, rtol=1e-05)


def test_missing_values(dataset1):
    """Test that missing values are handled properly

    Distances from daisy:

         --------------------------------------------
        |           r1        r2        r3        r4 |
        | r1 0.0000000 0.2413919 0.6586081 0.3678266 |
        | r2 0.2413919 0.0000000 0.8131868 0.4827839 |
        | r3 0.6586081 0.8131868 0.0000000 0.8666667 |
        | r4 0.3678266 0.4827839 0.8666667 0.0000000 |
         --------------------------------------------
    """
    dataset1.iloc[2, 3] = np.nan
    dataset1.iloc[1, 5] = np.nan

    gower_dist = gower.gower_distances(dataset1)
    target_dist = np.array([
        [0.0000000, 0.2413919, 0.6586081, 0.3678266],
        [0.2413919, 0.0000000, 0.8131868, 0.4827839],
        [0.6586081, 0.8131868, 0.0000000, 0.8666667],
        [0.3678266, 0.4827839, 0.8666667, 0.0000000],
    ])
    np.testing.assert_allclose(gower_dist, target_dist, rtol=1e-05)


def test_symm_boolean_values(dataset1):
    """Test that missing values are handled properly

    r4 = [True, False, False, False]

    Distances from daisy:

         --------------------------------------------
        |           r1        r2        r3        r4 |
        | r1 0.0000000 0.3178266 0.8488400 0.3178266 |
        | r2 0.3178266 0.0000000 0.6421245 0.3023199 |
        | r3 0.8488400 0.6421245 0.0000000 0.7222222 |
        | r4 0.3178266 0.3023199 0.7222222 0.0000000 |
         --------------------------------------------
    """
    dataset1["f"] = [True, False, False, False]

    gower_dist = gower.gower_distances(dataset1)
    target_dist = np.array([
        [0.0000000, 0.3178266, 0.8488400, 0.3178266],
        [0.3178266, 0.0000000, 0.6421245, 0.3023199],
        [0.8488400, 0.6421245, 0.0000000, 0.7222222],
        [0.3178266, 0.3023199, 0.7222222, 0.0000000],
    ])
    np.testing.assert_allclose(gower_dist, target_dist, rtol=1e-05)
