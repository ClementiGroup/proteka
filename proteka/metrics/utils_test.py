"""
test proteka.metrics.utils module,
mainly histogram routines.

#TO DO:
test behavior of weights
test behavior of non-integer bins argument
"""

import numpy as np
from proteka.metrics import (
    histogram_vector_features,
    histogram_features,
    histogram_features2d,
)

reference_series = np.array([[0, 0, 0, 1, 1, 5, 5], [0, 0, 0, 1, 1, 5, 5]]).T
target_series = np.array(
    [[-1, 0, 1, 1, 1, 5, 5, 5], [0, 0, 1, 1, 1, 5, 15, 10]]
).T

# Target histograms, assuming 5 bins derived based on target series
reference_histogram_true = np.array([6, 4, 0, 0, 4])
target_histogram_true_closed_ends = np.array([3, 6, 0, 0, 4])
target_histogram_true_open_ends = np.array([4, 6, 0, 0, 6])
reference_histogram_2d_true = np.array(
    [
        [3, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2],
    ]
)

reference_histogram_2d_true = np.array(
    [
        [3, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2],
    ]
)

target_histogram_2d_true_closed_ends = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]
)

target_histogram_2d_true_open_ends = np.array(
    [
        [2, 0, 0, 0, 0],
        [0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3],
    ]
)

reference_vector_histogram_true = np.array([[3, 2, 0, 0, 2], [3, 2, 0, 0, 2]]).T
target_vector_histogram_true_closed_ends = np.array(
    [[1, 3, 0, 0, 3], [2, 3, 0, 0, 1]]
).T


def test_histogram_features():
    target_histogram, reference_histogram = histogram_features(
        target_series, reference_series, bins=5
    )
    np.testing.assert_array_equal(
        target_histogram, target_histogram_true_closed_ends
    )
    np.testing.assert_array_equal(reference_histogram, reference_histogram_true)
    return


def test_histogram_features_open_ends():
    target_histogram, reference_histogram = histogram_features(
        target_series, reference_series, bins=5, open_edges=True
    )
    np.testing.assert_array_equal(
        target_histogram, target_histogram_true_open_ends
    )
    np.testing.assert_array_equal(reference_histogram, reference_histogram_true)
    return


def test_histogram_features2d():
    target_histogram_2d, reference_histogram_2d = histogram_features2d(
        target_series, reference_series, bins=5, open_edges=False
    )
    np.testing.assert_array_equal(
        reference_histogram_2d, reference_histogram_2d_true
    )
    np.testing.assert_array_equal(
        target_histogram_2d, target_histogram_2d_true_closed_ends
    )


def test_histogram_features2d_open_ends():
    target_histogram_2d, reference_histogram_2d = histogram_features2d(
        target_series, reference_series, bins=5, open_edges=True
    )
    np.testing.assert_array_equal(
        reference_histogram_2d, reference_histogram_2d_true
    )
    np.testing.assert_array_equal(
        target_histogram_2d, target_histogram_2d_true_open_ends
    )


def test_histogram_vector_features():
    (
        target_histogram_vector,
        reference_histogram_vector,
    ) = histogram_vector_features(
        target_series, reference_series, bins=5, open_edges=False
    )
    np.testing.assert_array_equal(
        target_histogram_vector, target_vector_histogram_true_closed_ends
    )
    np.testing.assert_array_equal(
        reference_histogram_vector, reference_vector_histogram_true
    )
    return
