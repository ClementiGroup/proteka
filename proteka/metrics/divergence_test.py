"""
test proteka.metrics.divergence module
"""
import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon as js
from proteka.metrics.divergence import (
    mse,
    kl_divergence,
    js_divergence,
    vector_kl_divergence,
    vector_js_divergence,
)


def manual_kl_div(h1, h2):
    """h1=target, h2=ref"""
    threshold = 1e-8
    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)
    valid_bins = np.logical_and(h1 > threshold, h2 > threshold)
    return np.sum(h2[valid_bins] * np.log(h2[valid_bins] / h1[valid_bins]))


def manual_js_div(h1, h2):
    """h1=target, h2=ref"""
    h1 = h1 / np.sum(h1)
    h2 = h2 / np.sum(h2)
    mean = (h1 + h2) / 2.0
    print(h1, h2, mean)
    jsd = (manual_kl_div(mean, h1) + manual_kl_div(mean, h2)) / 2.0
    print(jsd)
    return jsd


scaling = 5

target_histogram1 = np.array([0.1, 0.2, 0.7, 0.0])
reference_histogram1 = np.array([0.0, 0.3, 0.6, 0.1])

target_histogram2 = np.array([0.2, 0.2, 0.2, 0.4])
reference_histogram2 = np.array([0.0, 0.5, 0.3, 0.2])

reference_kld1 = manual_kl_div(target_histogram1, reference_histogram1)
reference_jsd1 = manual_js_div(target_histogram1, reference_histogram1)

reference_kld2 = manual_kl_div(target_histogram2, reference_histogram2)
reference_jsd2 = manual_js_div(target_histogram2, reference_histogram2)

reference_mse1 = np.average((target_histogram1 - reference_histogram1) ** 2)

ref_vector_hist = np.stack([reference_histogram1, reference_histogram2]).T
target_vector_hist = np.stack([target_histogram1, target_histogram2]).T

ref_vector_kl = np.array([reference_kld1, reference_kld2])
ref_vector_js = np.array([reference_jsd1, reference_jsd2])


def test_kl_divergence():
    """
    Test basic functionality
    """
    assert np.isclose(
        kl_divergence(
            target_histogram1, reference_histogram1, intersect_only=True
        ),
        reference_kld1,
    )


def test_kl_divergence2d():
    """
    Test basic functionality
    """
    assert np.isclose(
        kl_divergence(
            target_histogram1.reshape(2, 2),
            reference_histogram1.reshape(2, 2),
            intersect_only=True,
        ),
        reference_kld1,
    )


def test_kl_divergence_normalized():
    """
    Test normalization feature
    """
    assert np.isclose(
        kl_divergence(
            target_histogram1 * scaling,
            reference_histogram1 * scaling,
            intersect_only=True,
            threshold=1e-8,
        ),
        reference_kld1,
    )


def test_kl_divergence_shapes_match():
    """
    Test behavior when input shape mismatch
    """
    with pytest.raises(AssertionError):
        kl_divergence(target_histogram1[1::], reference_histogram1)


def test_mse():
    """
    Test basic functionality
    """
    assert np.isclose(
        mse(target_histogram1, reference_histogram1), reference_mse1
    )


def test_mse2d():
    """
    Test basic functionality
    """
    assert np.isclose(
        mse(
            target_histogram1.reshape(2, 2), reference_histogram1.reshape(2, 2)
        ),
        reference_mse1,
    )


def test_mse_normalized():
    """
    Test normalization feature
    """
    assert np.isclose(
        mse(
            target_histogram1 * scaling,
            reference_histogram1 * scaling,
        ),
        scaling**2 * reference_mse1,
    )


def test_mse_shapes_match():
    """
    Test behavior when input shape mismatch
    """
    with pytest.raises(AssertionError):
        mse(target_histogram1[1::], reference_histogram1)


def test_js_divergence():
    """
    Test basic functionality
    """
    assert np.isclose(
        js_divergence(
            target_histogram1,
            reference_histogram1,
            threshold=1e-8,
        ),
        reference_jsd1,
    )


def test_vector_kl_divergence():
    """Test basic functionality"""
    np.testing.assert_allclose(
        ref_vector_kl,
        vector_kl_divergence(
            target_vector_hist, ref_vector_hist, intersect_only=True
        ),
    )


def test_vector_js_divergence():
    """Test basic functionality"""
    np.testing.assert_allclose(
        ref_vector_js, vector_js_divergence(ref_vector_hist, target_vector_hist)
    )
