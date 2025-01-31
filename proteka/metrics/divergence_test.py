"""
test proteka.metrics.divergence module
"""

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon as js
from proteka.metrics.divergence import (
    optimal_offset,
    mse,
    mse_log,
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
    h1 = np.array([x if x > threshold else threshold for x in h1])
    h2 = np.array([x if x > threshold else threshold for x in h2])
    return np.sum(h2 * np.log(h2 / h1))


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

target_histogram1 = np.array([0.1, 0.2, 0.65, 0.05])
reference_histogram1 = np.array([0.2, 0.3, 0.4, 0.1])

target_histogram2 = np.array([0.2, 0.0, 0.4, 0.4])
reference_histogram2 = np.array([0.0, 0.5, 0.3, 0.2])

reference_kld1 = manual_kl_div(target_histogram1, reference_histogram1)
reference_jsd1 = manual_js_div(target_histogram1, reference_histogram1)

reference_kld2 = manual_kl_div(target_histogram2, reference_histogram2)
reference_jsd2 = manual_js_div(target_histogram2, reference_histogram2)

reference_mse1 = np.average((target_histogram1 - reference_histogram1) ** 2)
reference_offset_1 = np.average(
    np.log(reference_histogram1) - np.log(target_histogram1)
)
reference_mse_ldist1 = np.average(
    (
        np.log(target_histogram1)
        - np.log(reference_histogram1)
        + reference_offset_1
    )
    ** 2
)

ref_vector_hist = np.stack([reference_histogram1, reference_histogram2]).T
target_vector_hist = np.stack([target_histogram1, target_histogram2]).T

ref_vector_kl = np.array([reference_kld1, reference_kld2])
ref_vector_js = np.array([reference_jsd1, reference_jsd2])

# this is a test based on the explicit formulas for the kullback leibler divergence
# for a uniform and a gaussian distribution

n_samples = 1000000
a, b = 1, 2.5
c, d = 0, 6
reference_data_3 = (b - a) * np.random.rand(n_samples) + a
target_data_3 = (d - c) * np.random.rand(n_samples) + c
nbins = 201
bins = np.linspace(c, d, nbins)

reference_histogram3 = np.histogram(reference_data_3, bins=bins)[0] / n_samples
target_histogram3 = np.histogram(target_data_3, bins=bins)[0] / n_samples
# see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Uniform_distributions
reference_kld3 = np.log((d - c) / (b - a))


sigma1 = 1.5
mu1 = 3
sigma2 = 2
mu2 = 0.5
reference_data_4 = np.random.normal(mu1, sigma1, n_samples)
target_data_4 = np.random.normal(mu2, sigma2, n_samples)

a = np.minimum(reference_data_4.min(), target_data_4.min())
b = np.maximum(reference_data_4.max(), target_data_4.max())

bins = np.linspace(a, b, 501)

reference_histogram4 = np.histogram(reference_data_4, bins=bins)[0] / n_samples
target_histogram4 = np.histogram(target_data_4, bins=bins)[0] / n_samples

# see https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
reference_kld4 = (
    np.log(sigma2 / sigma1)
    + (sigma1**2 + (mu2 - mu1) ** 2) / (2 * sigma2**2)
    - 0.5
)


def test_kl_divergence():
    """
    Test basic functionality
    """
    assert np.isclose(
        kl_divergence(
            target_histogram1,
            reference_histogram1,
        ),
        reference_kld1,
    )


def test_kl_divergence_3():
    """
    Test basic functionality
    """
    print(reference_kld3)
    proteka_kl = kl_divergence(
        target_histogram3, reference_histogram3, threshold=1e-8
    )
    assert np.isclose(proteka_kl, reference_kld3, rtol=5e-2)


def test_kl_divergence_4():
    """
    Test basic functionality
    """
    proteka_kl = kl_divergence(
        target_histogram4,
        reference_histogram4,
        threshold=1e-8,
    )
    assert np.isclose(proteka_kl, reference_kld4, rtol=1e-1)


def test_kl_divergence_raise():
    """
    Test basic functionality
    """

    with pytest.raises(RuntimeError):
        proteka_kl = kl_divergence(
            target_histogram2,
            reference_histogram2,
        )


def test_kl_divergence2d():
    """
    Test basic functionality
    """
    assert np.isclose(
        kl_divergence(
            target_histogram1.reshape(2, 2),
            reference_histogram1.reshape(2, 2),
            threshold=1e-8,
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


def test_optimal_offset():
    proteka_offset = optimal_offset(
        np.log(target_histogram1), np.log(reference_histogram1)
    )
    assert np.isclose(proteka_offset, reference_offset_1)


def test_mse_ldist():
    mse_l = mse_log(target_histogram1, reference_histogram1)
    assert np.isclose(mse_l, reference_mse_ldist1)


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
            target_vector_hist,
            ref_vector_hist,
        ),
    )


def test_vector_js_divergence():
    """Test basic functionality"""
    np.testing.assert_allclose(
        ref_vector_js,
        vector_js_divergence(ref_vector_hist, target_vector_hist),
        rtol=1e-6,
    )
