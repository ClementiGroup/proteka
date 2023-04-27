"""
test proteka.metrics.divergence module
"""
import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon as js

from proteka.metrics.divergence import kl_divergence, js_divergence


target_histogram = np.array([0.1, 0.2, 0.7, 0.0])
reference_histogram = np.array([0.0, 0.3, 0.6, 0.1])
scaling = 5

target_histogram2 = np.array([0.2, 0.2, 0.2, 0.4])
reference_histogram2 = np.array([0.0, 0.5, 0.3, 0.2])
scaling = 5

reference_kld = 0e0
for i in [1, 2]:
    reference_kld += reference_histogram[i] * np.log(
        reference_histogram[i] / target_histogram[i]
    )

referenc_jsd = 0e0
for i in [1, 2]:
    reference_kld += reference_histogram[i] * np.log(
        reference_histogram[i] / target_histogram[i]
    )


def test_kl_divergence():
    """
    Test basic functionality
    """
    assert np.isclose(
        kl_divergence(target_histogram, reference_histogram), reference_kld
    )


def test_kl_divergence2d():
    """
    Test basic functionality
    """
    assert np.isclose(
        kl_divergence(
            target_histogram.reshape(2, 2), reference_histogram.reshape(2, 2)
        ),
        reference_kld,
    )


def test_kl_divergence_normalized():
    """
    Test normalization feature
    """
    assert np.isclose(
        kl_divergence(
            target_histogram * scaling,
            reference_histogram * scaling,
            threshold=1e-12,
        ),
        reference_kld,
    )


def test_kl_divergence_shapes_match():
    """
    Test behavior when input shape mismatch
    """
    with pytest.raises(AssertionError):
        kl_divergence(target_histogram[1::], reference_histogram)


def test_js_divergence():
    """
    Test basic functionality
    """
    # scipy.spatial.distance.jensenshannon doesn't support
    # simultaneuos calculation along several axis
    fn_output = js_divergence(target_histogram, reference_histogram)
    reference = js(target_histogram, reference_histogram) ** 2
    assert np.isclose(fn_output, reference)
