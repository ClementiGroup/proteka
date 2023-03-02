"""
test proteka.metrics.divergence module
"""
import numpy as np
import pytest

from proteka.metrics.divergence import kl_divergence


target_histogram = np.array([0.1, 0.2, 0.7, 0.0])
reference_histogram = np.array([0.0, 0.3, 0.6, 0.1])
scaling = 5

reference_kld = 0e0
for i in [1, 2]:
    reference_kld +=  target_histogram[i]*np.log(target_histogram[i]/reference_histogram[i])

def test_kl_divergence():
    assert kl_divergence(target_histogram, reference_histogram) == reference_kld

def test_kl_divergence_normalized():
     assert kl_divergence(target_histogram*scaling, reference_histogram*scaling, normalized=False) == reference_kld

def test_kl_divergence_shapes_match():
    with pytest.raises(AssertionError):
        kl_divergence(target_histogram[1::], reference_histogram)