import numpy as np
from scipy.special import rel_entr

"""
Module contains basic feature-agnostic metrics estimators
"""


def kl_divergence(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> float:
    r"""
     Compute Kullback-Leibler divergence between specified data sets.

     .. math :: D_{KL} = \sum_i p_i log (\frac{p_i}{q_i}), p_i, q_i \ne 0

     If  p_i or q_i <= `threshold`, bin i is excluded from the summation.
     The algorithm is the same as one used in CPPTRAJ (https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml)


     Parameters
     -----------

     target, reference : np.ndarray
                             Target and reference probability distributions (histograms).
                             Should have the same shape

     normalized : bool, True
         If true, the input distributions are assumed to be normalized. Otherwise, the histogram
         will be normalized such that all elements sum up to 1. Note that this histogram normalization
         is different from the default numpy histogram normalization.

     Returns : float
     ------
    KL divergence of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    # Find the valid bins
    valid_bins = np.logical_and(
        target_normalized > threshold, reference_normalized > threshold
    )
    terms = rel_entr(
        target_normalized[valid_bins], reference_normalized[valid_bins]
    )
    return terms.sum()


def js_divergence(target: np.ndarray, reference: np.ndarray, normalized: bool = True) -> float:
    """
     Compute Jensen_Shannon divergence between specified data sets.
     
     Parameters
     -----------

     target, reference : np.typing.ArrayLike
                             Target and reference probability distributions (histograms).
                             Should have the same shape

     normalized : bool, True
         If true, the input distributions are assumed to be normalized. Otherwise, the histogram
         will be normalized such that all elements sum up to 1. Note that this histogram normalization
         is different from the default numpy histogram normalization.

     Returns : float
     ------
     JS divergence of the target from the reference
    """
    target_norm = np.sum(target)
    reference_norm = np.sum(reference)
    if normalized:
        assert (
            np.isclose(target_norm, 1.0e0)
        ), f"Norm of the target dataset {target_norm}, expected 1.0e0"
        assert (
            np.isclose(reference_norm, 1.0e0)
        ), f"Norm of the reference dataset {reference_norm}, expected 1.0e0"
        target_normalized = target
        reference_normalized = reference
    if not normalized:
        target_normalized = target / target_norm
        reference_normalized = reference / reference_norm


    M = 0.5*(target + reference)
    jsd = 0.5*(kl_divergence(target, M) + kl_divergence(reference, M))
    return jsd

  
