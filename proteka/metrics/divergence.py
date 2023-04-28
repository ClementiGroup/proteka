import numpy as np
from scipy.special import rel_entr

"""
Module contains basic feature-agnostic metrics estimators
"""

__all__ = ["kl_divergence", "js_divergence", "mse"]


def kl_divergence(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> float:
    r"""
     Compute Kullback-Leibler divergence between specified data sets.

     .. math :: D_{KL} = \sum_i p_i log (\frac{p_i}{q_i}), p_i, q_i \ne 0

     Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.
     If  p_i or q_i <= `threshold`, bin i is excluded from the summation.
     The algorithm is the same as one used in CPPTRAJ (https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml)


     Parameters
     -----------

     target, reference : np.ndarray
                             Target and reference probability distributions (histograms).
                             Should have the same shape

     threshold : float, 1e-8
         Bin is not included in the summation if its value is less than the threshold

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
        reference_normalized[valid_bins],
        target_normalized[valid_bins],
    )
    return terms.sum()


def js_divergence(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> float:
    """
    Compute Jensen_Shannon divergence between specified data sets.

    Parameters
    -----------

    target, reference : np.typing.ArrayLike
                            Target and reference probability distributions (histograms).
                            Should have the same shape

    threshold : float, 1e-8
         Bin is not included in the summation if its value is less than the threshold

    Returns : float
    ------
    JS divergence of the target from the reference
    """
    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    M = 0.5 * (target_normalized + reference_normalized)
    jsd = 0.5 * (
        kl_divergence(M, target_normalized, threshold=threshold)
        + kl_divergence(M, reference_normalized, threshold=threshold)
    )
    return jsd

def mse(
    target: np.ndarray, reference: np.ndarray
) -> float:
    r"""
     Compute Mean Squared Error between specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (p_i - q_i)^2

     Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.


     Parameters
     -----------

     target, reference : np.ndarray
        Target and reference probability distributions (histograms).
        Should have the same shape

     Returns : float
     ------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"
    
    return np.average((target - reference) ** 2)