import numpy as np
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

"""
Module contains basic feature-agnostic metrics estimators
"""

__all__ = [
    "mse",
    "kl_divergence",
    "js_divergence",
    "vector_kl_divergence",
    "vector_js_divergence",
    "vector_mse",
]

def clean_distribution(
        array: np.ndarray, 
        value:float = 1e-12, 
        threshold: float = 1e-8
):
    if not threshold > value:
        raise ValueError(
            f"value {value} should be larger than threshold {threshold}"
        )
    new_array = [x if x > threshold else 1e-10 for x in array]
    return new_array


def kl_divergence(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> float:
    r"""
    Compute Kullback-Leibler divergence between specified data sets.

    .. math :: D_{KL} = \sum_i p_i log (\frac{p_i}{q_i}), p_i, q_i \ne 0

    Here p corresponds to the reference (True) distribution, q corresponds
    to the target distribution. If  p_i or q_i <= `threshold`, bin i is excluded
    from the summation. The algorithm is the same as one used in CPPTRAJ
    (https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml)

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
    """
    # Find the valid bins
    valid_bins = np.logical_and(
        target_normalized > threshold, reference_normalized > threshold
    )
    terms = rel_entr(
        reference_normalized[valid_bins],
        target_normalized[valid_bins],
    )
    return terms.sum()
    """
    target_normalized = clean_distribution(target_normalized)
    reference_normalized = clean_distribution(reference_normalized)
    kl = rel_entr(reference_normalized,target_normalized)
    return kl.sum()


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
    target_normalized = target  / np.sum(target)
    reference_normalized = reference  / np.sum(reference)

    M = 0.5 * (target_normalized + reference_normalized)
    jsd = 0.5 * (
        kl_divergence(M, target_normalized, threshold=threshold)
        + kl_divergence(M, reference_normalized, threshold=threshold)
    )
    return jsd


def mse(target: np.ndarray, reference: np.ndarray) -> float:
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

def mse_dist(target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8) -> float:
    r"""
     Compute Mean Squared Error between the log  specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

     Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.


     Parameters
     -----------

     target, reference : np.ndarray
        Target and reference probability distributions (histograms).
        Should have the same shape

    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold

     Returns : float
     ------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)
    
    val = mse(reference_normalized,target_normalized)
    
    return val

def mse_log(target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8) -> float:
    r"""
     Compute Mean Squared Error between the log  specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

     Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.


     Parameters
     -----------

     target, reference : np.ndarray
        Target and reference probability distributions (histograms).
        Should have the same shape

    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold

     Returns : float
     ------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)
    """
    # Find the valid bins
    valid_bins = np.logical_and(
    target_normalized > threshold, reference_normalized > threshold
    )
    """
    target_normalized = clean_distribution(target_normalized)
    reference_normalized = clean_distribution(reference_normalized)
    val = mse(np.log(reference_normalized),np.log(target_normalized))
    return val

def wasserstein(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> float:
    """
    Compute Wasserstein distance between specified data sets.

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

    target_normalized = target  / np.sum(target)
    reference_normalized = reference  / np.sum(reference)
    target_normalized = clean_distribution(target_normalized)
    reference_normalized = clean_distribution(reference_normalized)
    was = wasserstein_distance(reference_normalized, target_normalized)
    return was




def vector_kl_divergence(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> np.ndarray:
    """
    Compute independent KL divergences between specified vector data sets.

    Parameters
    -----------

    target, reference : np.typing.ArrayLike
                Target and reference probability distributions (histograms).
                Should have the same shape

    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold

    Returns : np.ndarray
    ------
        Vector KL divergence of the target from the reference
    """
    assert target.shape == reference.shape
    assert len(target.shape) > 1
    num_feat = target.shape[-1]
    kld = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        kld[i] = kl_divergence(
            target[:, i], reference[:, i], threshold=threshold
        )
    return kld


def vector_js_divergence(
    target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8
) -> np.ndarray:
    """
    Compute independent JS divergences between specified vector data sets.

    Parameters
    -----------

    target, reference : np.typing.ArrayLike
                Target and reference probability distributions (histograms).
                Should have the same shape

    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold

    Returns : np.ndarray
    ------
        Vector JS divergence of the target from the reference
    """
    assert target.shape == reference.shape
    assert len(target.shape) > 1
    num_feat = target.shape[-1]
    jsd = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        jsd[i] = js_divergence(
            target[:, i], reference[:, i], threshold=threshold
        )
    return jsd


def vector_mse(target: np.ndarray, reference: np.ndarray) -> float:
    r"""
     Compute Vector Mean Squared Error between specified data sets.

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

    target_normalized = target / np.sum(target, axis=0)
    reference_normalized = reference / np.sum(reference, axis=0)

    return np.average((target - reference) ** 2, axis=0)


def vector_mse_dist(target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8) -> float:
    r"""
     Compute Mean Squared Error between the log  specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

     Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.


     Parameters
     -----------

     target, reference : np.ndarray
        Target and reference probability distributions (histograms).
        Should have the same shape

    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold

     Returns : float
     ------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target, axis=0)
    reference_normalized = reference / np.sum(reference, axis=0)
    
    val = vector_mse(reference_normalized,target_normalized)
    
    return val

def vector_mse_log(target: np.ndarray, reference: np.ndarray, threshold: float = 1e-8) -> float:
    r"""
     Compute Mean Squared Error between the log  specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

     Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.


     Parameters
     -----------

     target, reference : np.ndarray
        Target and reference probability distributions (histograms).
        Should have the same shape

    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold

     Returns : float
     ------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target, axis=0)
    reference_normalized = reference / np.sum(reference, axis=0)
    """
    # Find the valid bins
    valid_bins = np.logical_and(
    target_normalized > threshold, reference_normalized > threshold
    )
    """
    num_feat = target.shape[-1]
    val = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        target = clean_distribution(target_normalized[:,i])
        reference = clean_distribution(reference_normalized[:,i])
        val[i] = mse(np.log(reference),np.log(target))
    return val
