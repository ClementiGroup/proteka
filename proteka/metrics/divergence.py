from typing import Optional
import numpy as np
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

"""
Module contains basic feature-agnostic metrics estimators
"""

__all__ = [
    "kl_divergence",
    "js_divergence",
    "mse",
    "mse_dist",
    "mse_log",
    "fraction_smaller",
    "wasserstein",
    "vector_kl_divergence",
    "vector_js_divergence",
    "vector_mse",
    "vector_mse_dist",
    "vector_mse_log",
    "vector_wasserstein",
]


def clean_distribution(
    array: np.ndarray,
    threshold: float = 1e-8,
):
    """Cleans input distributions

    Parameters
    ----------
    array:
        input normalized distribution
    threshold : float, 1e-8
        if the bin value is lower than this threshold, its value
        is replaced with this threshold.

    Returns
    -------
    new_array:
        A new distribution where values below the threshold are
        replaced by said threshold value. This array is also renormalized
        after threshold adjustment.
    """

    new_array = np.where(array > threshold, array, threshold)
    # renormalize
    new_array = new_array / np.sum(new_array)
    return new_array


def kl_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: Optional[float] = None,
    **kwagrs,
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
        Bin value is replaced by this threshold if it is lower than this threshold

    Returns : float
    ------
        KL divergence of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    # reshape everything into a flattened array
    target_normalized = np.squeeze(target_normalized)
    reference_normalized = np.squeeze(reference_normalized)

    target_normalized = target_normalized.flatten()
    reference_normalized = reference_normalized.flatten()

    if threshold is not None:
        target_normalized = clean_distribution(
            target_normalized,
            threshold=threshold,
        )
        reference_normalized = clean_distribution(
            reference_normalized, threshold=threshold
        )
    if threshold is None:
        if any(target_normalized == 0) or any(reference_normalized == 0):
            raise RuntimeError(
                "At least one reference or target bin contains zero counts, and the KL "
                "divergence is undefined. If you wish to override this behavior, please "
                "specify a (small) threshold with which empty bins may be filled."
            )
    kl = rel_entr(reference_normalized, target_normalized)
    return kl.sum()


def js_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: Optional[float] = 1e-8,
) -> float:
    """
    Compute Jensen_Shannon divergence between specified data sets.

    Parameters
    -----------

    target, reference : np.typing.ArrayLike
                Target and reference probability distributions (histograms).
                Should have the same shape
    threshold : float, 1e-8
        Bin value is replaced by this threshold if it is lower than this threshold

    Returns : float
    ------
        JS divergence of the target from the reference
    """
    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    M = 0.5 * (target_normalized + reference_normalized)
    jsd = 0.5 * (
        kl_divergence(
            M,
            target_normalized,
            threshold=threshold,
        )
        + kl_divergence(
            M,
            reference_normalized,
            threshold=threshold,
        )
    )
    return jsd


def mse(target: np.ndarray, reference: np.ndarray, offset: float = 0) -> float:
    r"""
    Compute Mean Squared Error between specified data sets.

    Parameters
    ----------

    target, reference : np.ndarray
       Target and reference data arrays.
       Should have the same shape

    offset : float
       offset to add to the reference array. 0 by default

    Returns : float
    -------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    if np.abs(offset) < 1e-12:
        return np.average((target - reference) ** 2)
    else:
        return np.average(((target - reference) + offset) ** 2)


def optimal_offset(target: np.ndarray, reference: np.ndarray) -> float:
    r"""
    Compute the value of lambda that minimizes the residual

    .. math  \sum_{i=1}^N (p_i -  q_i + \lambda)^2

    This uses the analytical solution given by

    .. math \lambda = -\frac{1}{N} \sum_{i=1}^N (p_i -  q_i)

    Parameters
    ----------

    target, reference : np.ndarray
       Target and reference probability distributions (histograms).
       Should have the same shape

    Returns : floats
    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    lam = np.mean(reference - target)
    return lam


def mse_dist(
    target: np.ndarray, reference: np.ndarray, use_optimal_offset: bool = False
) -> float:
    r"""
    Compute Mean Squared Error between the log  specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

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

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    # reshape everything into a flattened array
    target_normalized = np.squeeze(target_normalized)
    reference_normalized = np.squeeze(reference_normalized)

    target_normalized = target_normalized.flatten()
    reference_normalized = reference_normalized.flatten()

    if use_optimal_offset:
        offset = optimal_offset(target_normalized, reference_normalized)
    else:
        offset = 0

    val = mse(target_normalized, reference_normalized, offset=offset)

    return val


def mse_log(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    use_optimal_offset: bool = True,
) -> float:
    r"""
    Compute Mean Squared Error between the log  specified data sets.

     .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

    Here p corresponds to the reference (True) distribution, q corresponds to the target distribution.


    Parameters
    ----------

    target, reference : np.ndarray
       Target and reference probability distributions (histograms).
       Should have the same shape
    threshold : float, 1e-8
        Bin value is replaced by this threshold if it is lower than this threshold

    Returns : float
    -------
    Mean Squared Error of the target from the reference
    """

    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    target_normalized = clean_distribution(target_normalized)
    reference_normalized = clean_distribution(reference_normalized)

    # reshape everything into a flattened array
    target_normalized = np.squeeze(target_normalized)
    reference_normalized = np.squeeze(reference_normalized)

    target_normalized = target_normalized.flatten()
    reference_normalized = reference_normalized.flatten()

    target_normalized = clean_distribution(
        target_normalized,
        threshold=threshold,
    )
    reference_normalized = clean_distribution(
        reference_normalized,
        threshold=threshold,
    )
    log_ref = np.log(reference_normalized)
    log_tar = np.log(target_normalized)

    if use_optimal_offset:
        offset = optimal_offset(log_tar, log_ref)
    else:
        offset = 0

    val = mse(log_tar, log_ref, offset)

    return val


def fraction_smaller(
    target: np.ndarray,
    threshold: float = 0.25,
) -> float:
    """Computes the fraction of a data array smaller than a specfied threshold.

    Parameters
    ----------
    target: np.ndarray
        Target data array.
    threshold : float
        Value to which we will compare this feature.

    Returns : float
    """

    smaller = np.mean(target < threshold)
    return smaller


def wasserstein(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
) -> float:
    """
    Compute Wasserstein distance between specified data sets.

    Parameters
    -----------
    target, reference : np.typing.ArrayLike
        Target and reference probability distributions (histograms).
        Should have the same shape
    threshold : float, 1e-8
        Bin value is replaced by this threshold if it is lower than this threshold

    Returns : float
    ------
        Wasserstein distance of the target
    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    target_normalized = clean_distribution(
        target_normalized, threshold=threshold
    )
    reference_normalized = clean_distribution(
        reference_normalized, threshold=threshold
    )
    n = len(reference_normalized)

    # see the comment in the upper part for understanding this weird definition
    val = wasserstein_distance(
        np.arange(n), np.arange(n), reference_normalized, target_normalized
    )

    return val


def vector_kl_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
) -> np.ndarray:
    """
    Compute independent KL divergences between specified vector data sets.

    Parameters
    ----------
    target, reference : np.typing.ArrayLike
       Target and reference probability distributions (histograms).
       Should have the same shape
    threshold : float, 1e-8
        Bin value is replaced by this threshold if it is lower than this threshold

    Returns : np.ndarray
    -------
        Vector KL divergence of the target from the reference
    """
    assert target.shape == reference.shape
    assert len(target.shape) > 1
    num_feat = target.shape[-1]
    kld = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        kld[i] = kl_divergence(
            target[:, i],
            reference[:, i],
            threshold=threshold,
        )
    return kld


def vector_js_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
) -> np.ndarray:
    """
    Compute independent JS divergences between specified vector data sets.

    Parameters
    ----------

    target, reference : np.typing.ArrayLike
       Target and reference probability distributions (histograms).
       Should have the same shape
    threshold : float, 1e-8
        Bin value is replaced by this threshold if it is lower than this threshold

    Returns : np.ndarray
    -------
       Vector JS divergence of the target from the reference
    """

    assert target.shape == reference.shape
    assert len(target.shape) > 1
    num_feat = target.shape[-1]
    jsd = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        jsd[i] = js_divergence(
            target[:, i],
            reference[:, i],
            threshold=threshold,
        )
    return jsd


def vector_mse(
    target: np.ndarray,
    reference: np.ndarray,
) -> float:
    r"""
    Compute Vector Mean Squared Error between specified data sets.

    Parameters
    -----------

    target, reference : np.ndarray
       Target and reference data arrays.
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

    return np.average((target - reference) ** 2, axis=0)


def vector_mse_dist(
    target: np.ndarray,
    reference: np.ndarray,
) -> float:
    r"""
    Compute vector Mean Squared Error between histograms of the specified data sets.

    .. math :: MSE = \frac{1}{N} \sum_{i=1}^N (log(p_i) - log(q_i))^2

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

    val = vector_mse(target_normalized, reference_normalized)

    return val


def vector_mse_log(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
) -> float:
    r"""
    Compute vector Mean Squared Error between the log of distributions of specified data sets.

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

    num_feat = target.shape[-1]
    val = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        val[i] = mse_log(
            target[:, i],
            reference[:, i],
            threshold=threshold,
        )

    return val


def vector_wasserstein(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
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
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    num_feat = target.shape[-1]
    val = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        val[i] = wasserstein(
            target[:, i],
            reference[:, i],
            threshold=threshold,
        )

    return val
