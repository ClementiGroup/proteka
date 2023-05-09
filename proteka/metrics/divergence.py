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
    value: float = 1e-8,
    threshold: float = 1e-8,
    intersect_only=False,
):
    """Cleans input distributions

    Parameters
    ----------
    array:
        input normalized distribution
    threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold
        and intersect_only is `True`
    replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
    intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

    Returns
    -------
    valid_bins:
        If `intersect_only` is `True`, the indices of the bins that are above the `threshold`
    new_array:
        If `intersect_only` is `False`, a new distribution where values below the threshold are
        replaced by `value`
    """

    if intersect_only == True:
        valid_bins = np.argwhere(array > threshold).flatten()
        return valid_bins
    else:
        new_array = np.array([x if x > threshold else threshold for x in array])
        return new_array


def kl_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    replace_value: float = 1e-8,
    intersect_only: bool = True,
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
        and intersect_only is `True`
    replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
    intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

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

    if intersect_only == True:
        target_valid_bins = clean_distribution(
            target_normalized, threshold=threshold, intersect_only=True
        )
        reference_valid_bins = clean_distribution(
            reference_normalized, threshold=threshold, intersect_only=True
        )
        valid_bins = np.array(
            list(
                set(target_valid_bins.tolist()).intersection(
                    set(reference_valid_bins.tolist())
                )
            )
        )
        kl = rel_entr(
            reference_normalized[valid_bins],
            target_normalized[valid_bins],
        )
    else:
        target_normalized = clean_distribution(
            target_normalized, threshold=threshold, value=replace_value
        )
        reference_normalized = clean_distribution(
            reference_normalized, threshold=threshold, value=replace_value
        )
        kl = rel_entr(reference_normalized, target_normalized)
    return kl.sum()


def js_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    replace_value: float = 1e-8,
    intersect_only: bool = True,
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
        and intersect_only is `True`
    replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
    intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation) for KL
        divergence subcomputations

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
            replace_value=replace_value,
            intersect_only=intersect_only,
        )
        + kl_divergence(
            M,
            reference_normalized,
            threshold=threshold,
            replace_value=replace_value,
            intersect_only=intersect_only,
        )
    )
    return jsd


def mse(
    target: np.ndarray,
    reference: np.ndarray,
) -> float:
    r"""
     Compute Mean Squared Error between specified data sets.

     Parameters
     ----------

     target, reference : np.ndarray
        Target and reference data arrays.
        Should have the same shape

     Returns : float
     -------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    return np.average((target - reference) ** 2)


def mse_dist(
    target: np.ndarray,
    reference: np.ndarray,
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

     threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold
        and intersect_only is `True`
     replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
     intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

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

    val = mse(reference_normalized, target_normalized)

    return val


def mse_log(
    target: np.ndarray,
    reference: np.ndarray,
    replace_value: float = 1e-8,
    intersect_only: bool = False,
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
        Bin is not included in the summation if its value is less than the threshold
        and intersect_only is `True`
     replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
     intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

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

    if intersect_only == True:
        target_valid_bins = clean_distribution(
            target_normalized, threshold=threshold, intersect_only=True
        )
        reference_valid_bins = clean_distribution(
            reference_normalized, threshold=threshold, intersect_only=True
        )
        valid_bins = np.array(
            list(
                set(target_valid_bins.tolist()).intersection(
                    set(reference_valid_bins.tolist())
                )
            )
        )
        val = mse(
            np.log(reference_normalized[valid_bins]),
            np.log(target_normalized[valid_bins]),
        )
    else:
        target_normalized = clean_distribution(
            target_normalized, threshold=threshold, value=replace_value
        )
        reference_normalized = clean_distribution(
            reference_normalized, threshold=threshold, value=replace_value
        )
        val = mse(np.log(reference_normalized), np.log(target_normalized))

    return val


def wasserstein(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    replace_value: float = 1e-8,
    intersect_only: bool = False,
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
        and intersect_only is `True`
     replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
     intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

    Returns : float
    ------
        JS divergence of the target from the reference
    """

    target_normalized = target / np.sum(target)
    reference_normalized = reference / np.sum(reference)

    if intersect_only == True:
        target_valid_bins = clean_distribution(
            target_normalized, threshold=threshold, intersect_only=True
        )
        reference_valid_bins = clean_distribution(
            reference_normalized, threshold=threshold, intersect_only=True
        )
        valid_bins = np.array(
            list(
                set(target_valid_bins.tolist()).intersection(
                    set(reference_valid_bins.tolist())
                )
            )
        )
        was = wasserstein_distance(
            reference_normalized[valid_bins], target_normalized[valid_bins]
        )
    else:
        target_normalized = clean_distribution(
            target_normalized, threshold=threshold, value=replace_value
        )
        reference_normalized = clean_distribution(
            reference_normalized, threshold=threshold, value=replace_value
        )
        was = wasserstein_distance(reference_normalized, target_normalized)
    return was


def vector_kl_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    replace_value: float = 1e-8,
    intersect_only: bool = True,
) -> np.ndarray:
    """
    Compute independent KL divergences between specified vector data sets.

    Parameters
    ----------

    target, reference : np.typing.ArrayLike
                Target and reference probability distributions (histograms).
                Should have the same shape

     threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold
        and intersect_only is `True`
     replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
     intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

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
            replace_value=replace_value,
            intersect_only=intersect_only,
        )
    return kld


def vector_js_divergence(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    replace_value: float = 1e-8,
) -> np.ndarray:
    """
    Compute independent JS divergences between specified vector data sets.

    Parameters
    ----------

    target, reference : np.typing.ArrayLike
                Target and reference probability distributions (histograms).
                Should have the same shape

     threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold
        and intersect_only is `True`
     replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
     intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

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
            replace_value=replace_value,
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
        and intersect_only is `True`

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

    target_normalized = target / np.sum(target, axis=0)
    reference_normalized = reference / np.sum(reference, axis=0)

    val = vector_mse(reference_normalized, target_normalized)

    return val


def vector_mse_log(
    target: np.ndarray,
    reference: np.ndarray,
    threshold: float = 1e-8,
    replace_value: float = 1e-8,
    intersect_only: bool = False,
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

     threshold : float, 1e-8
        Bin is not included in the summation if its value is less than the threshold
        and intersect_only is `True`
     replace_value:
        if the bin has a normalized count lower than the `threshold`, and `intersect_only`
        is `False`, then the bin gets replaced with this value instead
     intersect_only:
        if `True`, distributions will only be compared over their consistent support overlaps
        (eg, only the mutual set of populated bins will be included in the computation)

     Returns : float
     ------
    Mean Squared Error of the target from the reference

    """
    assert (
        target.shape == reference.shape
    ), f"Dimension mismatch: target: {target.shape} reference: {reference.shape}"

    target_normalized = target / np.sum(target, axis=0)
    reference_normalized = reference / np.sum(reference, axis=0)

    num_feat = target.shape[-1]
    val = np.zeros(num_feat)
    # slow implementation I know
    for i in range(num_feat):
        val[i] = mse(np.log(reference_normalized), np.log(target_normalized))

    return val
