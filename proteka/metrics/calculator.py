"""Main entry point for calculating the metrics
"""
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import numpy as np
from typing import Union, Dict

from .featurizer import Featurizer
from ..dataset import Ensemble
from .divergence import kl_divergence, js_divergence, mse
from .utils import histogram_features, histogram_features2d, get_tica_features

__all__ = ["StructuralIntegrityMetrics", "EnsembleQualityMetrics"]


class IMetrics(metaclass=ABCMeta):
    """Abstract class defining interface for metrics calculators"""

    def __init__(self):
        self.results = {}

    @abstractmethod
    def __call__(self, ensemble: Ensemble):
        pass

    def report(self):
        return self.results

    @abstractmethod
    def compute(self, Ensemble, metrics: Iterable):
        """Method to compute the metrics"""
        pass


class StructuralIntegrityMetrics(IMetrics):
    """Class takes a dataset and checks if for chemical integrity"""

    def __init__(self):
        super().__init__()
        self.metrics_dict = {
            "ca_clashes": self.ca_clashes,
            "ca_pseudobonds": self.ca_pseudobonds,
        }

    def __call__(
        self,
        ensemble: Ensemble,
        metrics: Iterable = ["ca_clashes", "ca_pseudobonds"],
    ):
        self.compute(ensemble, metrics)
        return self.report()

    def compute(
        self,
        ensemble: Ensemble,
        metrics: Iterable = ["ca_clashes", "ca_pseudobonds"],
    ):
        """Method to compute the metrics"""
        for metric in metrics:
            self.results.update(self.metrics_dict[metric](ensemble))
        return

    @staticmethod
    def ca_clashes(ensemble: Ensemble) -> dict:
        """Compute total number of instances when there is a clash between CA atoms
        Clashes are defined as any 2 nonconsecutive CA atoms been closer than  0.4 nm
        """
        # Only consider distances between nonconsecutive CA atoms, hence the offset=1
        distances = Featurizer.get_feature(ensemble, "ca_distances", offset=1)
        clashes = np.where(distances < 0.4)[0]
        return {"N clashes": clashes.size}

    @staticmethod
    def ca_pseudobonds(ensemble: Ensemble) -> dict:
        """Computes maximum and rms z-score for d_ca_ca bonds over ensemble.
        Z-score is defined as (d_ca_ca - mean(d_ca_ca)) / std(d_ca_ca)
        d_ca_ca and std(d_ca_ca) are parametrized based on analysis of the proteka
        and pdb databases. Beware, that this metric will give high deviations for
        cis-proline peptide bonds
        """

        # Get the mean and std of d_ca_ca
        mean = 0.38  # nm
        std = 0.05  # nm
        d_ca_ca = Featurizer.get_feature(ensemble, "ca_bonds")
        z = (d_ca_ca - mean) / std
        return {
            "max z-score": np.max(z),
            "rms z-score": np.sqrt(np.mean(z**2)),
        }


class EnsembleQualityMetrics(IMetrics):
    """Metrics to compare a target ensemble to the reference ensemble"""

    metric_compute_map = {
        "kl_div": kl_divergence,
        "js_div": js_divergence,
        "mse": mse,
    }

    excluded_quantities = set(["top", "coords", "time", "forces"])

    def __init__(
        self,
        metrics_params={
            "tica_kl_div": {"lagtime": 10},
            "tica_js_div": {"lagtime": 10},
        },
    ):
        super().__init__()

        self.metrics_params = metrics_params

    def __call__(
        self,
        target: Ensemble,
        reference: Ensemble,
        features: Union[Iterable[str], str],
        metrics: Union[Iterable[str], str] = "all",
    ):
        """Calls the `compute` method and reports the results
        as a dictionary of metric_name: metric_value pairs
        See compute() for more details.
        """
        self.compute(target, reference, features, metrics)
        return self.report()

    def compute(
        self,
        target: Ensemble,
        reference: Ensemble,
        features: Union[Iterable[str], str] = "all",
        metrics: Union[Iterable[str], str] = "all",
    ):
        """
        Compute the metrics that compare the target ensemble to the reference

        Parameters:
        -----------
        target: Ensemble
            The target ensemble
        reference: Ensemble
            The reference ensemble, against which the target ensemble is compared
        features: Iterable of strings or str
            List of features from which the chosen metrics will be computed. If "all", every
            feature in the reference Ensemble will be grabbed
        metrics: Iterable of strings or str
            The metrics to compute. If "all" is passed, all available metrics will be computed
        """
        # Feature checks
        all_target_feats = set(target.list_quantities())
        all_ref_feats = set(reference.list_quantities())

        if features == "all":
            features = list(all_target_feats)
            if not all(
                [
                    (f in all_target_feats and f in all_ref_feats)
                    for f in features
                ]
            ):
                warnings.warn(
                    "target Ensemble has {} as registered quantities, but reference ensemble has {} as registered quantities, and {} are the requested features. The smallest mutual subset of features will have their metrics computed".format(
                        all_target_feats, all_ref_feats, features
                    )
                )
                features = list(
                    features.intersection(all_target_feats).intersection(
                        all_ref_feats
                    )
                )
        elif isinstance(features, Iterable):
            if not all(
                [
                    (f in all_target_feats and f in all_ref_feats)
                    for f in features
                ]
            ):
                warnings.warn(
                    "target Ensemble has {} as registered quantities, but reference ensemble has {} as registered quantities, and {} are the requested features. The smallest mutual subset of features will have their metrics computed".format(
                        all_target_feats, all_ref_feats, set(features)
                    )
                )
                features = list(
                    set(features)
                    .intersection(all_target_feats)
                    .intersection(all_ref_feats)
                )
            else:
                features = reference.list_quantities
        elif isinstance(features, str):
            if feature not in all_target_feats and feature not in all_ref_feats:
                raise ValueError(
                    "feature {} is not registered in target nor reference ensemble.".format(
                        feature
                    )
                )
            else:
                features = [features]

        for feat in features:
            if feat in EnsembleQualityMetrics.excluded_quantities:
                warnings.warn(
                    "feature {} is not available for metric comparison, and it will be skipped.".format(
                        feat
                    )
                )
                features.pop(feat)

        # Metric checks
        if metrics == "all":
            metrics = self.metric_compute_map.keys()
        elif isinstance(metrics, Iterable):
            assert all(
                [
                    m in EnsembleQualityMetrics.metric_compute_map.keys()
                    for m in metrics
                ]
            )
        elif isinstance(metrics, str):
            assert metrics in EnsembleQualityMetrics.metric_compute_map.keys()
            metrics = [metrics]

        for feature in features:
            for metric in metrics:
                param_key = f"{feature}_{metric}"
                params = self.metrics_params.get(param_key)
                if params is None:
                    params = {}
                result = EnsembleQualityMetrics.compute_metric(
                    target, reference, feature, metric, **params
                )
        return

    @staticmethod
    def compute_metric(
        target: Ensemble,
        reference: Ensemble,
        feature: str,
        metric: str = "kl_div",
        bins: Union[int, np.ndarray] = 100,
        **kwargs,
    ) -> Dict:
        """Computes the specified metric between the target and reference ensembles for the feature given as input.
        If the input feature is 'tica' the computation is done consistently with what was done before

        Parameters:
        -----------
        target: Ensemble
            target ensemble
        reference: Ensemble
            reference ensemble to compare to
        feature: str
            name of the feature to compute the metric on.
            can be any of the 1D-features from the Featurizer, names have to be consistent, or it can be 'tica'
        metric: str
            specifies which metric to calculate
        bins: np.ndarray
            Number of bins to create histrograms with. If np.ndarray, the bins are scattered according to these edges

        Returns:
        --------
        Dict:
            with the key corresponding to 'feature, KL divergence'
            and the value being the KL divergence between the reference and target ensembles on the selected feature.
        """

        if metric not in [
            "kl_div",
            "js_div",
            "mse",
        ]:
            raise ValueError("Metric '{}' not defined.".format(metric))

        if feature == "tica":
            target_feat, reference_feat = get_tica_features(
                target, reference, **kwargs
            )

        else:
            target_feat = Featurizer.get_feature(target, feature)
            reference_feat = Featurizer.get_feature(reference, feature)

        assert len(target_feat.shape) == len(
            reference_feat.shape
        ), f"mismatch in features dimensions : target has dimension {len(target_feat.shape)} and reference has dimension {len(reference_feat.shape)}"

        if np.squeeze(target_feat).ndim == np.squeeze(reference_feat).ndim == 1:
            hist_target, hist_ref = histogram_features(
                target_feat, reference_feat, bins=bins
            )
        elif (
            np.squeeze(target_feat).ndim == np.squeeze(reference_feat).ndim == 2
        ):
            hist_target, hist_ref = histogram_features2d(
                target_feat, reference_feat, bins=bins
            )
        else:
            raise NotImplementedError(
                f"No histogram computation implemented for feature dimension {np.squeeze(target_feat).ndim}"
            )

        result = EnsembleQualityMetrics.metric_compute_map[metric](
            hist_target, hist_ref
        )
        return {f"{feature}, {metric}": result}
