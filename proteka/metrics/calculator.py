"""Main entry point for calculating the metrics
"""
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import numpy as np
import mdtraj as md
from mdtraj.core.element import Element
from typing import Union, Dict, Optional, List, Tuple
from .featurizer import Featurizer, TICATransform
from ..dataset import Ensemble
from typing import Union, Dict
from .divergence import (
    mse,
    mse_dist,
    mse_log,
    fraction_smaller,
    wasserstein,
    kl_divergence,
    js_divergence,
    vector_kl_divergence,
    vector_js_divergence,
    vector_mse,
    vector_mse_log,
)
from .utils import (
    get_general_distances,
    histogram_features,
    histogram_vector_features,
    histogram_features2d,
)

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

    acceptors_or_donors = ["N", "O", "S"]

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
    def ca_clashes(ensemble: Ensemble) -> Dict[str, int]:
        """Compute total number of instances when there is a clash between CA atoms
        Clashes are defined as any 2 nonconsecutive CA atoms been closer than  0.4 nm
        """
        # Only consider distances between nonconsecutive CA atoms, hence the offset=1
        distances = Featurizer.get_feature(ensemble, "ca_distances", offset=1)
        clashes = np.where(distances < 0.4)[0]
        return {"CA-CA clashes": clashes.size}

    @staticmethod
    def general_clashes(
        ensemble: Ensemble,
        atom_name_pairs: List[Tuple[str, str]],
        thresholds: Optional[List[float]] = None,
        res_offset: int = 1,
        stride: Optional[int] = None,
        allowance: float = 0.07,
    ) -> Dict[str, int]:
        """ "Compute clashes between atoms of types `atom_name_1/2` according
        to user-supplied thresholds or according to the method of allowance-modified
        VDW radii overlap described here:

        https://www.cgl.usf.edu/chimera/docs/ContributedSoftware/findclash/findclash.html

        with VDW radii in nm taken from `mdtraj.core.element.Element`. If the pair
        is composed of atom species that can potentially form hydrogen bonds
        (e.g., ("N", "O", "S")), then an additional default allowance of 0.07 nm is permitted.

        Parameters
        ----------
        ensemble:
            `Ensemble` over which clashes should be detected
        atom_name_pairs:
            List of `str` tuples that denote the first atom type pairs according to
            the MDTraj selection language
        thresholds:
            List of clash thresholds for each type pair in atom_name_pairs. If `None`,
            the clash thresholds are calculated according to:

                thresh = r_vdw_i + r_vdw_j - allowance

            for atoms i,j.
        allowance:
            Additional distance tolerance for atoms involved in hydrogen bonding. Only
            used if thresholds is `None`. Set to 0.07 nm by default
        res_offset:
            `int` that determines the minimum residue separation for inclusion in distance
            calculations; two atoms that belong to residues i and j are included in the
            calculations if `|i-j| > res_offset`.
        stride:
            If specified, this stride is applied to the trajectory before the distance
            calculations

        Returns
        -------
        Dict[str, int]:
            Dictionary with keys `{name1}_{name2}_clashes` and values reporting
            the number of clashes found for those name pairs
        """
        if not isinstance(atom_name_pairs, list):
            raise ValueError(
                "atom_name_pairs must be a list of tuples of strings"
            )

        # populate default thresholds
        if thresholds == None:
            thresholds = []
            atoms = np.array(list(ensemble.top.atoms))
            for pair in atom_name_pairs:
                assert len(pair) == 2
                # Take elements from first occurrence in topology - selection language gaurantees that they
                # all should be the same (unless you have made a very nonstandard topology)
                idx1, idx2 = (
                    ensemble.top.select(f"name {pair[0]}")[0],
                    ensemble.top.select(f"name {pair[1]}")[0],
                )
                vdw_r1, vdw_r2 = (
                    atoms[idx1].element.radius,
                    atoms[idx2].element.radius,
                )
                threshold = vdw_r1 + vdw_r2

                # Handle hydrogen bonding allowances
                # between donors and acceptors with
                # different names
                if all(
                    [
                        p in StructuralIntegrityMetrics.acceptors_or_donors
                        for p in pair
                    ]
                ):
                    threshold = threshold - allowance
                thresholds.append(threshold)

        if not isinstance(thresholds, list):
            raise ValueError("thresholds must be a list of floats")
        if len(atom_name_pairs) != len(thresholds):
            raise RuntimeError(
                f"atom_name_pairs and thresholds are {len(atom_name_pairs)} and {len(thresholds)} long, respectively, but they should be the same length"
            )

        clash_dictionary = {}

        for atom_names, threshold in zip(atom_name_pairs, thresholds):
            atom_name_1, atom_name_2 = atom_names[0], atom_names[1]
            distances = get_general_distances(
                ensemble, atom_names, res_offset, stride
            )
            clashes = np.where(distances < threshold)[0]
            clash_dictionary[
                f"{atom_name_1}-{atom_name_2} clashes"
            ] = clashes.size
        return clash_dictionary

    @staticmethod
    def ca_pseudobonds(ensemble: Ensemble) -> Dict[str, float]:
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

    metric_types = set(
        [
            "kl_div",
            "js_div",
            "mse",
            "mse_dist",
            "mse_ldist",
            "fraction_smaller"
            # "wasserstein"
        ]
    )
    scalar_metrics = {
        "kl_div": kl_divergence,
        "js_div": js_divergence,
        "mse": mse,
        "mse_dist": mse_dist,
        "mse_ldist": mse_log,
        "fraction_smaller": fraction_smaller
        # "wasserstein": wasserstein,
    }
    vector_metrics = {
        "kl_div": vector_kl_divergence,
        "js_div": vector_js_divergence,
        "mse": vector_mse,
        "mse_dist": vector_mse,
        "mse_ldist": vector_mse_log,
    }
    metrics_2d = {
        "kl_div": kl_divergence,
        "js_div": js_divergence,
        "mse": mse,
        "mse_dist": mse_dist,
        "mse_ldist": mse_log,
        "fraction_smaller": fraction_smaller,
    }

    excluded_quantities = set(
        [
            "top",
            "coords",
            "time",
            "forces",
            "cell_angles",
            "cell_lengths",
            "trjs",
        ]
    )

    scalar_features = set(
        ["rg", "ca_distances", "rmsd", "end2end_distance", "tic1", "tic2"]
    )
    vector_features = set(["local_contact_number", "dssp"])
    features_2d = set(["tic1_tic2"])

    def __init__(
        self,
        metrics_params={
            "tica_kl_div": {"lagtime": 10},
            "tica_js_div": {"lagtime": 10},
        },
    ):
        super().__init__()
        self.metrics_params = metrics_params
        self.metrics_dict = {}

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

    @staticmethod
    def _feature_contraction(
        all_target_feats: List[str],
        all_ref_feats: List[str],
        features: List[str],
    ) -> Iterable[str]:
        """Contracts two feature sets to their intesection"""
        all_target_feats, all_ref_feats, features = (
            set(all_target_feats),
            set(all_ref_feats),
            set(features),
        )
        warnings.warn(
            f"target Ensemble has {all_target_feats} as registered quantities, but"
            f" reference ensemble has {all_ref_feats} as registered quantities,"
            f" and {features} are the requested features. The smallest"
            f" mutual subset of features will have their metrics computed"
        )
        features = list(
            features.intersection(all_target_feats).intersection(all_ref_feats)
        )
        return features

    @staticmethod
    def _check_features(
        target: Ensemble,
        reference: Ensemble,
        features: Union[Iterable[str], str] = "all",
    ) -> List[str]:
        """Checks feature inputs for `compute()`

        Parameters
        ----------
        metrics:
            specified metric(s) for computation.

        Returns
        -------
        metrics:
            List of string(s) specifying the vetted features(s)
            for which metrics should be computed downstream
        """

        # Feature checks
        all_target_feats = target.list_quantities()
        all_ref_feats = reference.list_quantities()

        if isinstance(features, str) and features != "all":
            # Case where single feature is specified
            if (
                features not in all_target_feats
                and features not in all_ref_feats
            ):
                raise ValueError(
                    f"feature {feature} is not registered in target nor reference ensemble."
                )
            else:
                features = [features]
        elif features == "all" or isinstance(features, Iterable):
            # Cases where "all" is chosen and target and ref have different feature sets
            if features == "all":
                # if "all" take all reference features
                features = list(all_target_feats)
            if not all(
                [
                    (f in all_target_feats and f in all_ref_feats)
                    for f in features
                ]
            ):
                features = EnsembleQualityMetrics._feature_contraction(
                    all_target_feats, all_ref_feats, features
                )

        # Unavailable feature filter
        skip_idx = []
        for idx, feature in enumerate(features):
            if feature in EnsembleQualityMetrics.excluded_quantities:
                warnings.warn(
                    f"feature {feature} is not available for metric comparison, and it will be skipped."
                )
                skip_idx.append(idx)
        features = [f for idx, f in enumerate(features) if idx not in skip_idx]
        return features

    @staticmethod
    def _check_metrics(
        metrics: Union[Iterable[str], str] = "all",
    ) -> Iterable[str]:
        """Checks to make sure requested metrics are valid

        Parameters
        ----------
        metrics:
            specified metric(s) for computation.

        Returns
        -------
        metrics:
            List of string(s) specifying the vetted metric(s)
            that should be computed downstream
        """
        valid_metrics = EnsembleQualityMetrics.metric_types
        if metrics == "all":
            metrics = valid_metrics
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif isinstance(metrics, Iterable) and all(
            [isinstance(met, str) for met in metrics]
        ):
            pass
        else:
            raise ValueError(
                f"metrics {metrics} not in '['all', str, Iterable[str]']'"
            )

        if not all([m in valid_metrics for m in metrics]):
            raise ValueError(
                f"Requested metrics '{metrics}' not compatible with valid metrics '{valid_metrics}'"
            )
        return metrics

    def compute(
        self,
        target: Ensemble,
        reference: Ensemble,
        features: Union[Iterable[str], str] = "all",
        metrics: Union[Iterable[str], str] = "all",
    ):
        """
        Compute the metrics that compare the target ensemble to the reference over
        the specified features. Comute metrics are stored in the `EnsembleQualityMetrics.results`
        attribute.

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
        features = EnsembleQualityMetrics._check_features(
            target, reference, features
        )
        metrics = EnsembleQualityMetrics._check_metrics(metrics)
        for feature in features:
            for metric in metrics:
                param_key = f"{feature}_{metric}"
                params = self.metrics_params.get(param_key)
                if params is None:
                    params = {}
                result = EnsembleQualityMetrics.compute_metric(
                    target, reference, feature, metric, **params
                )
                self.results.update(result)
        return

    @staticmethod
    def compute_metric(
        target: Ensemble,
        reference: Ensemble,
        feature: str,
        metric: str = "kl_div",
        bins: Union[int, np.ndarray] = 100,
        **kwargs,
    ) -> Dict[str, float]:
        """Computes metric for desired feature between two ensembles.

        Parameters
        ----------
        target:
            Target ensemble
        reference:
            Refernce ensemble
        feature:
            string specifying the feature for which the desired metric should be computed
            over from the target to the reference ensemble. Valid features can be
            scalars (eg, `EnsembleQualityMetrics.scalar_features`) or vector features
            (eg, `EnsembleQualityMetrics.vector_features`)
        metric:
            string specifying the metric to compute for the desire feature between the
            target and reference ensembles. Valid metrics are contained in
            `EnsembleQualityMetrics.metrics`
        bins:
            In the case that the metric is calculated over probability distributions,
            this integer number of bins or `np.ndarray` of bins is used to compute
            histograms for both the target and reference ensembles

        Returns
        -------
        result:
            dict of the form {"{feature}, {metric}" : metric_result} for
            the specified feature and metric between the target and
            reference ensembles.
        """

        if metric not in EnsembleQualityMetrics.metric_types:
            raise ValueError(f"Metric '{metric}' not defined.")

        if metric == "fraction_smaller" and feature != "rmsd":
            return {f"{feature}, {metric}": None}

        if feature == "tica":
            target_feat, reference_feat = get_tica_features(
                target, reference, **kwargs
            )
        else:
            target_feat = Featurizer.get_feature(target, feature)
            reference_feat = Featurizer.get_feature(reference, feature)

        reference_weights = (
            reference.weights if hasattr(reference, "weights") else None
        )
        target_weights = target.weights if hasattr(target, "weights") else None

        if feature in EnsembleQualityMetrics.scalar_features:
            metric_computer = EnsembleQualityMetrics.scalar_metrics[metric]
            hist_target, hist_ref = histogram_features(
                target_feat,
                reference_feat,
                bins=bins,
                target_weights=target_weights,
                reference_weights=reference_weights,
            )
        elif feature in EnsembleQualityMetrics.vector_features:
            metric_computer = EnsembleQualityMetrics.vector_metrics[metric]
            hist_target, hist_ref = histogram_vector_features(
                target_feat,
                reference_feat,
                bins=bins,
                target_weights=target_weights,
                reference_weights=reference_weights,
            )
        elif feature in EnsembleQualityMetrics.features_2d:
            metric_computer = EnsembleQualityMetrics.metrics_2d[metric]
            hist_target, hist_ref = histogram_features2d(
                target_feat,
                reference_feat,
                bins=bins,
                target_weights=target_weights,
                reference_weights=reference_weights,
            )
            hist_target = hist_target.flatten()
            hist_ref = hist_ref.flatten()
        else:
            raise ValueError(
                f"feature {feature} not registered in vector or scalar features"
            )
        metric_args = {
            k: v
            for k, v in kwargs.items()
            if not k in ["bins", "reference_weights"]
        }
        if metric in ["mse", "fraction_smaller"]:
            # mse should be computed over the exact values, not over the prob distribution
            result = metric_computer(target_feat, reference_feat, **metric_args)
        else:
            result = metric_computer(hist_target, hist_ref, **metric_args)
        return {f"{feature}, {metric}": result}
