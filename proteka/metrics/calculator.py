"""Main entry point for calculating the metrics
"""
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Mapping
import numpy as np
import mdtraj as md
from ruamel.yaml import YAML
from mdtraj.core.element import Element
from typing import Union, Dict, Optional, List, Tuple
from .featurizer import Featurizer, TICATransform
from ..dataset import Ensemble
from typing import Union, Dict
from .divergence import *
from .utils import (
    get_general_distances,
    histogram_features,
    histogram_vector_features,
    histogram_features2d,
)

__all__ = [
    "StructuralIntegrityMetrics",
    "StructuralQualityMetrics",
    "EnsembleQualityMetrics",
]

yaml = YAML(typ="safe")


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
        if thresholds is None:
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
                f"atom_name_pairs and thresholds are {len(atom_name_pairs)} and {len(thresholds)} long, "
                f"respectively, but they should be the same length"
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


class StructuralQualityMetrics(IMetrics):
    """Metrics that compare an ensemble to a single structure

        {
            "reference_structure": md.core.trajectory.Trajectory
            "features": {
                "rmsd": {
                    "feature_params": {"atom_selection": "name CA"},
                    "metric_params": {"fraction_smaller": 0.25},
                },
                ...
            }
        },

    Specifying computation and metric parameters for each feature/metric
    for comparisons between target and reference structure.
    """

    does_not_require_ref_struct = set(["rmsd, fraction_smaller"])
    scalar_features = set(["rmsd"])
    scalar_metrics = {
        "fraction_smaller": fraction_smaller,
    }

    def __init__(self, metrics: Dict):
        assert isinstance(
            metrics["reference_structure"], md.core.trajectory.Trajectory
        )
        assert metrics["reference_structure"].n_frames == 1
        self.metrics = metrics
        self.results = {}

    @classmethod
    def from_config(cls, config_file: str):
        """Instances an StructuralQualityMetrics
        from a config file. The config should have the example following structure:

            StructuralQualityMetrics:
              reference_structure: "my_structure.pdb"
              features:
                rmsd:
                  feature_params:
                    atom_selection: "name CA"
                  metric_params:
                    fraction_smaller:
                      threshold: 0.25
                ...

        Parameters
        ----------
        config_file:
            YAML file specifying feature and config options
        """

        config = yaml.load(open(config_file, "r"))
        sqm_config = config["StructuralQualityMetrics"]
        # load reference structure
        reference_structure_path = sqm_config["reference_structure"]
        sqm_config["reference_structure"] = md.load(reference_structure_path)

        return cls(sqm_config)

    def __call__(
        self,
        target: Ensemble,
    ):
        """calls the `compute` method and reports the results
        as a dictionary of metric_name: metric_value pairs
        see compute() for more details.
        """
        self.compute(target)
        return self.report()

    def compute(
        self,
        target: Ensemble,
    ):
        """
        compute the metrics that compare the target ensemble to the reference strucure over
        the specified features. compute metrics are stored in the `StructuralQualityMetrics.results`
        attribute.

        parameters:
        -----------
        target: Ensemble
            the target ensemble
        """
        for feature in self.metrics["features"].keys():
            # compute feature in target ensemble if needed
            if feature_params not in list(
                self.metrics["features"][feature].keys()
            ):
                feature_params = {}
            else:
                feature_params = self.metrics["features"][feature][
                    "feature_params"
                ]

            # additional args
            args = [self.metrics["reference_structure"]]
            Featurizer.get_feature(target, feature, *args, **feature_params)
            for metric in self.metrics["features"][feature][
                "metric_params"
            ].keys():
                params = self.metrics["features"][feature]["metric_params"][
                    metric
                ]
                if params is None:
                    params = {}
                result = StructuralQualityMetrics.compute_metric(
                    target,
                    self.metrics["reference_structure"],
                    feature,
                    metric,
                    **params,
                )
                self.results.update(result)
        return

    @staticmethod
    def compute_metric(
        target: Ensemble,
        reference_structure: md.Trajectory,
        feature: str,
        metric: str = "fraction_smaller",
        **kwargs,
    ) -> dict[str, float]:
        """computes metric for desired feature between two ensembles.

        parameters
        ----------
        target:
            target ensemble
        reference_structure:
            reference structure
        feature:
            string specifying the feature for which the desired metric should be computed
            over from the target to the reference structure.
        metric:
            string specifying the metric to compute for the desired feature between the
            target and reference structure.
        bins:
            in the case that the metric is calculated over probability distributions,
            this integer number of bins or `np.ndarray` of bins is used to compute
            histograms for both the target and reference ensembles

        returns
        -------
        result:
            dict of the form {"{feature}, {metric}" : metric_result} for
            the specified feature and metric between the target and
            reference ensembles.
        """

        target_feat = Featurizer.get_feature(target, feature)
        if feature in StructuralQualityMetrics.scalar_features:
            metric_computer = StructuralQualityMetrics.scalar_metrics[metric]

        if (
            f"{feature}, {metric}"
            in StructuralQualityMetrics.does_not_require_ref_struct
        ):
            result = metric_computer(target_feat, **kwargs)
        else:
            result = metric_computer(target_feat, reference_structure, **kwargs)
        return {f"{feature}, {metric}": result}


class EnsembleQualityMetrics(IMetrics):
    """Metrics to compare a target ensemble to the reference ensemble.
    Input metric configs must be a dictionary of the following form:

         {
            "features": {
                "rg": {
                    "feature_params": {"atom_selection": "name CA"},
                    "metric_params": {"js_div": {"bins": 100}},
                },
                "ca_distances": {
                    "feature_params": None,
                    "metric_params": {"js_div": {"bins": 100}},
                },
                "dssp": {
                    "feature_params": {"digitize": True},
                    "metric_params": {
                        "mse_ldist": {"bins": np.array([0, 1, 2, 3, 4])}
                    },
                },
            }
        },

    Specifying computation and metric parameters for each feature/metric
    for comparisons between target and reference ensembles.
    """

    metric_types = set(
        [
            "kl_div",
            "js_div",
            "mse",
            "mse_dist",
            "mse_ldist",
            "fraction_smaller",
            "wasserstein",
        ]
    )
    scalar_metrics = {
        "kl_div": kl_divergence,
        "js_div": js_divergence,
        "mse": mse,
        "mse_dist": mse_dist,
        "mse_ldist": mse_log,
        "fraction_smaller": fraction_smaller,
        "wasserstein": wasserstein,
    }
    vector_metrics = {
        "kl_div": vector_kl_divergence,
        "js_div": vector_js_divergence,
        "mse": vector_mse,
        "mse_dist": vector_mse,
        "mse_ldist": vector_mse_log,
        "wasserstein": vector_wasserstein,
    }

    metrics_2d = {
        "kl_div": kl_divergence,
        "js_div": js_divergence,
        "mse": mse,
        "mse_dist": mse_dist,
        "mse_ldist": mse_log,
        "fraction_smaller": fraction_smaller,
        "wasserstein": wasserstein,
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

    def __init__(self, metrics: Dict):
        super().__init__()
        self.metrics = metrics
        self.metrics_results = {}

    def __call__(
        self,
        target: Ensemble,
        reference: Ensemble,
    ):
        """calls the `compute` method and reports the results
        as a dictionary of metric_name: metric_value pairs
        see compute() for more details.
        """
        self.compute(target, reference)
        return self.report()

    @classmethod
    def from_config(cls, config_file: str):
        """instances an EnsembleQualityMetrics
        from a config file. the config should have the example following structure:

            EnsembleQualityMetrics:
              features:
                rmsd:
                  feauture_params:
                    reference_structure: path_to_struct.pdb
                    atom_selection: "name ca"
                  metric_params:
                    js_div:
                      bins: 100
                    mse_ldist:
                      -bins:
                        start: 0
                        stop: 100
                        num: 1000
                ...

        for specific metrics, bins can be either an integer or a dictionary
        of key value pairs corresponding to kwargs of `np.linspace` to instance
        equal-width bins over a specific range of values. For 2D metrics, a
        list of binopts can be specified through the "-" operator.

        parameters
        ----------
        config_file:
            yaml file specifying feature and config options
        """

        config = yaml.load(open(config_file, "r"))
        eqm_config = config["EnsembleQualityMetrics"]

        for feature in eqm_config["features"].keys():
            feature_dict = eqm_config["features"][feature]
            for metric in feature_dict["metric_params"].keys():
                if "bins" in list(feature_dict["metric_params"][metric].keys()):
                    binopts = feature_dict["metric_params"][metric]["bins"]
                    if isinstance(binopts, int) or binopts is None:
                        # Simple "num bins" or histogram default
                        continue
                    elif isinstance(binopts, Mapping):
                        # 1D specified bin array using np.linspace
                        eqm_config["features"][feature]["metric_params"][
                            metric
                        ]["bins"] = np.linspace(**binopts)
                    elif isinstance(binopts, list):
                        print(binopts)
                        # 2D histogram handling for np.histogram2d
                        if len(binopts) != 2:
                            raise ValueError(
                                f"Currently only 2D distributions are supported"
                            )

                        if all([isinstance(opt, int) for opt in binopts]):
                            # List of ints
                            continue
                        elif all([isinstance(opt, Mapping) for opt in binopts]):
                            # list of 1D arrays for each dimension
                            converted_bins = []
                            for bin_opt in binopts:
                                print(bin_opt)
                                # reinstance bins with np.linspace
                                c_bins = np.linspace(**bin_opt)
                                converted_bins.append(c_bins)

                            eqm_config["features"][feature]["metric_params"][
                                metric
                            ]["bins"] = converted_bins
                        else:
                            raise ValueError(
                                f"Currently, only List[int] or List[dict] are accepted for multiple bin specifications, but {binopts} was supplied"
                            )
                else:
                    raise ValueError(f"unknown bin options {binopts}")
        return cls(eqm_config)

    def compute(
        self,
        target: Ensemble,
        reference: Ensemble,
    ):
        """
        compute the metrics that compare the target ensemble to the reference over
        the specified features. comute metrics are stored in the `EnsembleQualityMetrics.results`
        attribute.

        parameters:
        -----------
        target: Ensemble
            the target ensemble
        reference: Ensemble
            the reference ensemble, against which the target ensemble is compared
        """
        for feature in self.metrics["features"].keys():
            # compute feature in target/ref ensemble if needed
            if feature_params not in list(
                self.metrics["features"][feature].keys()
            ):
                feature_params = {}
            else:
                feature_params = self.metrics["features"][feature][
                    "feature_params"
                ]

            Featurizer.get_feature(target, feature, **feature_params)
            Featurizer.get_feature(reference, feature, **feature_params)
            for metric in self.metrics["features"][feature][
                "metric_params"
            ].keys():
                params = self.metrics["features"][feature]["metric_params"][
                    metric
                ]
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
    ) -> dict[str, float]:
        """computes metric for desired feature between two ensembles.

        parameters
        ----------
        target:
            target ensemble
        reference:
            refernce ensemble
        feature:
            string specifying the feature for which the desired metric should be computed
            over from the target to the reference ensemble. valid features can be
            scalars (eg, `EnsembleQualityMetrics.scalar_features`) or vector features
            (eg, `EnsembleQualityMetrics.vector_features`)
        metric:
            string specifying the metric to compute for the desire feature between the
            target and reference ensembles. valid metrics are contained in
            `EnsembleQualityMetrics.metrics`
        bins:
            in the case that the metric is calculated over probability distributions,
            this integer number of bins or `np.ndarray` of bins is used to compute
            histograms for both the target and reference ensembles

        returns
        -------
        result:
            dict of the form {"{feature}, {metric}" : metric_result} for
            the specified feature and metric between the target and
            reference ensembles.
        """

        if metric not in EnsembleQualityMetrics.metric_types:
            raise ValueError(f"metric '{metric}' not defined.")

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
        result = metric_computer(hist_target, hist_ref, **metric_args)
        return {f"{feature}, {metric}": result}
