"""Main entry point for calculating the metrics
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import numpy as np
from typing import Union, Dict

from .featurizer import Featurizer
from ..dataset import Ensemble
from .divergence import (
    kl_divergence,
    js_divergence,
    vector_kl_divergence,
    vector_js_divergence,
)
from .utils import (
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
        return {"N clashes": clashes.size}

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

    def __init__(self, metrics_params={"tica_div": {"lagtime": 10}}):
        super().__init__()
        self.metrics_dict = {
            "end2end_distance_kl_div": self.end2end_distance_kl_div,
            "rg_kl_div": self.rg_kl_div,
            "ca_distance_kl_div": self.ca_distance_kl_div,
            "ca_distance_js_div": self.ca_distance_js_div,
            "local_contact_number_js_div": self.local_contact_number_js_div,
            "tica_div": self.tica_div,
        }
        self.metrics_params = metrics_params

    def __call__(
        self,
        target: Ensemble,
        reference: Ensemble,
        metrics: Union[Iterable[str], str] = "all",
    ):
        """Calls the `compute` method and reports the results
        as a dictionary of metric_name: metric_value pairs
        See compute() for more details.
        """
        self.compute(target, reference, metrics)
        return self.report()

    def compute(
        self,
        target: Ensemble,
        reference: Ensemble,
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
        metrics: Iterable of strings or str
            The metrics to compute. If "all" is passed, all available metrics will be computed
        """
        if metrics == "all":
            metrics = self.metrics_dict.keys()
        elif isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            params = self.metrics_params.get(metric)
            if params is None:
                self.results.update(
                    self.metrics_dict[metric](target, reference)
                )
            else:
                self.results.update(
                    self.metrics_dict[metric](target, reference, **params)
                )
        return

    @staticmethod
    def ca_distance_kl_div(
        target: Ensemble, reference: Ensemble
    ) -> Dict[str, float]:
        """Compute the KL divergence for a mixed histogram of CA distances

        All the pairwise distances are computed for each ensemble and then
        a histogram for all the distances simultaneously [1]_ is computed for both
        ensembles. The KL divergence is then computed between the two histograms.

        Reference:
        ----------
        .. [1] M.G. Reese, O. Lund, J. Bohr, H. Bohr, J.E. Hansen, S. Brunak,
        Distance distributions in proteins: a six-parameter representation,
        Protein Engineering, Design and Selection, Volume 9, Issue 9,
        September 1996, Pages 733–740, https://doi.org/10.1093/protein/9.9.733

        """

        ca_distance_reference = Featurizer.get_feature(
            reference, "ca_distances"
        )
        ca_distance_target = Featurizer.get_feature(target, "ca_distances")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            ca_distance_reference, ca_distance_target, bins=100
        )
        kl = kl_divergence(hist_ref, hist_target)
        return {"CA distance, KL divergence": kl}

    @staticmethod
    def ca_distance_js_div(
        target: Ensemble, reference: Ensemble
    ) -> Dict[str, float]:
        ca_distance_reference = Featurizer.get_feature(
            reference, "ca_distances"
        )
        ca_distance_target = Featurizer.get_feature(target, "ca_distances")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            ca_distance_reference, ca_distance_target, bins=100
        )
        js = js_divergence(hist_ref, hist_target)
        return {"CA distance, JS divergence": js}

    @staticmethod
    def local_contact_number_js_div(
        target: Ensemble, reference: Ensemble
    ) -> Dict[str, np.ndarray]:
        """Calculates local contact number JS divergence PER atom/residue"""

        local_contact_num_reference = Featurizer.get_feature(
            reference, "local_contact_number"
        )
        local_contact_num_target = Featurizer.get_feature(
            reference, "local_contact_number"
        )

        # Histogram of the local_contacts. Will use 100 bins from 0 to num_res
        hist_ref, hist_target = histogram_vector_features(
            local_contact_num_reference, local_contact_num_target, bins=100
        )
        js = vector_js_divergence(hist_ref, hist_target)
        return {"local contact number, JS divergence": js}

    @staticmethod
    def tica_div(
        target: Ensemble, reference: Ensemble, **kwargs
    ) -> Dict[str, float]:
        """Perform TICA on the reference enseble and use it to transform target ensemble.
        Then compute KL divergence between the two TICA projections, using the first 2 TICA components

        Parameters
        ----------
        target : Ensemble
            target ensemble
        reference : Ensemble
            reference ensemble, will be used for TICA model fitting

        Returns
        -------
        dict
            Resulting scores
        """
        from deeptime.decomposition import TICA

        # Fit TICA model on the reference ensemble
        estimator = TICA(dim=2, **kwargs)
        # will fit on the CA distances of the reference ensemble.
        ca_reference = Featurizer.get_feature(reference, "ca_distances")
        estimator.fit(ca_reference)
        model = estimator.fetch_model()
        # Transform the reference ensemble
        tica_reference = model.transform(ca_reference)
        # Transform the target ensemble
        ca_target = Featurizer.get_feature(target, "ca_distances")
        tica_target = model.transform(ca_target)
        # histogram data
        hist_ref, hist_target = histogram_features2d(
            tica_reference, tica_target, bins=100
        )
        # Compute KL divergence
        kl = kl_divergence(hist_ref, hist_target)
        js = js_divergence(hist_ref, hist_target)
        return {"TICA, KL divergence": kl, "TICA, JS divergence": js}

    @staticmethod
    def rg_kl_div(target: Ensemble, reference: Ensemble) -> Dict[str, float]:
        """Computes kl divergence for radius of gyration"""
        rg_reference = Featurizer.get_feature(reference, "rg")
        rg_target = Featurizer.get_feature(target, "rg")

        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            rg_reference, rg_target, bins=100
        )
        kl = kl_divergence(hist_ref, hist_target)
        return {"Rg, KL divergence": kl}

    @staticmethod
    def end2end_distance_kl_div(
        target: Ensemble, reference: Ensemble
    ) -> Dict[str, float]:
        """Computes kl divergence for end2end distance. Currently work with a single chain."""
        d_e2e_reference = Featurizer.get_feature(reference, "end2end_distance")
        d_e2e_target = Featurizer.get_feature(target, "end2end_distance")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from
        # the reference ensemble
        hist_reference, hist_target = histogram_features(
            d_e2e_reference, d_e2e_target, bins=100
        )
        kl = kl_divergence(hist_reference, hist_target)
        return {"d end2end, KL divergence": kl}
