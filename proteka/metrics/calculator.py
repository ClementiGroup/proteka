"""Main entry point for calculating the metrics
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import numpy as np
from typing import Union

from .featurizer import Featurizer
from ..dataset import Ensemble
from .divergence import kl_divergence, js_divergence
from .utils import histogram_features


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
    """Class takes a dataset and checks if for chemical integrity
    """
    def __call__(self, ensemble: Ensemble, resolution="CA"):
        self.compute(ensemble, resolution)
        return self.report()

    def compute(self, ensemble: Ensemble, resolution: str ="CA"):
        """Method to compute the metrics
        """

        self.results.update(self.compute_CA_clashes(ensemble))
        self.results.update(self.compute_CA_pseudobonds(ensemble))
        
    
    @staticmethod
    def compute_CA_clashes(ensemble: Ensemble) -> dict:
        """Compute total number of instances when there is a clash between CA atoms
        Clashes are defined as any 2 nonconsecutive CA atoms been closer than  0.4 nm
        
        """
        try: 
            d_ca_ca = ensemble.ca_ca_distances
        except AttributeError:
            featurizer=  Featurizer(ensemble)
            featurizer.add_ca_distances()
            d_ca_ca = ensemble.ca_ca_distances
        
        distances = d_ca_ca
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

    def __init__(self):
        super().__init__()
        self.metrics_dict = {
            "end2end_distance_kl_div": self.end2end_distance_kl_div,
            "rg_kl_div": self.rg_kl_div,
            "ca_distance_kl_div": self.ca_distance_kl_div,
            "ca_distance_js_div": self.ca_distance_js_div,
            "tica_kl_div": self.tica_kl_div,
            "tica_js_div": self.tica_js_div,
        }

    def __call__(
        self,
        target: Ensemble,
        reference: Ensemble,
        metrics: Union[Iterable[str], str] = 'all',
    ):
        self.compute(target, reference, metrics)
        return self.report()

    def compute(
        self,
        target: Ensemble,
        reference: Ensemble,
        metrics: Iterable = ["end2end_distance_kl_div"],
    ):
        """
        Compute the metrics that compare the target ensemble to the reference
        """
        if metrics == 'all':
            metrics = self.metrics_dict.keys()
        for metric in metrics:
            self.results.update(self.metrics_dict[metric](target, reference))
        return

    @staticmethod
    def ca_distance_kl_div(target: Ensemble, reference: Ensemble) -> dict:
        ca_distance_reference = Featurizer.get_feature(reference, "ca_distances")
        ca_distance_target = Featurizer.get_feature(target, "ca_distances")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            ca_distance_reference, ca_distance_target, bins=100
        )
        kl = kl_divergence(hist_ref, hist_target)
        return {"CA distance, KL divergence": kl}

    @staticmethod
    def ca_distance_js_div(target: Ensemble, reference: Ensemble) -> dict:
        ca_distance_reference = Featurizer.get_feature(reference, "ca_distances")
        ca_distance_target = Featurizer.get_feature(target, "ca_distances")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            ca_distance_reference, ca_distance_target, bins=100
        )
        js = js_divergence(hist_ref, hist_target)
        return {"CA distance, JS divergence": js}

    @staticmethod
    def tica_kl_div(target: Ensemble, reference: Ensemble) -> dict:
        return {"TICA, KL divergence": None}

    @staticmethod
    def tica_js_div(target: Ensemble, reference: Ensemble) -> dict:
        return {"TICA, JS divergence": None}

    @staticmethod
    def rg_kl_div(target: Ensemble, reference: Ensemble) -> dict:
        rg_reference = Featurizer.get_feature(reference, "rg")
        rg_target = Featurizer.get_feature(target, "rg")

        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            rg_reference, rg_target, bins=100
        )
        kl = kl_divergence(hist_ref, hist_target)
        return {"Rg, KL divergence": kl}

    @staticmethod
    def end2end_distance_kl_div(target: Ensemble, reference: Ensemble) -> dict:
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
