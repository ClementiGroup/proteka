"""Main entry point for calculating the metrics
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from .featurizer import Featurizer
from ..dataset import Ensemble, Quantity
from .divergence import kl_divergence

class IMetrics(metaclass=ABCMeta):
    """Abstract class defining interface for metrics calculators
    """
        
    def __init__(self):
        self.results = {}
        
    @abstractmethod
    def __call__(self, ensemble: Ensemble):
        pass
    

    def report(self):
        return self.results
    

    @abstractmethod
    def compute(self, Ensemble, resolution="CA" 
                ):
        """Method to compute the metrics
        """
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
    def compute_CA_pseudobonds(ensemble: Ensemble) -> dict:
        """Computes maximum and rms z-score for d_ca_ca bonds over ensemble.
            Z-score is defined as (d_ca_ca - mean(d_ca_ca)) / std(d_ca_ca)
            d_ca_ca and std(d_ca_ca) are parametrized based on analysis of the proteka
            and pdb databases. Beware, that this metric will give high deviations for
            cis-proline peptide bonds
        """
        
        # Get the mean and std of d_ca_ca
        mean = 0.38
        std = 0.05
        try: 
            d_ca_ca = ensemble.ca_ca_pseudobonds
        except AttributeError:
            featurizer=  Featurizer(ensemble)
            featurizer.add_ca_bonds()
            d_ca_ca = ensemble.ca_ca_pseudobonds
        
        
        z = (d_ca_ca - mean) / std
        return {"max z-score": np.max(z), "rms z-score": np.sqrt(np.mean(z ** 2))}
    
class EnsembleQualityMetrics(IMetrics):
    """Metrics to compare a target ensemble to the reference ensemble
    """
    
    def __call__(self, target: Ensemble, reference: Ensemble):
        self.compute(target, reference)
        return self.report()

    def compute(self, target: Ensemble, reference: Ensemble):
        """
        Compute the metrics that compare the target ensemble to the reference 
        """
        self.results.update(self.end2end_distance_kl_div(target,reference))
        return

    @staticmethod
    def ca_distance_kl_div(target: Ensemble, reference: Ensemble) -> dict:
        ca_distance_reference = Featurizer.get_feature(reference, "ca_distance")
        ca_distance_target = Featurizer.get_feature(target, "ca_distance")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            ca_distance_reference, ca_distance_target, bins=100
        )
        kl = kl_divergence(hist_ref, hist_target)
        return {"CA distance, KL divergence": kl}

    @staticmethod
    def ca_distance_js_div(target: Ensemble, reference: Ensemble) -> dict:
        ca_distance_reference = Featurizer.get_feature(reference, "ca_distance")
        ca_distance_target = Featurizer.get_feature(target, "ca_distance")
        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_ref, hist_target = histogram_features(
            ca_distance_reference, ca_distance_target, bins=100
        )
        js = js_divergence(hist_ref, hist_target)
        return {"CA distance, JS divergence": js}

    @staticmethod
    def tica_kl_div(target: Ensemble, reference: Ensemble) -> dict:
        raise NotImplementedError("TICA KL divergence is not implemented yet")

    @staticmethod
    def tica_js_div(target: Ensemble, reference: Ensemble) -> dict:
        raise NotImplementedError("TICA JS divergence is not implemented yet")
        return {"d end2end, KL divergence": kl}
        