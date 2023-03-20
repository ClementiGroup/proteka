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
    def end2end_distance_kl_div(target: Ensemble, reference: Ensemble) -> dict:
        """Computes maximum and rms z-score for d_ca_ca bonds over ensemble.
            Z-score is defined as (d_ca_ca - mean(d_ca_ca)) / std(d_ca_ca)
            d_ca_ca and std(d_ca_ca) are parametrized based on analysis of the proteka
            and pdb databases. Beware, that this metric will give high deviations for
            cis-proline peptide bonds
        """
        
        try: 
            d_e2e_reference = reference.ca_ca_end2end_distance
        except AttributeError:
            featurizer=  Featurizer(reference)
            featurizer.add_end2end_distance()
            d_e2e_reference = reference.ca_ca_end2end_distance
        
        try: 
            d_e2e_target = target.ca_ca_end2end_distance
        except AttributeError:
            featurizer=  Featurizer(target)
            featurizer.add_end2end_distance()
            d_e2e_target = target.ca_ca_end2end_distance
        
        # Histogram of the distances. Will use 100 bins and bin edges extracted from 
        # the reference ensemble        
        try:
            weights = reference.weights
        except AttributeError:
            weights = None
        hist_reference, bin_edges = np.histogram(d_e2e_reference, bins=100, weights=weights)
        try:
            weights = target.weights
        except AttributeError:
            weights = None
        hist_target, _ = np.histogram(d_e2e_target, bins=bin_edges, weights=weights)
        kl = kl_divergence(hist_reference, hist_target, normalized=False)
        return {"d end2end, KL divergence": kl}
        