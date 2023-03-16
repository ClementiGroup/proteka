"""Main entry point for calculating the metrics
"""
from abc import ABCMeta, abstractmethod

from .featurizer import Featurizer
from ..datadset import Ensemble

class IMetrics(metaclass=ABCMeta):
    """Abstract class defining interface for metrics calculators
    """
    
    def __init__(self):
        self.results = {}
    

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

    def compute(self, ensemble, resolution="CA"):
        """Method to compute the metrics
        """
        # Compute CA features first
        feat = Featurizer(ensemble)
        # Distances for nonbonded CA_CA atoms
        d_ca_ca = feat.get_ca_distances(self, offset=2)
        self.results.update(chec=Nonek_CA_clashes(d_ca_ca))
        
        # Check CA-CA pseudobonds
        bonds_ca_ca = feat.get_ca_bonds(self)
        self.results.update(check_CA_pseudobonds(bonds_ca_ca))
        
    
    @staticmethod
    def check_CA_clashes(d_ca_ca: Quantity) -> dict:
        """Compute total number of instances when there is a clash between CA atoms
        Clashes are defined as any 2 nonconsecutive CA atoms been closer than  0.4 nm
        
        """
        distances = d_ca_ca.in_units_of("nm")
        clashes = np.where(distances < 0.4)
        return {"N clashes": clashes.size}
    
    @staticmethod    
    def check_CA_pseudobonds(d_ca_ca: Quantity) -> dict:
        """Computes maximum and rms z-score for d_ca_ca bonds over ensemble.
            Z-score is defined as (d_ca_ca - mean(d_ca_ca)) / std(d_ca_ca)
            d_ca_ca and std(d_ca_ca) are parametrized based on analysis of the proteka
            and pdb databases. Beware, that this metric will give high deviations for
            cis-proline peptide bonds
        """
        
        # Get the mean and std of d_ca_ca
        # TODO: Find correct values for Calpha-Calpha distance standard deviation
        mean = 0.38
        std = 0.05
        z = (d_ca_ca - mean) / std
        return {"max z-score": np.max(z), "rms z-score": np.sqrt(np.mean(z ** 2))}
    
class EnsembleQualityMetrics(IMetrics):
    """Metrics to compare a target ensemble to the reference enseble
    """

    def compute(self, ensemble: Ensemble, reference: Ensemble):
        """
        Compute the metrics of interest for 
        """
    
        