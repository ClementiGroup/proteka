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

