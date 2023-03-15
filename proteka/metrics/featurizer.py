"""Featurizer takes a dataset and extract features from it
    """
import numpy as np
import mdtraj as md
from ..dataset import Ensemble, Quantity


class Featurizer:
    """Extract features from an Ensemble entity and 
       return them as Quantity objects"""

    def __init__(self, ensemble: Ensemble):
        self.ensemble = ensemble

    def get_ca_bonds(self) -> Quantity:
        """
        Returns a numpy array containing length of pseudobonds between consecutive CA atoms.
        """
        # TODO: implement this method
