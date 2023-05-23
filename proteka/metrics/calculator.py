"""Main entry point for calculating the metrics
"""
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
import numpy as np
import mdtraj as md
from mdtraj.core.element import Element
from typing import Union, Dict, Optional, List, Tuple

from .featurizer import Featurizer, TICATransform
from ..dataset import Ensemble
from .divergence import (
    kl_divergence,
    js_divergence,
    vector_kl_divergence,
    vector_js_divergence,
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
        is composed of hydrogen bonding atom species (e.g., ("N", "O", "S")), then
        an additional default allowance of 0.6 nm is permitted.

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
                if (
                    all(
                        [
                            p in StructuralIntegrityMetrics.acceptors_or_donors
                            for p in pair
                        ]
                    )
                    and pair[0] != pair[1]
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

    def __init__(self, metrics_params=None):
        super().__init__()
        self.metrics_dict = {
            "end2end_distance_kl_div": self.end2end_distance_kl_div,
            "rg_kl_div": self.rg_kl_div,
            "ca_distance_kl_div": self.ca_distance_kl_div,
            "ca_distance_js_div": self.ca_distance_js_div,
            "local_contact_number_js_div": self.local_contact_number_js_div,
            "tica_div": self.tica_div,
        }
        if metrics_params is None:
            metrics_params = {}
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
        hist_target, hist_ref = histogram_features(
            ca_distance_target, ca_distance_reference, bins=100
        )
        kl = kl_divergence(hist_target, hist_ref)
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
        hist_target, hist_ref = histogram_features(
            ca_distance_target, ca_distance_reference, bins=100
        )
        js = js_divergence(hist_target, hist_ref)
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
            target, "local_contact_number"
        )
        # Histogram of the local_contacts. Will use 100 bins from 0 to num_res
        hist_target, hist_ref = histogram_vector_features(
            local_contact_num_target, local_contact_num_reference, bins=100
        )
        js = vector_js_divergence(hist_target, hist_ref)
        return {"local contact number, JS divergence": js}

    @staticmethod
    def tica_div(
        target: Ensemble,
        reference: Ensemble,
        transform: Union[TICATransform, str, None] = None,
    ) -> Dict[str, float]:
        """Perform TICA on the reference enseble and use it to transform target ensemble.
        Then compute KL divergence between the two TICA projections, using the first 2 TICA components

        Parameters
        ----------
        target : Ensemble
            target ensemble
        reference : Ensemble
            reference ensemble, will be used for TICA model fitting
        transform : TICATransform
            TICA transform used to compute TICA

        Returns
        -------
        dict
            Resulting scores
        """
        if transform is None:
            transform = TICATransform(
                features={"ca_distances": {"offset": 1}},
                estimation_params={"lagtime": 10, "dim": 2},
            )
        elif type(transform) == str:
            # Here, the plan is to either load transformer from hdf5 file or
            # deserialize a json string
            raise NotImplementedError()

        tica_reference = Featurizer.get_feature(
            reference, "tica", transform=transform
        )
        # Transform can be modified during the call to get_feature, so we need to extract it again
        # Later, here will be deserialization of the transform
        transform_reference = reference["tica"].metadata["transform"]

        tica_target = Featurizer.get_feature(
            target, "tica", transform=transform_reference
        )
        # histogram data
        hist_target, hist_ref = histogram_features2d(
            tica_target[:, :2], tica_reference[:, :2], bins=100
        )
        # Compute KL divergence
        kl = kl_divergence(hist_target, hist_ref)
        js = js_divergence(hist_target, hist_ref)
        return {"TICA, KL divergence": kl, "TICA, JS divergence": js}

    @staticmethod
    def rg_kl_div(target: Ensemble, reference: Ensemble) -> Dict[str, float]:
        """Computes kl divergence for radius of gyration"""
        rg_reference = Featurizer.get_feature(reference, "rg")
        rg_target = Featurizer.get_feature(target, "rg")

        # Histogram of the distances. Will use 100 bins and bin edges extracted from the reference ensemble
        hist_target, hist_ref = histogram_features(
            rg_target, rg_reference, bins=100
        )
        kl = kl_divergence(hist_target, hist_ref)
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
        hist_target, hist_ref = histogram_features(
            d_e2e_target, d_e2e_reference, bins=100
        )
        kl = kl_divergence(hist_target, hist_ref)
        return {"d end2end, KL divergence": kl}
