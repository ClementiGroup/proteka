"""
Calculation of integrity scores: 
- Correct bond distances
- Correct Angles
- Correct dihedral angles
"""


def compute_z_score(array, mean, std):
    """
    Compute z-score for each element of an array assuming
    it is distributed with a given mean and standard deviation.
    """

    return (array - mean) / std
