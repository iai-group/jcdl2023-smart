"""Computation of common similarity measures."""

from typing import Set


def jaccard(set1: Set, set2: Set) -> float:
    """Computes the Jaccard similarity of two sets.

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Similarity in [0..1].
    """
    if len(set1) == 0 or len(set2) == 0:
        return 0
    return float(len(set1.intersection(set2)) / len(set1.union(set2)))
