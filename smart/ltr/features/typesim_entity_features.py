"""Computes type similarity based on the sets of overlapping entities.

Note: This is the code that was used for generating the type similarities for
the type clustering step of X-BERT (and not for LTR-based type ranking).
"""

from smart.ltr.features.features import Features, FeatureMap, FeatureType
from smart.utils.similarities import jaccard
from smart.utils.type_system import TypeSystem


class TypeSimEntityFeatures(Features):
    def __init__(self, type_system: TypeSystem) -> None:
        """Computes type similarity.

        Args:
            type_system: Type system.
        """
        super().__init__(FeatureType.T)
        self._type_system = type_system

    def get_features(self, type_id: str, query_id: str = None) -> FeatureMap:
        """Computes the similarity of the given type against all other types.

        Args:
            type_id: Type ID.
            query_id: Not used (included only to conform to the interface).

        Returns:
            Dictionary with similiarities between the input type and all types
            (including itself).
        """
        typesim = {}
        for t2 in self._type_system.get_types():
            typesim[t2] = jaccard(
                self._type_system.get_type_entities(type_id),
                self._type_system.get_type_entities(t2),
            )
        return typesim
