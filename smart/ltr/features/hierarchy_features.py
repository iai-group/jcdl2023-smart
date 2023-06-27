"""Computes type hierarchy features."""

from smart.ltr.features.features import Features, FeatureMap, FeatureType
from smart.utils.type_system import TypeSystem


class HierarchyFeatures(Features):
    def __init__(self, type_system: TypeSystem) -> None:
        """Computes type hierarchy features.

        Args:
            type_system: Type system.
        """
        super().__init__(FeatureType.T)
        self._type_system = type_system

    def get_features(self, type_id: str, query_id: str = None) -> FeatureMap:
        """Computes type hierarchy related features.

        Args:
            type_id: Type ID.
            query_id: Not used (included only to conform to the interface).

        Returns:
            Features map with keys 'depth', 'num_children', 'num_siblings',
            'num_entities'.
        """
        if not self._type_system.is_type(type_id):
            print(f"Missing type: {type_id}")
        return {
            # Depth, normalized by the max depth of the taxonomy.
            'depth': self._type_system.get_depth(type_id)
            / self._type_system.get_max_depth(),
            # Number of children, i.e., subtypes.
            'num_children': len(self._type_system.get_subtypes(type_id)),
            # Number of siblings = number of subtypes of parent type - 1.
            'num_siblings': len(
                self._type_system.get_subtypes(
                    self._type_system.get_supertype(type_id)
                )
            )
            - 1,
            # Number of entities assigned to that type.
            'num_entities': len(self._type_system.get_type_entities(type_id)),
        }
