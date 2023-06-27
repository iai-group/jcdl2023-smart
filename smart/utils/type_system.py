"""Defines a common interface for type systems."""

from abc import ABC, abstractmethod
from typing import Set, Optional


class TypeSystem(ABC):
    def __init__() -> None:
        pass

    @abstractmethod
    def get_types(self) -> Set[str]:
        """Returns the set of types in the type system.

        Returns:
            Set of type IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def is_type(self, type_id: str) -> bool:
        """Checks if a type exists in the type system.

        Args:
            type_id: Type ID.

        Returns:
            True if the type exists in the type system, otherwise False.
        """
        raise NotImplementedError

    @abstractmethod
    def get_depth(self, type_id) -> int:
        """Returns the depth level of a given type.

        Args:
            type_id: Type ID.

        Returns:
            Depth in the hierarchy counted from root (which has a depth of 0).
        """
        raise NotImplementedError

    @abstractmethod
    def get_max_depth(self) -> int:
        """Returns the maximum depth of the type hierarchy."""
        raise NotImplementedError

    @abstractmethod
    def get_entity_types(self, entity_id: str) -> Set[str]:
        """Return the set of types assigned to a given entity.

        Arsg:
            entity_id: Entity ID.

        Returns:
            Set of type IDs (empty set if there are no types assigned).
        """
        raise NotImplementedError

    @abstractmethod
    def get_type_entities(self, type_id: str) -> Set[str]:
        """Returns the set of entities belonging to a given type.
        (It considers only direct type-entity assignments, and transitive ones
        via sub-types.)

        Args:
            type_id: Type ID.

        Returns:
            Set of entity IDs (empty set if no entities for the type).
        """
        raise NotImplementedError

    @abstractmethod
    def get_supertype(self, type_id) -> Optional[str]:
        """Returns the parent type of a given type.

        Args:
            type_id: Type ID.

        Returns:
            ID of the parent type of None (if type_id is ROOT).
        """
        raise NotImplementedError

    @abstractmethod
    def get_subtypes(self, type_id) -> Set[str]:
        """Returns the set of (direct) subtypes for a given type.

        Args:
            type_id: Type ID.

        Returns:
            Set of type IDs (empty if leaf type).
        """
        raise NotImplementedError
