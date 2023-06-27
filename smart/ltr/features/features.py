"""Abstract class representing a set of features, which are to be computed by
the same feature generator."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict


class FeatureType(str, Enum):
    Q = "Q"
    T = "T"
    QT = "QT"


FeatureMap = Dict[str, float]


class Features(ABC):
    def __init__(self, feature_type: FeatureType) -> None:
        """Initializes feature set.

        Args:
            feature_cat: Feature type (Q, T, or QT).
        """
        self._feature_type = feature_type

    @abstractmethod
    def get_features(
        self, query_id: str = None, type_id: str = None
    ) -> FeatureMap:
        """Generates feature values for a given query-type pair.

        Args:
            query_id: Query ID (or None, in case of FeatureType.T).
            type_id: Type ID (or None, in case of FeatureType.Q).

        Returns:
            Feature names and corresponding values.
        """
        pass

    @property
    def feature_type(self) -> FeatureType:
        return self._feature_type
