"""Generates features vectors from pre-computed features.

It assumes that pre-computed features are dumped in tsv files, with columns
feature_type, query_id, type_id, feature_1, feature_2, ...

"""
import pandas as pd
from collections import defaultdict
from typing import List, Tuple

from smart.ltr.features.features import FeatureType
from smart.utils.dataset import Dataset

# TODO: file/class may be renamed.
class CombinedFeatures:
    def __init__(
        self, feature_dump_files: List[str], feature_types: List[FeatureType]
    ) -> None:
        """Initializes class by loading pre-computed features.

        Args:
            feature_dump_files: List of TSV dumpfiles.
            feature_tpyes: List of corresponding feature types.
        """
        self._feature_names = []
        self._feature_types = []
        self._lookup = {}
        for ft in [FeatureType.Q, FeatureType.T, FeatureType.QT]:
            # NOTE: it is assumed that feature values are floats.
            self._lookup[ft] = defaultdict(lambda: defaultdict(float))

        for dump_filename, feature_type in zip(
            feature_dump_files, feature_types
        ):
            self._load_feature_dump(dump_filename, feature_type)

    def _get_key(
        self,
        feature_type: FeatureType,
        query_id: str,
        type_id: str,
    ) -> str:
        """Generates key from query and type IDs depending on feature type.

        Args:
            feature_type: Feature type.
            query_id: Query ID.
            type_id: Type ID.

        Returns:
            Key containing query and/or type IDs.
        """
        if feature_type == FeatureType.Q:
            return query_id
        elif feature_type == FeatureType.T:
            return type_id
        elif feature_type == FeatureType.QT:
            return f"{query_id}__{type_id}"
        else:
            raise ValueError("Unknown feature type")

    def _load_feature_dump(
        self, dump_filename: str, feature_type: FeatureType
    ) -> None:
        """Loads features from a given dump file.

        Args:
            dump_filename: File name.
            feature_type: Feature type
        """
        # TODO: This is to be implemented by reading data from file.
        # It's illustrated with some dummy content.
        if dump_filename == "dummy_T.tsv":
            file_contents = [
                "type_id    type_feat1  type_feat2",
                "t001   0   1",
                "t002   1   2",
            ]
        elif dump_filename == "dummy_Q.tsv":
            file_contents = [
                "query_id    q_feat1",
                "q01   11",
                "q02   22",
                "q03   33",
            ]
        elif dump_filename == "dummy_QT.tsv":
            file_contents = [
                "query_id   type_id    qt_feat1 qt_feat2    qt_feat3",
                "q01    t001   0.1  0.1 0",
                "q01    t002   0.0  0.1 0",
                "q02    t001   1.0  1.0 0",
                "q03    t002   0.5  0 0.5",
            ]

        # TODO: This should be reading from a file, it's just an illustration.
        # Get feature names from heading row.
        start_idx = 2 if feature_type == FeatureType.QT else 1
        feature_names = file_contents[0].split()[start_idx:]
        print(
            f"{dump_filename} contains the following {feature_type} features: ",
            feature_names,
        )
        self._feature_names.extend(feature_names)
        self._feature_types.extend([feature_type] * len(feature_names))

        # Load feature values.
        for line in file_contents[1:]:
            values = line.split()
            query_id = values[0] if not feature_type == FeatureType.T else None
            type_id = None
            if feature_type == FeatureType.T:
                type_id = values[0]
            elif feature_type == FeatureType.QT:
                type_id = values[1]
            key = self._get_key(feature_type, query_id, type_id)

            key = (
                f"{values[0]}__{values[1]}"
                if feature_type == FeatureType.QT
                else values[0]
            )
            for idx, feature_name in enumerate(feature_names):
                self._lookup[feature_type][feature_name][key] = values[
                    start_idx + idx
                ]

    def get_features(self) -> List[Tuple[str, FeatureType]]:
        """Returns the list of features (names and types)."""
        return [
            (feature_name, feature_type)
            for feature_name, feature_type in zip(
                self._feature_names, self._feature_types
            )
        ]

    def get_feature_vector(self, query_id: str, type_id: str) -> List[float]:
        """Returns the feature vector for an instance (query-type pair).

        Args:
            query_id: Query ID.
            type_id: Type ID.

        Returns:
            List of float values.
        """
        values = []
        for feature_name, feature_type in zip(
            self._feature_names, self._feature_types
        ):
            values.append(
                self._lookup[feature_type][feature_name].get(
                    self._get_key(
                        feature_type, query_id=query_id, type_id=type_id
                    ),
                    0,  # Missing features values default to 0.
                )
            )

        return values


if __name__ == "__main__":
    knowledge_base = "dbpedia"
    split = "train"
    dataset = Dataset(knowledge_base, split)

    feature_dump_files = ["dummy_T.tsv", "dummy_Q.tsv", "dummy_QT.tsv"]
    feature_types = [FeatureType.T, FeatureType.Q, FeatureType.QT]
    cf = CombinedFeatures(feature_dump_files, feature_types)

    # Print combined feature set.
    print(cf.get_features())

    # All types are considered as candidates.
    candidate_types = dataset.get_unique_types()
    # TEMP: overwritten with dummy types
    candidate_types = ["t001", "t002"]

    # Generate feature vectors for all query-type pairs.
    # TEMP: overwritten with dummy queries.
    for q_id in ["q01", "q02", "q03"]:  # dataset.get_queries().keys():
        for type_id in candidate_types:
            # TODO: Use L2RDataFormatter to dump the features in libsvm format.
            # Currently, only printed for illustration
            feature_vector = cf.get_feature_vector(q_id, type_id)
            print(q_id, type_id, feature_vector)

            label = None
            if split == "train":
                # TODO: Look up label if it's a train split.
                # if type expansion is performed, that should be done here
                # (not in Dataset!) by propagating the labels along the path of
                # gt types.
                pass
