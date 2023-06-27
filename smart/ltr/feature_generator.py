"""Generates features for a learning-to-rank approach.

Computation is broken down into various feature subsets.

Typical usage (computing and dumping a set of features to a file):
```
fg = FeatureGenerator(dataset, output_dir="output_folder_name")
features_cls = YourFeatures()  # Class that implements the Features interface
fg.compute_and_dump(features_cls)
```
"""

import argparse
import os
from typing import Any, List, TextIO, Tuple

from smart.utils.dataset import Dataset
from smart.utils.dbpedia import DBpedia
from smart.ltr.features.features import Features, FeatureType
from smart.ltr.features.hierarchy_features import HierarchyFeatures
from smart.ltr.features.typesim_entity_features import TypeSimEntityFeatures
from smart.utils.wikidata import Wikidata

# TODO: Integrate features from old codebase
# See: https://github.com/iai-group/wsdm2022-smart/issues/96
# from smart.ltr.features.w2v_features import W2VFeatures


class FeatureGenerator:
    def __init__(self, dataset: Dataset, output_dir: str) -> None:
        """Initializes a feature generation instance.

        Args:
            dataset (Dataset): Dataset.
            output_dir (str): Output directory.
        """
        self._dataset = dataset
        self._queries = dataset.get_queries()
        self._output_dir = output_dir
        self._header_written = None

    def _write_features_to_file(
        self,
        f_out: TextIO,
        feature_type: FeatureType,
        query_id: str,
        type_id: str,
        features: List[Any],
    ) -> None:
        """Writes a set of features to output file.

        Args:
            f_out (TextIO): Output file stream.
            feature_type (FeatureType): Feature type.
            query_id (str): Query ID.
            type_id (str): Type ID.
            features (List[Any]): List of feature values.
        """
        prefix = []
        prefix_header = []
        if feature_type in [FeatureType.Q, FeatureType.QT]:
            prefix.append(query_id)
            prefix_header.append("query_id")
        if feature_type in [FeatureType.T, FeatureType.QT]:
            prefix.append(type_id)
            prefix_header.append("type_id")

        # Write header with feature names.
        if not self._header_written:
            f_out.write(
                "\t".join(prefix_header + [k for k, _ in features.items()])
                + "\n"
            )
            self._header_written = True

        # Write feature values.
        f_out.write(
            "\t".join(prefix + [str(v) for _, v in features.items()]) + "\n"
        )

    def _instance_generator(self, feature_type: FeatureType) -> Tuple[str, str]:
        """Yields (query_id, type_id) tuples for feature generation, depending
        on the feature type.

        For query-type (QT) features, all possible combinations of queries and
        type are enumerated. For query-only (Q) or type-only (T) features, only
        queries and types are enumerated, respectively, while the other
        component is set to None.

        Args:
            feature_type (FeatureType): Feature type (Q, T or QT).

        Yields:
            Tuple[str, str]: (query ID, type ID) tuple.
        """
        query_ids = (
            self._queries.keys() if feature_type != FeatureType.T else [None]
        )

        type_ids = [None]
        if feature_type != FeatureType.Q:
            type_ids = self._dataset.get_unique_types(lowercase=False)

        for query_id in query_ids:
            for type_id in type_ids:
                yield query_id, type_id

    def compute_and_dump(self, features_cls: Features) -> None:
        """Computes a subset of features and dumps the results to a tsv file.

        Args:
            features_cls: Features class instance.
        """
        features_name = features_cls.__class__.__name__
        feature_type = features_cls.feature_type
        print(f"Computing features {feature_type.name}-{features_name}...")
        os.makedirs(self._output_dir, exist_ok=True)  # Create output folder.
        with open(
            f"{self._output_dir}/{feature_type.name}_{features_name}.tsv", "w"
        ) as fout:
            self._header_written = False
            # Compute features for all query-type pairs.
            for query_id, type_id in self._instance_generator(feature_type):
                features = features_cls.get_features(
                    query_id=query_id, type_id=type_id
                )
                self._write_features_to_file(
                    fout, feature_type, query_id, type_id, features
                )


def arg_parser():
    """Function to parse command line arguments.

    Returns: ArgumentParser object with parsed arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge_base",
        type=str,
        help="specify either dbpedia, dbpedia_summaries or wikidata. \
        It can be a list too",
        default="dbpedia",
        const=1,
        nargs="?",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    knowledge_base = args.knowledge_base
    # /data/collections/dbpedia/dbpedia-2016-10/core-i18n/en/
    dbpedia_dump_path = "data/dbpedia/dump_sample/"
    # wikidata_dump_path = /data/collections/wikidata/20210519/
    wikidata_dump_path = "data/wikidata/dump_sample/"
    # wikidata_extract_path = /data/collections/wikidata/20210519/extracts/
    wikidata_extract_path = "data/wikidata/extracts_sample/"
    dataset = Dataset(knowledge_base, "train")
    output_dir = (
        f"data/{knowledge_base}/type_features"  # "data/dbpedia/features_train"
    )
    if knowledge_base == "dbpedia":
        type_system = DBpedia(dbpedia_dump_path)
    elif knowledge_base == "wikidata":
        type_system = Wikidata(wikidata_dump_path, wikidata_extract_path)
    else:
        raise NotImplementedError(
            f"Knowledge base {knowledge_base} is not supported!"
        )
    fg = FeatureGenerator(dataset, output_dir=output_dir)
    for f in ["typesim_entities", "hierarchy", "w2v"]:
        if f == "typesim_entities":
            # fg.load_type_entities()
            features_cls = TypeSimEntityFeatures(type_system)
        # Currently hierarchy features are supported only for DBPedia
        elif f == "hierarchy" and knowledge_base == "dbpedia":
            # fg.load_entity_types()
            features_cls = HierarchyFeatures(type_system)
        # TODO: https://github.com/iai-group/wsdm2022-smart/issues/96
        #     elif f == "w2v":
        #         features_cls = W2VFeatures(fg.types)
        fg.compute_and_dump(features_cls)
