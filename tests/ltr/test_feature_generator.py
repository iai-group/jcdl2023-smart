"""Tests for FeatureFenerator."""

import os

import pandas as pd
import pytest
from smart.ltr.feature_generator import FeatureGenerator
from smart.ltr.features.features import FeatureMap, Features, FeatureType
from smart.ltr.features.typesim_entity_features import TypeSimEntityFeatures
from smart.utils.dataset import Dataset
from smart.utils.wikidata import Wikidata

OUTPUT_DIR = "data/tests"
WIKIDATA_DUMP_DIR = "data/wikidata/dump_sample"
WIKIDATA_EXTRACTS_DIR = "data/wikidata/extracts_sample"


class DummyDataset(Dataset):
    def __init__(self) -> None:
        """Creates a dummy dataset class for testing."""
        super().__init__("dummy", None)

    def _load_data(self, filename: str, expand_types: bool = False) -> None:
        self._data = [
            {
                "id": "q1",
                "question": "query1",
                "category": "boolean",
                "type": ["boolean"],
            },
            {
                "id": "q2",
                "question": "query2",
                "category": "resource",
                "type": ["dbo:Opera", "dbo:MusicalWork", "dbo:Work"],
            },
        ]


# Wikidata object to be shared across multiple test cases.
@pytest.fixture
def wikidata_dump_sample():
    return Wikidata(
        dump_dir=WIKIDATA_DUMP_DIR, extracts_dir=WIKIDATA_EXTRACTS_DIR
    )


class DummyFeatures(Features):
    def __init__(self, feature_type: FeatureType) -> None:
        """Creates a dummy feature generator for testing purposes.

        Args:
            feature_type (FeatureType): Feature type.
        """
        super().__init__(feature_type)

    def get_features(self, query_id: str, type_id: str) -> FeatureMap:
        """Returns the same feature values for any input.

        Args:
            query_id (str): Query ID.
            type_id (str): Type ID.

        Returns:
            Dictionary with feature names and corresponding values.
        """
        return {"feat_0": 0, "feat_1": 1, "feat_2": 2}


@pytest.fixture
def feature_generator() -> FeatureGenerator:
    dataset = DummyDataset()
    return FeatureGenerator(dataset, output_dir=OUTPUT_DIR)


@pytest.fixture
def smart_wikidata_full() -> Dataset:
    return Dataset("wikidata", "train")


@pytest.fixture
def wikidata_feature_generator(smart_wikidata_full) -> FeatureGenerator:
    return FeatureGenerator(smart_wikidata_full, output_dir=OUTPUT_DIR)


@pytest.mark.usefixtures("wikidata_dump_sample")
def test_type_sim_features_wikidata(
    wikidata_dump_sample, wikidata_feature_generator
):
    wikidata_type_sim = TypeSimEntityFeatures(wikidata_dump_sample)
    wikidata_feature_generator.compute_and_dump(wikidata_type_sim)
    feature_file = f"{OUTPUT_DIR}/T_TypeSimEntityFeatures.tsv"
    # Checks if features tsv file got created.
    assert os.path.isfile(feature_file)
    # Loads generated feature file into a Pandas DataFrame.
    df = pd.read_csv(feature_file, sep="\t")
    # Checks column names.
    assert df.shape[0] == df.shape[1] - 1
    os.remove(feature_file)


def test_query_features(feature_generator) -> None:
    dummy_features = DummyFeatures(FeatureType.Q)
    feature_generator.compute_and_dump(dummy_features)
    feature_file = f"{OUTPUT_DIR}/Q_DummyFeatures.tsv"
    # Checks if features tsv file got created.
    assert os.path.isfile(feature_file)
    # Loads generated feature file into a Pandas DataFrame.
    df = pd.read_csv(feature_file, sep="\t")
    # Checks column names.
    assert list(df.columns) == ["query_id", "feat_0", "feat_1", "feat_2"]
    # Checks query IDs.
    assert list(df['query_id'].values) == ["q1", "q2"]
    # Checks values for a given feature.
    assert list(df['feat_1'].values) == [1, 1]
    # Delete the dummy files after the test
    os.remove(feature_file)


def test_type_features(feature_generator) -> None:
    dummy_features = DummyFeatures(FeatureType.T)
    feature_generator.compute_and_dump(dummy_features)
    feature_file = f"{OUTPUT_DIR}/T_DummyFeatures.tsv"
    # Checks if features tsv file got created.
    assert os.path.isfile(feature_file)
    # Loads generated feature file into a Pandas DataFrame.
    df = pd.read_csv(feature_file, sep="\t")
    # Checks column names.
    assert list(df.columns) == ["type_id", "feat_0", "feat_1", "feat_2"]
    # Checks query IDs.
    assert sorted(list(df['type_id'].values)) == [
        "dbo:MusicalWork",
        "dbo:Opera",
        "dbo:Work",
    ]
    # Checks values for a given feature.
    assert list(df['feat_2'].values) == [2, 2, 2]
    # Delete the dummy files after the test
    os.remove(feature_file)


def test_query_type_features(feature_generator) -> None:
    dummy_features = DummyFeatures(FeatureType.QT)
    feature_generator.compute_and_dump(dummy_features)
    feature_file = f"{OUTPUT_DIR}/QT_DummyFeatures.tsv"
    # Checks if features tsv file got created.
    assert os.path.isfile(feature_file)
    # Loads generated feature file into a Pandas DataFrame.
    df = pd.read_csv(feature_file, sep="\t")
    # Checks column names.
    assert list(df.columns) == [
        "query_id",
        "type_id",
        "feat_0",
        "feat_1",
        "feat_2",
    ]
    # Checks the number of rows (queries x types).
    assert len(df) == 6
    # Check if all query-type pairs are found.
    queries_types = set(
        [
            "-".join([query_id, type_id])
            for query_id in ["q1", "q2"]
            for type_id in ["dbo:MusicalWork", "dbo:Opera", "dbo:Work"]
        ]
    )
    for _, row in df.iterrows():
        key = "-".join([row["query_id"], row["type_id"]])
        try:
            queries_types.remove(key)
        except KeyError:
            pytest.fail("Missing query-type pair")
    assert len(queries_types) == 0
