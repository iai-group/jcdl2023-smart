"""Tests for Wikidata class."""

import pytest

from smart.utils.wikidata import Wikidata

WIKIDATA_DUMP_DIR = "data/wikidata/dump_sample"
WIKIDATA_EXTRACTS_DIR = "data/wikidata/extracts_sample"


# Wikidata object to be shared across multiple test cases.
@pytest.fixture
def wikidata_dump_sample():
    return Wikidata(
        dump_dir=WIKIDATA_DUMP_DIR, extracts_dir=WIKIDATA_EXTRACTS_DIR
    )


# Wikidata object to be shared across multiple test cases.
@pytest.fixture
def wikidata_extracts_sample():
    return Wikidata(extracts_dir=WIKIDATA_EXTRACTS_DIR)


def test_extracts_labels(wikidata_extracts_sample):
    extracts_labels = wikidata_extracts_sample.get_extracts(
        Wikidata.EXTRACTS_FILE_LABEL
    )
    assert "Q12345678" not in extracts_labels
    assert extracts_labels["Q62"] == "San Francisco"
    assert extracts_labels["Q63985"] == "Friedrich von Hagedorn"


def test_extracts_descriptions(wikidata_extracts_sample):
    extracts_desc = wikidata_extracts_sample.get_extracts(
        Wikidata.EXTRACTS_FILE_DESCRIPTION
    )
    assert "Q12345678" not in extracts_desc
    assert (
        extracts_desc["Q62"]
        == "consolidated city-county in California, United States"
    )
    assert extracts_desc["Q63985"] == "German poet"


def test_extracts_instance_of(wikidata_extracts_sample):
    extracts_instance_of = wikidata_extracts_sample.get_extracts(
        Wikidata.EXTRACTS_FILE_INSTANCE_OF
    )
    assert "Q12345678" not in extracts_instance_of
    assert set(extracts_instance_of["Q62"]) == set(
        ["Q3301053", "Q515", "Q3559093"]
    )
    assert extracts_instance_of["Q99"] == ["Q35657"]


def test_extracts_subclass_of(wikidata_extracts_sample):
    extracts_subclass_of = wikidata_extracts_sample.get_extracts(
        Wikidata.EXTRACTS_FILE_SUBCLASS_OF
    )
    assert "Q12345678" not in extracts_subclass_of
    assert set(extracts_subclass_of["Q401"]) == set(["Q29517555", "Q3953107"])
    assert extracts_subclass_of["Q123"] == ["Q18602249"]


def test_get_types(wikidata_extracts_sample):
    all_types = wikidata_extracts_sample.get_types()
    assert "medical attribute" in all_types
    assert "country" in all_types
    assert "cuneiform" not in all_types


def test_get_type_descriptions(wikidata_extracts_sample):
    type_descriptions = wikidata_extracts_sample.get_type_descriptions()
    assert (
        type_descriptions["optical instrument"]
        == "scientific instrument using light waves for image viewing."
    )
    assert (
        type_descriptions["electric vehicle"]
        == "vehicle propelled by one or more electric motors."
    )


def test_get_type_labels(wikidata_extracts_sample):
    type_labels = wikidata_extracts_sample.get_type_labels()
    assert type_labels["Q1751850"] == "optical instrument"
    assert type_labels["Q13629441"] == "electric vehicle"


def test_get_entity_types(wikidata_extracts_sample):
    assert "Q484170" in wikidata_extracts_sample.get_entity_types("Q59254")
    assert wikidata_extracts_sample.get_entity_types("Q697") == {
        "Q6256",
        "Q3624078",
        "Q112099",
    }


def test_get_type_entities(wikidata_extracts_sample):
    # Type with no entities in sample.
    assert (
        len(
            wikidata_extracts_sample.get_type_entities(
                "administrative territorial entity of India"
            )
        )
        == 0
    )
    # Check if specific entity in the sample extract is retrieved.
    assert "Q16506" in wikidata_extracts_sample.get_type_entities(
        "municipality of the Czech Republic"
    )
    # Leaf type with multiple entities.
    assert wikidata_extracts_sample.get_type_entities("public university") == {
        "Q15568",
        "Q15576",
        "Q24676",
    }


def test_type_id(wikidata_extracts_sample):
    assert (
        wikidata_extracts_sample.get_type_id("optical instrument") == "Q1751850"
    )
