"""Tests for DBpedia class."""

import pytest

from smart.utils.dbpedia import DBpedia

DBPEDIA_DUMP = "data/dbpedia/dump_sample"
DBPEDIA_HIERARCHY = "data/dbpedia/dbpedia_types.tsv"

# DBpedia objects to be shared across multiple test cases.
@pytest.fixture
def dbpedia_sample():
    return DBpedia(DBPEDIA_DUMP)


def test_get_types(dbpedia_sample):
    types = dbpedia_sample.get_types()
    assert len(types) == 762
    assert "dbo:Agent" in types
    assert "dbo:Location" not in types  # dbo:Place is to be used instead


def test_is_type(dbpedia_sample):
    assert dbpedia_sample.is_type("dbo:Agent")
    assert not dbpedia_sample.is_type("dbo:Location")


def test_get_max_depth(dbpedia_sample):
    assert dbpedia_sample.get_max_depth() == 7


def test_get_depth(dbpedia_sample):
    assert dbpedia_sample.get_depth("owl:Thing") == 0
    assert dbpedia_sample.get_depth("dbo:Agent") == 1
    assert dbpedia_sample.get_depth("dbo:Person") == 2
    assert dbpedia_sample.get_depth("dbo:Artist") == 3
    assert dbpedia_sample.get_depth("dbo:Dancer") == 4


def test_get_entity_types(dbpedia_sample):
    # Only owl:Thing, which is not returned as type.
    assert len(dbpedia_sample.get_entity_types("Algae")) == 0
    # Only in instance_types.
    assert dbpedia_sample.get_entity_types("Academy_Awards") == {"dbo:Award"}
    # Types both in instance_types and in instance_types_transitive
    assert dbpedia_sample.get_entity_types("Alabama") == {
        "dbo:AdministrativeRegion",
        "dbo:Region",
        "dbo:PopulatedPlace",
        "dbo:Place",
    }


def test_get_type_entities(dbpedia_sample):
    # Type with no entities in sample.
    assert len(dbpedia_sample.get_type_entities("dbo:RacingDriver")) == 0
    # Leaf type with single entity.
    assert dbpedia_sample.get_type_entities("dbo:Film") == {"Actrius"}
    # Leaf type with multiple entities.
    assert dbpedia_sample.get_type_entities("dbo:ArtificialSatellite") == {
        "Apollo_8",
        "Apollo_11",
    }
    # Non-leaf type with single entity.
    assert dbpedia_sample.get_type_entities("dbo:PopulatedPlace") == {"Alabama"}
    # Non-leaf type with multiple entities.
    assert dbpedia_sample.get_type_entities("dbo:Writer") == {
        "Ayn_Rand",
        "Aldous_Huxley",
    }


def test_get_supertype(dbpedia_sample):
    assert dbpedia_sample.get_supertype("owl:Thing") is None
    assert dbpedia_sample.get_supertype("dbo:Agent") == "owl:Thing"
    assert dbpedia_sample.get_supertype("dbo:Person") == "dbo:Agent"
    assert dbpedia_sample.get_supertype("dbo:Artist") == "dbo:Person"
    assert dbpedia_sample.get_supertype("dbo:Dancer") == "dbo:Artist"


def test_get_subtypes(dbpedia_sample):
    assert dbpedia_sample.get_subtypes("dbo:Agent") == {
        "dbo:Family",
        "dbo:Employer",
        "dbo:Person",
        "dbo:Organisation",
        "dbo:Deity",
    }
    assert dbpedia_sample.get_subtypes("dbo:Dancer") is None
