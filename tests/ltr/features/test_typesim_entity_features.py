"""Test for TypeSimEntityFeatures."""

import pytest

from smart.ltr.features.typesim_entity_features import TypeSimEntityFeatures


@pytest.mark.usefixtures("dbpedia_sample")
def test_typesim_entity_features_dbpedia(dbpedia_sample):
    typesim_entity_features = TypeSimEntityFeatures(dbpedia_sample)
    features_work = typesim_entity_features.get_features(type_id="dbo:Work")
    assert features_work["dbo:Work"] == 1
    assert features_work["dbo:WrittenWork"] == 0.5
    assert features_work["dbo:Writer"] == 0


@pytest.mark.usefixtures("wikidata_dump_sample")
def test_typesim_entity_features_wikidata(wikidata_dump_sample):
    wikidata_type_sim = TypeSimEntityFeatures(wikidata_dump_sample)
    features = wikidata_type_sim.get_features(type_id="river")
    # Type with at least one shared entity
    assert features["watercourse"] > 0.0
    # Type with at least no shared enity
    assert features["country"] == 0.0
    # Self comparision should be 1.0
    assert features["river"] == 1.0
    assert len(features) == len(wikidata_dump_sample.get_types())
