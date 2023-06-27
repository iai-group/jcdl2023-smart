"""Test for HierarchyFeatures."""

import pytest

from smart.ltr.features.hierarchy_features import HierarchyFeatures


@pytest.mark.usefixtures("dbpedia_sample")
def test_hierarchy_features(dbpedia_sample):
    hierarchy_features = HierarchyFeatures(dbpedia_sample)
    # dbo:Person
    features_person = hierarchy_features.get_features(type_id="dbo:Person")
    assert features_person["depth"] == 2 / 7
    assert features_person["num_children"] == 50
    assert features_person["num_siblings"] == 4
    # 2 from instance_types (Allan_Dwan, Andrei_Tarkovsky)
    # 2 from instance_types_transitive (Abraham_Lincoln, Alain_Connes)
    assert features_person["num_entities"] == 4

    # dbo:WrittenWork
    features_writtenwork = hierarchy_features.get_features(
        type_id="dbo:WrittenWork"
    )
    assert features_writtenwork["depth"] == 2 / 7
    assert features_writtenwork["num_children"] == 15
    assert features_writtenwork["num_siblings"] == 14
    # 1 from instance_types_transitive (Animalia_(book))
    assert features_writtenwork["num_entities"] == 1
