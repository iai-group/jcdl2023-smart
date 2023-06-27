"""Tests smart.ltr.features.cluster_features.ClusterFeatures class.
"""

from smart.ltr.features.cluster_features import ClusterFeatures
from smart.utils.dataset import Dataset


def test_dbpedia_cluster_features_train_split():
    cluster_features = ClusterFeatures(Dataset("dbpedia", "train"))
    feature_map = cluster_features.get_features(
        query_id="dbpedia_881", type_id="dbo:Person"
    )
    assert feature_map["cluster_score_13"] == 0.0001324370583509946
    assert feature_map["cluster_score_25"] == 0.0005355250941971144
    assert feature_map["cluster_score_27"] == 9.885997050698897e-05
    assert feature_map["cluster_score_61"] == 0.0
    # At least one cluster should contain this type
    assert any(val == 1.0 for _, val in feature_map.items())


def test_dbpedia_cluster_features_test_split():
    cluster_features = ClusterFeatures(Dataset("dbpedia", "test"))
    feature_map = cluster_features.get_features(
        query_id="dbpedia_8484", type_id="dbo:Organisation"
    )
    assert feature_map["cluster_score_13"] == 0.0
    assert feature_map["cluster_score_26"] == 0.0020568486738660025
    assert feature_map["cluster_score_27"] == 0.0485571310916887
    assert feature_map["cluster_score_7"] == 0.9266889706795073
    # At least one cluster should contain this type
    assert any(val == 1.0 for _, val in feature_map.items())


def test_wikidata_cluster_features_train_split():
    cluster_features = ClusterFeatures(Dataset("wikidata", "train"))
    feature_map = cluster_features.get_features(
        query_id="24488", type_id="soft drink"
    )
    print(feature_map)
    assert feature_map["cluster_score_172"] == 0.13631812451285852
    assert feature_map["cluster_score_468"] == 0.0
    assert feature_map["cluster_score_154"] == 1.0
    # At least one cluster should contain this type
    assert any(val == 1.0 for _, val in feature_map.items())


def test_wikidata_cluster_features_test_split():
    cluster_features = ClusterFeatures(Dataset("wikidata", "test"))
    feature_map = cluster_features.get_features(
        query_id="20258", type_id="country"
    )
    assert feature_map["cluster_score_300"] == 0.00861532813409974
    assert feature_map["cluster_score_468"] == 1.0
    assert feature_map["cluster_type_membership_502"] == 1.0
