from smart.utils.clusters import Clusters


def test_cluster_preds():
    cl = Clusters()
    cluster_preds = cl.get_cluster_preds("dbpedia_15835")
    assert len(cluster_preds) == 64
    assert cluster_preds[12] == 0.0005653772696013567
    assert cluster_preds[55] == 0.0004631295634579067


def test_int_qid():
    cl = Clusters()
    assert cl.qid_map["dbpedia_15835"] == 2963
    assert cl.qid_map["dbpedia_14427"] == 0


def test_cluster_members():
    cl = Clusters()
    cluster_dict = cl.get_cluster_assignment_dict()
    assert cluster_dict[0] == [75, 102, 139, 161]


def test_num_clusters():
    cl = Clusters()
    assert cl.get_num_clusters() == 64
