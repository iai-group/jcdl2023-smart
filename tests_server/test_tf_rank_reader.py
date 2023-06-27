import numpy as np
import pytest
from scipy.sparse import lil_matrix

from smart.utils.tf_rank_to_smart import TFRankOutputReader
from smart.utils.tfrank_data_generator import LearningToRankDataGenerator


@pytest.fixture
def dbpedia_test_data_reader():
    dataset = "dbpedia"
    embedding = "pifa-neural"
    model = "xlnet-large-cased"
    split = "test"
    dense_flag = False
    xpr = LearningToRankDataGenerator(
        dataset=dataset,
        embedding=embedding,
        model=model,
        split=split,
        dense=dense_flag,
        use_extended_types=False,
    )
    return xpr


@pytest.fixture
def dbpedia_train_data_reader():
    dataset = "dbpedia"
    embedding = "pifa-neural"
    model = "xlnet-large-cased"
    split = "train"
    dense_flag = False
    xpr = LearningToRankDataGenerator(
        dataset=dataset,
        embedding=embedding,
        model=model,
        split=split,
        dense=dense_flag,
        use_extended_types=False,
    )
    return xpr


@pytest.fixture
def dbpedia_train_data_reader_extended():
    dataset = "dbpedia"
    embedding = "pifa-neural"
    model = "xlnet-large-cased"
    split = "train"
    dense_flag = False
    xpr = LearningToRankDataGenerator(
        dataset=dataset,
        embedding=embedding,
        model=model,
        split=split,
        dense=dense_flag,
    )
    return xpr


def test_str_to_int_label(dbpedia_train_data_reader):
    assert (
        dbpedia_train_data_reader._get_int_label_id("dbo:AcademicJournal") == 0
    )
    assert (
        dbpedia_train_data_reader._get_int_label_id("dbo:NaturalRegion") == 190
    )


def test_int_to_str_label(dbpedia_train_data_reader):
    assert dbpedia_train_data_reader._get_str_label(0) == "dbo:AcademicJournal"
    assert dbpedia_train_data_reader._get_str_label(190) == "dbo:NaturalRegion"


def test_tf_rank_reader(dbpedia_test_data_reader):
    xpr = dbpedia_test_data_reader
    num_labels = len(list(xpr.mlb.classes_))
    ranked_types_sparse = lil_matrix((1, num_labels), dtype=np.int8)
    tfrank_reader = TFRankOutputReader()
    rankings = tfrank_reader._rankings.get(0)

    ground_truth_labels = {"dbo:Country", "dbo:Planet", "dbo:Port"}
    ground_truth_label_relevance = {
        "dbo:Country": 3,
        "dbo:Planet": 2,
        "dbo:Port": 1,
    }
    print(ranked_types_sparse.shape)
    for ranking in rankings:
        if ranking["label"] > 0:
            print(ranking)
            assert type(ranking["pred_score"]) is float
            ranked_types_sparse[0, ranking["doc_id"]] = 1

    pred_str_labels = list(xpr.mlb.inverse_transform(ranked_types_sparse)[0])
    print(pred_str_labels)
    for str_label, ranking in zip(pred_str_labels, rankings):
        if ranking["label"] > 0:
            print(ranking)
            assert ranking["label"] == ground_truth_label_relevance[str_label]
    for label_str in pred_str_labels:
        assert label_str in ground_truth_labels


def test_extended_types(dbpedia_train_data_reader_extended):
    xpr = dbpedia_train_data_reader_extended
    ground_truth_labels = {"dbo:Country", "dbo:PopulatedPlace", "dbo:Place"}
    extended_types_gt = {
        'dbo:NaturalRegion': 2,
        'dbo:GatedCommunity': 1,
        'dbo:Continent': 1,
        'dbo:Agglomeration': 1,
        'dbo:HistoricalSettlement': 2,
        'dbo:Village': 2,
        'dbo:CityDistrict': 2,
        'dbo:Community': 1,
        'dbo:Settlement': 1,
        'dbo:HistoricalCountry': 1,
        'dbo:City': 2,
        'dbo:Intercommunality': 1,
        'dbo:Region': 1,
        'dbo:Town': 2,
        'dbo:Street': 1,
        'dbo:State': 1,
        'dbo:Territory': 1,
        'dbo:OldTerritory': 2,
        'dbo:HistoricalRegion': 2,
        'dbo:AdministrativeRegion': 2,
        'dbo:Island': 1,
        'dbo:Atoll': 2,
        'dbo:Locality': 1,
        'dbo:Glacier': 2,
        'dbo:MilitaryStructure': 2,
        'dbo:CountrySeat': 1,
        'dbo:Garden': 1,
        'dbo:Arena': 2,
        'dbo:Zoo': 2,
        'dbo:Infrastructure': 2,
        'dbo:ArchitecturalStructure': 1,
        'dbo:Galaxy': 2,
        'dbo:MountainRange': 2,
        'dbo:Beach': 2,
        'dbo:Park': 1,
        'dbo:MountainPass': 2,
        'dbo:BodyOfWater': 2,
        'dbo:SiteOfSpecialScientificInterest': 1,
        'dbo:Venue': 2,
        'dbo:NaturalPlace': 1,
        'dbo:Pyramid': 2,
        'dbo:WineRegion': 1,
        'dbo:Monument': 1,
        'dbo:CelestialBody': 1,
        'dbo:ConcentrationCamp': 1,
        'dbo:Desert': 2,
        'dbo:HistoricPlace': 1,
        'dbo:Forest': 2,
        'dbo:Cemetery': 1,
        'dbo:Planet': 2,
        'dbo:CoalPit': 2,
        'dbo:Asteroid': 2,
        'dbo:Mine': 1,
        'dbo:Archipelago': 2,
        'dbo:Cave': 2,
        'dbo:Volcano': 2,
        'dbo:SportFacility': 2,
        'dbo:Valley': 2,
        'dbo:Cape': 2,
        'dbo:WorldHeritageSite': 1,
        'dbo:Tunnel': 2,
        'dbo:Memorial': 2,
        'dbo:Square': 2,
        'dbo:Building': 2,
        'dbo:Tower': 2,
        'dbo:HotSpring': 2,
        'dbo:AmusementParkAttraction': 2,
        'dbo:Satellite': 2,
        'dbo:Mill': 2,
        'dbo:Swarm': 2,
        'dbo:ProtectedArea': 1,
        'dbo:Mountain': 2,
        'dbo:Crater': 2,
        'dbo:Gate': 2,
        'dbo:Star': 2,
        'dbo:GraveMonument': 2,
        'dbo:Constellation': 2,
    }
    depth = 2
    extended_types = xpr.get_extended_types(ground_truth_labels, depth)
    for type, depth in extended_types_gt.items():
        assert type in extended_types
        assert extended_types[type] == depth
