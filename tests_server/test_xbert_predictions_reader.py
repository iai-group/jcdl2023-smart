import numpy as np
import pytest
from scipy.sparse.csr import csr_matrix
from smart.utils.tfrank_data_generator import LearningToRankDataGenerator


class TestXBERTPredictionsReader:
    @pytest.fixture
    def dbpedia_train_data_reader(self):
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
    def dbpedia_test_data_reader(self):
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

    def test_question_features_train(
        self, dbpedia_train_data_reader, dbpedia_test_data_reader
    ):

        question_features = dbpedia_train_data_reader.x_features
        print(dbpedia_train_data_reader.x_features.shape)
        assert (
            question_features.shape[0] == 9524
            and question_features.shape[1] == 1024
        )
        question_features = dbpedia_test_data_reader.x_features
        assert (
            question_features.shape[0] == 2438
            and question_features.shape[1] == 1024
        )

    def test_cluster_pred(self, dbpedia_train_data_reader):
        cluster_preds = dbpedia_train_data_reader.clusters.z_preds
        assert cluster_preds.shape[0] == 9524
        assert cluster_preds.shape[1] == 64

    def test_type_features(self, dbpedia_train_data_reader):
        type_features = dbpedia_train_data_reader.t_features
        assert type_features.shape[0] == 315
        assert type_features.shape[1] == 1024

    def test_train_feature_correctness(self, dbpedia_train_data_reader):
        qid, features, ranks, docs = dbpedia_train_data_reader.get_ltr_features(
            0
        )
        assert len(qid) == 153
        assert len(features) == 153
        assert len(features[0]) == 2048
        assert len(ranks) == 153
        assert len(docs) == 153
        # qid, features, ranks, docs = dbpedia_train_data_reader.get_ltr_data()
        # assert len(qid) == len(features) == len(ranks) == len(docs)

    def test_test_feature_correctness(self, dbpedia_test_data_reader):
        qid, features, ranks, docs = dbpedia_test_data_reader.get_ltr_features(
            0
        )
        assert len(qid) == 102
        assert len(features) == 102
        assert len(features[0]) == 2048
        assert len(ranks) == 102
        assert len(docs) == 102
        # qid, features, ranks, docs = dbpedia_test_data_reader.get_ltr_data()
        # assert len(qid) == len(features) == len(ranks) == len(docs)

    def test_positive_samples(self, dbpedia_train_data_reader):
        q_id = 0
        x_feats = dbpedia_train_data_reader.get_query_features(q_id)
        features, ranks = dbpedia_train_data_reader.get_positive_ltr_samples(
            q_id
        )
        gold_types = dbpedia_train_data_reader.get_gold_types(q_id)
        assert len(ranks) == len(gold_types) == len(features)
        for feature, gold_type in zip(features, gold_types):
            type_features = dbpedia_train_data_reader.get_type_features(
                gold_type
            )
            # Make sure features in original data matches formatted data
            assert x_feats == feature[:1024]
            # Make sure type features match
            type_features_from_data = feature[1024:]
            assert type_features == type_features_from_data

    def test_negative_samples(self, dbpedia_train_data_reader):
        q_id = 0
        x_feats = dbpedia_train_data_reader.get_query_features(q_id)
        (
            features,
            ranks,
            docs,
        ) = dbpedia_train_data_reader.get_negative_train_ltr_samples(q_id)
        gold_types = dbpedia_train_data_reader.get_gold_types(q_id)
        assert (
            len(features)
            == len(gold_types) * dbpedia_train_data_reader.NEGATIVE_FACTOR
        )
        for feature, neg_type in zip(features, docs):
            type_features = dbpedia_train_data_reader.get_type_features(
                neg_type
            )
            # Make sure features in original data matches formatted data
            assert x_feats == feature[:1024]
            # Make sure type features match
            type_features_from_data = feature[1024:]
            assert type_features == type_features_from_data

    def test_str_label_mapping(self, dbpedia_train_data_reader):
        _, _, _, docs = dbpedia_train_data_reader.get_ltr_features(0)
        num_labels = len(list(dbpedia_train_data_reader.mlb.classes_))
        ranked_types_sparse = csr_matrix((1, num_labels), dtype=np.int8)
        for type_id in docs:
            ranked_types_sparse[0, type_id] = 1
        pred_str_labels = set(
            dbpedia_train_data_reader.mlb.inverse_transform(
                ranked_types_sparse
            )[0]
        )
        assert "dbo:Opera" in pred_str_labels
        assert "dbo:MusicalWork" in pred_str_labels
        assert "dbo:Work" in pred_str_labels
