""" This script generates features for learning to rank in libsvm format. It
can dump both sparse (TF-IDF) and dense features (BERT). It also dumps
features per type in separate files to train one model per type.
"""

import argparse
from random import shuffle
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy as sp
import scipy.sparse as smat
from sklearn.preprocessing import normalize as sk_normalize
from smart.utils.learning_to_rank import L2RDataFormatter
from smart.utils.clusters import Clusters
from scipy.sparse import lil_matrix
from tqdm import tqdm


from data.smart_dataset.dbpedia.evaluation.evaluate import (
    load_type_hierarchy,
    get_expanded_types,
    get_type_distance,
)
from smart.classification.type.predict_types_using_xbert import (
    read_multi_label_binarizer_pickle,
)


class LearningToRankDataGenerator:
    """
    Class for reading XBERT feature files
    """

    def __init__(
        self,
        dense: bool,
        split: str = "train",
        dataset: str = "dbpedia",
        embedding: str = "pifa-tfidf",
        model: str = "roberta-large",
        use_x1: bool = False,
        use_x2: bool = True,
        negative_factor: int = 50,
        topk_clusters: int = 3,
        sort_clusters_for_ng: bool = True,
        use_extended_types=True,
    ):
        self.NEGATIVE_FACTOR = negative_factor
        self.TOPK_CLUSTERS = topk_clusters
        self.sort_clusters_for_ng = sort_clusters_for_ng
        self.use_extended_types = use_extended_types
        XBERT_DIR = "smart/classification/type/X-Transformer/"
        DATA_DIR = f"{XBERT_DIR}/datasets/{dataset}_full/"
        MODEL_DIR = f"{XBERT_DIR}/save_models/{dataset}_full/{embedding}-s0"
        MATCHER_DIR = f"{MODEL_DIR}/matcher/{model}/"
        INDEXER_DIR = f"{MODEL_DIR}/indexer/"
        PROC_DIR = f"{XBERT_DIR}/save_models/{dataset}_full/proc_data/"
        TEST_RAW_TEXT_FILE = f'{DATA_DIR}/test_raw_texts.txt'
        label_file = f"{DATA_DIR}/mlb.pkl"
        self.mlb = read_multi_label_binarizer_pickle(label_file)
        self.test_raw_data = self._read_raw_text(TEST_RAW_TEXT_FILE)

        self.split = split
        self.x1_file = f"{DATA_DIR}/X.{split}.npz"
        self.x2_file = f"{MATCHER_DIR}/{split}_embeddings.npy"
        self.y_file = f"{DATA_DIR}/Y_ranked.{split}.npz"
        self.y_unrakned_file = f"{DATA_DIR}/Y.{split}.npz"
        self.clusters_pred = f"{MATCHER_DIR}/C_{split}_pred.npz"
        self.clusters_file = f"{INDEXER_DIR}/code.npz"
        self.type_features_file = f"{PROC_DIR}/L.{embedding}.npz"
        self.x_features: sp.sparse.csr_matrix = self._load_question_features(
            use_x1, use_x2
        )
        self.y_labels: sp.sparse.csr_matrix = self._load_type_labels(
            self.y_file
        )
        self.y_unranked_labels: sp.sparse.csr_matrix = self._load_type_labels(
            self.y_unrakned_file
        )
        self.t_features: sp.sparse.csr_matrix = self._load_type_features()
        self.dense = dense
        self.features_dict: Dict[int, Any] = {}
        self.type_hierarchy_file = (
            "data/smart_dataset/dbpedia/evaluation/dbpedia_types.tsv"
        )
        self.type_hierarchy, self.max_depth = load_type_hierarchy(
            self.type_hierarchy_file
        )
        self.clusters = Clusters(
            dataset=dataset, split=split, embedding=embedding, model=model
        )

    @staticmethod
    def _read_raw_text(file_name: str) -> List[str]:
        lines = []
        with open(file_name, 'r') as f:
            for line in f:
                lines.append(line)
        return lines

    def _load_feature_matrix(
        self, file_name: str, dtype=sp.float32
    ) -> sp.sparse.csr_matrix:
        if file_name.endswith(".npz"):
            return smat.load_npz(file_name).tocsr().astype(dtype)
        elif file_name.endswith(".npy"):
            return smat.csr_matrix(
                np.ascontiguousarray(np.load(file_name), dtype=dtype)
            )
        else:
            raise ValueError("src must end with .npz or .npy")

    def _get_str_label(self, type_id: int) -> str:
        """Given an int id convert it to corresponding str id.

        Args:
            type_id: [description]

        Returns:
            [description]
        """
        num_labels = len(list(self.mlb.classes_))
        ranked_types_sparse = lil_matrix((1, num_labels), dtype=np.int8)
        ranked_types_sparse[0, type_id] = 1
        pred_str_label = self.mlb.inverse_transform(ranked_types_sparse)
        return pred_str_label[0][0]

    def _get_int_label_id(self, type_str: str) -> int:
        """Given an str type like dbo:Place return it's corresponding int id
        used by xbert.

        Args:
            type_str: type in string format.

        Returns:
            type int id if the label is found in MultiLabelBinarizer object,
            else returns None.
        """

        label_mat = self.mlb.transform([[type_str]])[0]
        print(type_str, label_mat)
        if label_mat.shape[0] > 0 and len(label_mat.nonzero()) > 0:
            return label_mat.nonzero()[0][0]
        else:
            return None

    def get_extended_types(
        self, ground_truth_types: set, max_allowed_depth: int = None
    ) -> Dict[str, int]:
        """Loads extended types for a given set of groundtruth types.

        Args:
            ground_truth_types: Set of groundtruth types to expand.
            max_allowed_depth: Maximum number of hops from the groundtruth type.
                Defaults to max_depth means all extended types are included.

        Returns:
            Returns extended groundtruth types and their minimum distance from
            the groundtruth type.
        """
        if max_allowed_depth is None:
            max_allowed_depth = self.max_depth
        extended_types_dict = {}
        for ground_truth_type in ground_truth_types:

            extended_types = get_expanded_types(
                [ground_truth_type], self.type_hierarchy
            )
            for extended_type in extended_types:
                if extended_type not in ground_truth_types:
                    type_distance = get_type_distance(
                        ground_truth_type, extended_type, self.type_hierarchy
                    )
                    if type_distance > max_allowed_depth:
                        continue
                    if (extended_type not in extended_types_dict) or (
                        extended_type in extended_type
                        and type_distance < extended_types_dict[extended_type]
                    ):
                        extended_types_dict[extended_type] = type_distance
        return extended_types_dict

    def _load_question_features(
        self, use_x1: bool, use_x2: bool
    ) -> sp.sparse.csr_matrix:
        X1 = self._load_feature_matrix(self.x1_file)
        if use_x1 and not use_x2:
            return X1
        X2 = self._load_feature_matrix(self.x2_file)
        if use_x2 and not use_x1:
            return X2
        X = smat.hstack(
            [sk_normalize(X1, axis=1), sk_normalize(X2, axis=1)]
        ).tocsr()
        return X

    def _load_type_features(self) -> sp.sparse.csr_matrix:
        return smat.load_npz(self.type_features_file)

    def _load_type_labels(self, file_name) -> sp.sparse.csr_matrix:
        labels = smat.load_npz(file_name)
        return labels

    def _get_predicted_clusters(
        self, query_id: int, sort_by_score: bool = True
    ) -> List[int]:
        """Sorts the given sparse matrix in decreasing order of the values.

        Args:
            matrix: Sparse matrix

        Returns:
            Tuple of row, col and value. Only non-zero rows are returned.
        """
        preds = self.clusters.get_cluster_preds_int_qid(query_id)
        cols = [x for x in range(len(preds))]
        if sort_by_score:
            return sorted(cols, key=lambda x: preds[x], reverse=True)
        else:
            shuffle(cols)
            return cols

    def get_ltr_data(
        self,
    ) -> Tuple[List[int], List[List[float]], List[int], List[int]]:
        """For each query in the dataset, get training samples with qid,
        features, ranks and type ids.

        Returns: A tuple of list of query ids, 2d-array of feature vector and a
        corresponding array of rank labels, and corresponding type ids.

        """
        all_qids = []
        all_features = []
        all_ranks = []
        all_docs = []
        for query_id in tqdm(range(self.x_features.shape[0])):
            qid, features, ranks, docs = self.get_ltr_features(query_id)
            all_qids.extend(qid)
            all_features.extend(features)
            all_ranks.extend(ranks)
            all_docs.extend(docs)
        return all_qids, all_features, all_ranks, all_docs

    def get_ltr_features(
        self, query_id: int
    ) -> Tuple[List[int], List[List[float]], List[int], List[int]]:
        """For a given query id, get positive samples from the gold types and
        negative samples using other types in the same cluster which are not
        gold types.

        Args:
            query_id: query id as int

        Returns: A tuple of list of query ids, 2d-array of feature vector and a
        corresponding array of rank labels, and corresponding type ids.

        """
        all_samples: List[List[float]] = []
        all_ranks: List[int] = []
        all_docs: List[int] = []
        if self.split == "train":
            # All groundtruth positive samples should be in training split.
            positive_samples, pos_ranks = self.get_positive_ltr_samples(
                query_id
            )
            (
                negative_samples,
                neg_ranks,
                neg_types,
            ) = self.get_negative_train_ltr_samples(
                query_id, self.sort_clusters_for_ng
            )
            # First add all positive pairs then negative pairs.
            all_samples.extend(positive_samples)
            all_samples.extend(negative_samples)
            all_ranks.extend(pos_ranks)
            all_ranks.extend(neg_ranks)
            gold_type_label_ids = list(
                self.y_labels.getrow(query_id).nonzero()[1]
            )
            all_docs.extend(gold_type_label_ids)
            all_docs.extend(neg_types)
        else:
            # Test data should have all types from top-k clusters.
            all_samples, all_ranks, all_docs = self.get_test_ltr_samples(
                query_id, self.TOPK_CLUSTERS
            )
        return (
            [query_id] * len(all_samples),
            all_samples,
            all_ranks,
            all_docs,
        )

    def get_positive_ltr_samples(
        self, query_id: int
    ) -> Tuple[List[List[float]], List[int]]:
        """For a given query id, get positive samples from the gold types.

        Args:
            query_id: query id as int
            x_feat: feature vector for the query_id

        Returns: A tuple of 2d-array of feature vector and a corresponding
        array of rank labels.

        """
        query_feat = self.get_query_features(query_id)
        gold_type_label_ids = self.get_gold_types(query_id)
        positive_pairs = []
        ranks = []
        for gold_type in gold_type_label_ids:
            type_features = self.get_type_features(gold_type)
            rank = self.y_labels[query_id, gold_type]
            ranks.append(rank)
            query_feat_copy = []
            query_feat_copy.extend(query_feat)
            query_feat_copy.extend(type_features)
            positive_pairs.append(query_feat_copy)
        return positive_pairs, ranks

    def get_test_ltr_samples(
        self, query_id: int, topk_clusters: int
    ) -> Tuple[List[List[float]], List[int], List[int]]:
        """For a given query id, get candidate types from top num_clusters.

        Args:
            query_id: Query id as int.
            num_clusters: Number of clusters to consider.

        Returns: A tuple of 2d-array of feature vector, array of rank labels
        and array of negative type ids.

        """
        query_feat = self.get_query_features(query_id)
        gold_type_label_ids = self.get_gold_types(query_id)
        cols = self._get_predicted_clusters(query_id, sort_by_score=True)
        negative_pairs = []
        ranks = []
        labels = []
        count = 0
        max_candidates = (self.NEGATIVE_FACTOR + 1) * len(gold_type_label_ids)
        for col in zip(cols):
            if count >= max_candidates:
                break
            cluster_id = col[0]
            cluster_type_ids = self.clusters._cluster_dict[cluster_id]
            for type_id in cluster_type_ids:
                if count >= max_candidates:
                    break
                labels.append(type_id)
                type_features = self.get_type_features(type_id)
                # Extend the query features with type features
                x_feat_copy = []
                x_feat_copy.extend(query_feat)
                x_feat_copy.extend(type_features)
                # Append all negative pairs
                negative_pairs.append(x_feat_copy)
                # For test set if the negative type is in gold types consider
                # it as a positive instance
                if type_id in gold_type_label_ids:
                    rank = self.y_labels[query_id, type_id]
                    ranks.append(rank)
                else:
                    ranks.append(0)
                count += 1
                # Select all types from top-k clusters.

        return negative_pairs, ranks, labels

    def get_negative_train_ltr_samples(
        self, query_id: int, sort_clusters_by_score: bool = True
    ) -> Tuple[List[List[float]], List[int], List[int]]:
        """For a given query id, get negative samples from the same cluster
        it belongs to.

        Args:
            query_id: query id as int
            x_feat: feature vector for the query_id

        Returns: A tuple of 2d-array of feature vector, array of rank labels
        and array of negative type ids.

        """
        query_feat = self.get_query_features(query_id)
        gold_type_label_ids = self.get_gold_types(query_id)
        cols = self._get_predicted_clusters(query_id, sort_clusters_by_score)
        neg_count = 0
        negative_pairs = []
        ranks = []
        neg_labels = []
        neg_samples_target = len(gold_type_label_ids) * self.NEGATIVE_FACTOR
        # neg_samples_target = self.NEGATIVE_FACTOR - len(gold_type_label_ids)
        for col in zip(cols):
            cluster_id = col[0]
            cluster_type_ids = self.clusters._cluster_dict[cluster_id]
            for neg_type_id in cluster_type_ids:
                # If we got enough pairs break.
                if neg_count >= neg_samples_target:
                    break
                # Skip the negative pairs which are in groundtruth
                if neg_type_id in gold_type_label_ids:
                    continue
                neg_labels.append(neg_type_id)
                neg_count += 1
                type_features = self.get_type_features(neg_type_id)
                # Extend the query features with type features
                x_feat_copy = []
                x_feat_copy.extend(query_feat)
                x_feat_copy.extend(type_features)
                # Append all negative pairs
                negative_pairs.append(x_feat_copy)
                ranks.append(0)
        return negative_pairs, ranks, neg_labels

    def get_gold_types(self, query_id: int) -> List[int]:
        gold_type_label_ids = self.y_labels.getrow(query_id).nonzero()[1]
        # If extended types are needed, load the extended list of types and
        # their corresponding relevance score: self.max_depth - type_min_depth.
        if self.use_extended_types:
            # first need to convert int type to str type
            str_gold_labels_set = []
            for gold_type in gold_type_label_ids:
                str_gold_type = self._get_str_label(gold_type)
                str_gold_labels_set.append(str_gold_type)
            extended_types = self.get_extended_types(
                set(str_gold_labels_set), max_allowed_depth=1
            )
            int_extended_types = []
            extended_type_relevance = []
            for extended_type, distance in extended_types.items():
                print(extended_type)
                int_extended_types.append(self._get_int_label_id(extended_type))
                extended_type_relevance.append(self.max_depth - distance)

        return gold_type_label_ids

    def get_query_features(self, query_id: int) -> List[float]:
        """This function returns dense feature vector for the query query_id."""
        return self.x_features.getrow(query_id).todense().tolist()[0]

    def get_type_features(self, gold_label_id: int) -> List[float]:
        """This function returns dense feature vector for the type
        gold_label_id."""
        return self.t_features.getrow(gold_label_id).todense().tolist()[0]


def arg_parser():
    """parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="specify either dbpedia, dbpedia_summaries or "
        + "wikidata it can be a list too",
        default="dbpedia",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="input embedding to be used pifa-tfidf or "
        + "pifa-neural or pifa-neural-rdf2vec or text-emb",
        default="pifa-tfidf",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="transformer model used roberta or xlnet or bert",
        default="roberta",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "-ng",
        "--tfrank_ng",
        type=int,
        nargs="?",
        default=3,
        const=1,
        help="Negative sampling factor for tfrank.",
    )
    parser.add_argument(
        "-nc",
        "--tfrank_nc",
        type=int,
        nargs="?",
        default=3,
        const=1,
        help="Number of top-k clusters to use.",
    )
    parser.add_argument(
        "-sc",
        '--sort_clusters_for_ng',
        action='store_const',
        default=False,
        const=True,
        help='Sort the clusters by their prediction store. Default = True',
    )
    parser.add_argument(
        "-x1",
        '--use_sparse',
        action='store_const',
        default=False,
        const=True,
        help='Use TF-IDF features',
    )
    parser.add_argument(
        "-x2",
        '--use_dense',
        action='store_const',
        default=False,
        const=True,
        help='Use Dense features',
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    dataset = args.dataset
    """TODO: dataset name is inconsistent in xbert and in our dataset make
    it consistent and remove .tolower()"""
    dataset_lower = dataset.lower()
    embedding = args.embedding
    xbert_model_name = ""
    if args.model == "roberta":
        xbert_model_name = "roberta-large"
    if args.model == "xlnet":
        xbert_model_name = "xlnet-large-cased"
    if args.model == "bert":
        xbert_model_name = "bert-large-cased"
    print(args.use_sparse)
    print(args.use_dense)
    # Generate features for train and test splits with dense and sparse features
    for split in ["test", "train"]:
        for dense_flag in [False]:
            xor = LearningToRankDataGenerator(
                dataset=dataset,
                embedding=embedding,
                model=xbert_model_name,
                split=split,
                dense=dense_flag,
                negative_factor=args.tfrank_ng,
                topk_clusters=args.tfrank_nc,
                sort_clusters_for_ng=args.sort_clusters_for_ng,
                use_x1=args.use_sparse,
                use_x2=args.use_dense,
                use_extended_types=True,
            )
            if args.sort_clusters_for_ng:
                sorted_str = "sorted"
            else:
                sorted_str = "random"
            out_file = (
                f'{dataset}_{embedding}_{xbert_model_name}_{split}_use_sparse_'
                f'{args.use_sparse}_use_dense_{args.use_dense}'
                f'_ng{args.tfrank_ng}_nc{args.tfrank_nc}_{sorted_str}_l2r.txt'
            )
            print("Generating ", out_file)
            qid, features, ranks, docs = xor.get_ltr_data()
            print("num queries", len(qid))
            print(
                "num queries x num features ", len(features), len(features[0])
            )
            print("labels size ", len(ranks))
            print("types size ", len(docs))
            ltr_formatter = L2RDataFormatter(qid, features, ranks, docs)
            ltr_formatter.write_libsvm_format(
                f"ranking/data/{out_file}", dense=True
            )
