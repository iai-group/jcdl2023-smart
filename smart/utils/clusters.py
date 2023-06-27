"""Loads cluster assignment and returns cluster memberships.
"""

import pickle
from typing import Dict, List

import scipy.sparse as sp
from sklearn.preprocessing import MultiLabelBinarizer

_XBERT_DIR = "smart/classification/type/X-Transformer/"


class Clusters:
    def __init__(
        self,
        dataset: str = "dbpedia",
        split: str = "train",
        embedding: str = "pifa-neural",
        model: str = "roberta-large",
    ) -> None:
        model_dir = f"{_XBERT_DIR}/save_models/{dataset}_full/{embedding}-s0"
        matcher_dir = f"{model_dir}/matcher/{model}/"
        indexer_dir = f"{model_dir}/indexer/"
        dataset_dir = f"{_XBERT_DIR}/datasets/{dataset}_full/"
        self._clusters_pred = f"{matcher_dir}/C_{split}_pred.npz"
        self._clusters_file = f"{indexer_dir}/code.npz"
        self._z_preds: sp.csr_matrix = self._load_cluster_pred()
        self._cluster_dict: Dict[
            int, List[int]
        ] = self.get_cluster_assignment_dict()
        self._type_cluster_map = self.get_type_cluster_membership()
        self.qid_map = self._read_id_map_file(
            f"{dataset_dir}/{split}_id_map.txt"
        )
        self._mlb = self.read_multi_label_binarizer_pickle(
            f"{dataset_dir}/mlb.pkl"
        )

    def _load_cluster_assignment(self) -> sp.csr_matrix:
        """Loads cluster membership of types

        Returns:
            Sparse matrix with types in rows and clusters in columns
        """
        return sp.load_npz(self._clusters_file)

    def _load_cluster_pred(self) -> sp.csr_matrix:
        """Loads cluster predictions for questions.

        Returns:
            Sparse matrix with questions in rows and cluster values in columns
        """
        return sp.load_npz(self._clusters_pred)

    def get_cluster_assignment_dict(self) -> Dict[int, List[int]]:
        """This function loads cluster type members dictionary.

        Returns: A sparse csr_matrix

        """
        cluster_dict = {}
        cluster_matrix = self._load_cluster_assignment()
        for i in range(cluster_matrix.shape[0]):
            cluster_id = cluster_matrix.getrow(i).nonzero()[1][0]
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = [i]
            else:
                cluster_dict[cluster_id].append(i)
        return cluster_dict

    def get_type_cluster_membership(self) -> Dict[int, int]:
        """Creates a dict to map the type id to cluster id.

        Returns:
            Dictionary with type id as key and cluster id as value.
        """
        type_cluster_map = {}
        cluster_matrix = self._load_cluster_assignment()
        for i in range(cluster_matrix.shape[0]):
            cluster_id = cluster_matrix.getrow(i).nonzero()[1][0]
            type_cluster_map[i] = cluster_id
        return type_cluster_map

    def get_cluster_preds_int_qid(self, query_id: int) -> List[int]:
        """Returns the prediction scores for all the clusters for a given query.

        Args:
            query_id: Int query id

        Returns:
            An array of prediction scores.
        """
        preds = self._z_preds.getrow(query_id).toarray()[0]
        return preds

    def get_cluster_preds(self, qid: str) -> None or List[float]:
        """Returns the prediction scores for all the clusters for a given query.
        First converts the string qid to int query id.

        Args:
            query_id: String query id for example, dbpedia_xxxxx

        Returns:
            An array of prediction scores.
        """
        if qid not in self.qid_map:
            return None
        return self.get_cluster_preds_int_qid(self.qid_map[qid])

    def _read_id_map_file(self, file) -> Dict[str, int]:
        """Reads mapping between string qids and int qids from a given tsv file.

        Args:
            file: Path to the file containing qid mapping.

        Returns:
            A dictionary of string qid to int qid used by XBERT.
        """
        id_map = {}
        with open(file) as in_file:
            for line in in_file:
                tokens = line.rstrip().split("\t")
                id_map[tokens[0]] = int(tokens[1])
        return id_map

    def read_multi_label_binarizer_pickle(
        self, label_file
    ) -> MultiLabelBinarizer:
        """Loads a MultiLabelBinarizer pickle to reverse transform it after
        training."""
        new_mlb = pickle.load(open(label_file, "rb"))
        return new_mlb

    def get_int_label_id(self, type_str: str) -> int:
        """Given an str type like dbo:Place return it's corresponding int id
        used by xbert.

        Args:
            type_str: type in string format.

        Returns:
            type int id if the label is found in MultiLabelBinarizer object,
            else returns None.
        """
        label_mat = self._mlb.transform([[type_str]])[0]
        if label_mat.shape[0] > 0 and len(label_mat.nonzero()) > 0:
            return label_mat.nonzero()[0][0]
        else:
            return None

    def get_type_cluster(self, type_id: str) -> int:
        """For a given type return the cluster id to which it belongs to.

        Args:
            type_str: type in string format.

        Returns:
            cluster id if the label is found in MultiLabelBinarizer object,
            else returns None.

        """
        type_id_int = self.get_int_label_id(type_id)
        return self._type_cluster_map[type_id_int] if type_id_int else None

    def get_num_clusters(self) -> int:
        """Return number of clusters. Number of keys in cluster_dict is the
        number of clusters.

        Returns:
            Integer value containing number of clusters
        """
        return len(self._cluster_dict)
