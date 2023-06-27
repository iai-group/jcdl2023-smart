"""Generates features for queries in terms of cluster membership. Each query is
represented by the relevance score for each cluster returned by the mather
phase.

Usage:
    cf = ClusterFeatures()
    for index, row in cf._dataset_df.iterrows():
        if row["category"] == "resource":
            print(row["id"], cf.get_features(row["id"], "dbo:Person"))
"""

from smart.ltr.features.features import Features, FeatureMap, FeatureType
from smart.utils.clusters import Clusters
from smart.utils.dataset import Dataset


class ClusterFeatures(Features):
    def __init__(
        self,
        dataset: Dataset,
        embedding: str = "pifa-neural",
        model: str = "xlnet-large-cased",
    ):
        """Computes cluster membership features.

        Args:
            dataset: Dataset for which the features are to be generated
            currently dbpedia and wikidata are supported.
            split: train or test split
        """
        # Query features w.r.t cluster membership
        super().__init__(FeatureType.Q)
        self._clusters = Clusters(
            dataset.get_dataset_name(),
            dataset.get_split_name(),
            embedding,
            model,
        )
        self._num_clusters = self._clusters.get_num_clusters()
        self._dataset_df = dataset.get_df()

    def get_features(self, query_id: str, type_id: str = None) -> FeatureMap:
        """Given a query_id compute the cluster features for this query.

        Args:
            query_id: String query id which is the query id used in
            smart datasets e.g, dbpedia_xxxxx for dbpedia and an integer value
            for wikidata. Defaults to None.
            type_id (optional): If not None one-hot encoding specifying the
            cluster membership of the type id is also included in feature map.

        Returns:
            A dict of feature name and values.
        """
        cluster_preds = (
            self._clusters.get_cluster_preds(query_id) if query_id else None
        )
        feature_map = {
            f"cluster_score_{cluster}": pred_score
            for (cluster, pred_score) in enumerate(cluster_preds)
        }
        if type_id is not None:
            feature_map.update(self._get_type_features(type_id))
        return feature_map

    def _get_type_features(self, type_id: str) -> FeatureMap:

        """Given a type_id compute the cluster membership one-hot encoding
        for this type.

        Args:
            type_id: type id like dbo:Person for which one-hot
            encoding is needed.

        Returns:
            A dict of feature name and values. All clusters except the one to
            which the type_id belongs to will be 0.
        """
        type_cluster = self._clusters.get_type_cluster(type_id)
        features_dict = {
            f"cluster_type_membership_{cluster}": 0.0
            for cluster in range(self._num_clusters)
        }
        features_dict[f"cluster_type_membership_{type_cluster}"] = 1.0
        return features_dict
