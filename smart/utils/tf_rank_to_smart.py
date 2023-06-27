from typing import List, Dict, Any


class TFRankOutputReader:
    def __init__(
        self,
        test_file: str = "ranking/data/dbpedia_type-features_roberta-large_test"
        "_dense_False_ng3_c3_l2r.txt",
        pred_file: str = "ranking/data/dbpedia_type-features_roberta-large_ng3_"
        "c3_approx_ndcg_loss_preds.txt",
    ) -> None:
        self._tf_rank_test_file = test_file
        self._tf_rank_pred_file = pred_file
        self._rankings: Dict[Any, Any] = {}
        self._read_tf_rank_files()

    def _read_tf_rank_files(self) -> None:
        with open(self._tf_rank_test_file) as feature_file, open(
            self._tf_rank_pred_file
        ) as pred_file:
            for features, pred in zip(feature_file, pred_file):
                pred = float(pred.replace("\n", ""))
                features_tokens = features.split(" ")
                qid = int(features_tokens[1].split(":")[1])
                label = int(features_tokens[0])
                doc_id = int(
                    features_tokens[-1].replace("#", "").replace("\n", "")
                )
                ranking = {
                    "doc_id": doc_id,
                    "label": label,
                    "pred_score": pred,
                }
                if qid in self._rankings:
                    self._rankings.get(qid).append(ranking)
                else:
                    self._rankings[qid] = [ranking]

    def fetch_topk_docs(self, qid: int, k: int = 10) -> List[Dict[str, Any]]:
        """Fetches the top k docs based on their score.
            If k > len(self._rankings[qid]), the slicing automatically
            returns all elements in the list in sorted order.
            Returns an empty array if there are no documents added to the
            ranker.
        Args:
            k: Number of docs to fetch.
        Returns:
            Ordered list of doc_id, score tuples.
        """
        return sorted(
            self._rankings[qid],
            key=lambda x: x["pred_score"],
            reverse=True,
        )[:k]


if __name__ == "__main__":
    tfr = TFRankOutputReader()
    print(tfr.fetch_topk_docs(0))
    print(tfr.fetch_topk_docs(1))
    print(tfr.fetch_topk_docs(2))
    print(tfr.fetch_topk_docs(3))
