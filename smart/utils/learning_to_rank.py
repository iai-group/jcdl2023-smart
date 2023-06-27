from tqdm import tqdm
from typing import List


class L2RDataFormatter:
    """Generates data for learning to rank, in LibSVM format"""

    def __init__(
        self,
        qid: List[int],
        features: List[List[float]],
        label: List[int],
        doc_id: List[int],
    ):
        self.qid = qid
        self.features = features
        self.label = label
        self.docid = doc_id

    def write_libsvm_format(self, file_name: str, dense: bool = False) -> None:
        """This function writes the data in libmvm format to a given file"""
        print("Writing libsvm data!")
        with open(file_name, "w") as out_file:
            for id, feature, label, doc in tqdm(
                zip(self.qid, self.features, self.label, self.docid),
                total=len(self.qid),
            ):
                feature_str = ""
                for i in range(len(feature)):
                    if dense or feature[i] != 0:
                        feature_str = f"{feature_str}{i+1}:{feature[i]} "
                out_file.write(f"{label} qid:{id} {feature_str}#{doc}\n")
