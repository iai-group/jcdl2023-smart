import numpy as np
from smart.utils.rdf2vec_embeddings import RDF2VEC
from smart.utils.dataset import Dataset


def test_rdf2vec_train():
    data_df = Dataset("dbpedia", "train").get_df()
    RDF2VEC().encode_rdf2vec_embeddings(data_df, input_data_dir="./")
    with open('./X.train.finetune.rdf2vec.npy', 'rb') as f:
        v = np.load(f)
        print(v.shape)
        assert v.shape[0] == data_df[data_df["category"] == "resource"].shape[0]
        assert v.shape[1] == 200
