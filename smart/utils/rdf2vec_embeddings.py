import os

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
from smart.utils.wikidata import Wikidata

RDF2VEC_FILE = (
    'data/dbpedia/dbpedia-2016-10/embeddings/'
    'DBpediaVecotrs200_ontology_vectors.txt'
)
RDF2VEC_W2V_FILE = (
    'data/dbpedia/dbpedia-2016-10/embeddings/sg200_dbpedia_500_8_df_vectors.kv'
)
DP_WD_EQC_FILE = (
    'data/dbpedia/dbpedia-2016-10/embeddings/'
    'dbpedia_wikidata_equivalent_classes.tsv'
)


class RDF2VEC:
    def __init__(self, dataset="dbpedia") -> None:
        self._dataset = dataset
        if self._dataset == "wikidata":
            self._wikidata = Wikidata()
            self._missing = set()

    def load_dbpedia_wikidata_equivalent_classes(self):
        eq_classes = {}
        with open(DP_WD_EQC_FILE, 'r') as f:
            for line in f:
                fields = line.split('\t')
                eq_classes[fields[0]] = fields[1].replace('\n', '')
        return eq_classes

    # def load_type_similarity_features():

    def load_RDF2Vec_w2v_vectors(
        self,
        data_category,
        data_type,
        use_label_description=False,
        label_description=None,
    ):
        eq_classes = self.load_dbpedia_wikidata_equivalent_classes()
        wv = KeyedVectors.load(RDF2VEC_W2V_FILE, mmap='r')
        not_found_count = 0
        question_types_with_no = set()
        all_embeddings = []
        found = 0

        for category, types in tqdm(zip(data_category, data_type)):
            if category == 'resource':
                type_embeddings = []
                for type in types:
                    type_embedding = self.get_type_embedding(
                        type, wv, eq_classes
                    )
                    if type_embedding is not None:
                        found = found + 1
                        type_embeddings.append(type_embedding)
                        assert len(type_embedding) == 200

                np_type_embeddings = np.array(type_embeddings)
                mean_vec = np.mean(np_type_embeddings, axis=0)
                mean_vec = mean_vec.tolist()
                # if not isinstance(mean_vec, list):
                # print(types, type_embeddings)
                if found == 0 or not isinstance(mean_vec, list):
                    not_found_count = not_found_count + 1
                    question_types_with_no.update(types)
                    # If no embeddings found fill it with random values
                    mean_vec = np.random.rand(1, 200).tolist()[0]
                assert len(mean_vec) == 200
                # print(mean_vec)
                all_embeddings.append(mean_vec)

                # print(len(all_embeddings))
                # if len(all_embeddings) == 10:
                #     break
        print(not_found_count, ' have all type embeddings missing')
        numpy_embeddings = np.array(all_embeddings)
        # print(all_embeddings)
        print(numpy_embeddings.shape)
        return numpy_embeddings

    def encode_rdf2vec_embeddings(self, train_df, input_data_dir):
        train_data_category = train_df.category
        train_data_type = train_df.type
        x_train_npy_array = self.load_RDF2Vec_w2v_vectors(
            train_data_category, train_data_type
        )

        oup_train_feat_path = os.path.join(
            input_data_dir, 'X.train.finetune.rdf2vec.npy'
        )
        np.save(oup_train_feat_path, x_train_npy_array)
        return x_train_npy_array

    def get_type_embedding(self, type, wv, eq_classes):
        if self._dataset == "dbpedia":
            type_with_uri = type.replace('dbo:', 'http://dbpedia.org/ontology/')
        elif self._dataset == "wikidata":
            type_id = self._wikidata.get_type_id(type)
            if not type_id:
                self._missing.add(type)
            type_with_uri = f"http://www.wikidata.org/entity/{type_id}"
        if type_with_uri in wv.vocab:
            return wv[type_with_uri]
        elif (
            type_with_uri in eq_classes
            and eq_classes[type_with_uri] in wv.vocab
        ):
            return wv[eq_classes[type_with_uri]]

    def load_RDF2Vec_type_vectors():
        RDF2vec_dict = {}
        with open(RDF2VEC_FILE, 'r') as f:
            for lines in tqdm(f):
                tokens = lines.split(' ')
                key = tokens[0]
                # if key.startswith('<http://dbpedia.org/ontology') or
                # key.contains('dbo:') or key.contains('rdf:type'):
                vec = []
                for val in tokens[1:]:
                    vec.append(float(val))
                RDF2vec_dict[key] = np.array(vec)
        return RDF2vec_dict

    def get_RDF2Vec_vector(self, data_df):
        RDF2vec_dict = self.load_RDF2Vec_type_vectors()
        not_found = set()
        # for key, val in RDF2vec_dict.items():
        #     print(key)
        not_found_count = 0
        for index, row in data_df.iterrows():
            found = 0
            if row['category'] == 'resource':
                types = row['type']
                for type in types:
                    type = type.replace('dbo:', '<http://dbpedia.org/ontology/')
                    type = type + '>'
                    # print(type)
                    if type in RDF2vec_dict:
                        found = found + 1
                        # print(RDF2vec_dict[type])
                    else:
                        not_found.add(type)
                if found == 0:
                    print(row, 'none of the types found')
                    not_found_count = not_found_count + 1
        print(len(not_found))
        print(not_found_count, ' have all type embeddings missing')
        print(not_found)
        # break
