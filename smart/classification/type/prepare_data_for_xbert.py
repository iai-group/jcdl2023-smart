"""This module prepares the data for the X-bert classifier as specified in
https://github.com/OctoberChang/X-Transformer.

This module write the following files:
X.train.npz: the instance TF-IDF feature matrix for the train set. The data type
    is scipy.sparse.csr_matrix of size (N_train, D_tfidf), where N_train is the
    number of train instances and D_tfidf is the number of features.
X.test.npz: the instance TF-IDF feature matrix for the test set. The data type
    is scipy.sparse.csr_matrix of size (N_test, D_tfidf), where N_test is the
    number of test instances and D_tfidf is the number of features.
Y.train.npz: the instance-to-label matrix for the train set. The data type is
    scipy.sparse.csr_matrix of size (N_train, L), where n_train is the number of
    train instances and L is the number of labels.
Y.test.npz: the instance-to-label matrix for the test set. The data type is
    scipy.sparse.csr_matrix of size (N_test, L), where n_test is the number of
    test instances and L is the number of labels.
train_raw_texts.txt: The raw text of the train set.
test_raw_texts.txt: The raw text of the test set.
label_map.txt: the label's text description.
mlb.pkl: A pickle file of MultiLabelBinarizer object to inverse transform the
    binarized labels


  Typical usage example:
  python -m smart.classification.type.prepare_data_for_xbert dbpedia

"""

import argparse
import copy
import os
import pickle
from typing import Dict

import numpy as np
import scipy.sparse as smat
from data.smart_dataset.dbpedia.evaluation.evaluate import load_type_hierarchy
from scipy.sparse import save_npz
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from smart.utils.dataset import Dataset
from smart.utils.rdf2vec_embeddings import RDF2VEC
from smart.utils.type_description import TypeDescription

SUBMISSION_MODE = True

SMART_TASK_DATA_FOLDER = "data/smart_dataset/"

NUM_SPLITS = 5

XBERT_FOLDER = "smart/classification/type/X-Transformer/"
XBERT_DATA_FOLDER = XBERT_FOLDER + "datasets/"
XBERT_PROC_FOLDER = XBERT_FOLDER + "save_models/dbpedia_full/proc_data/"


TYPE_FEATURES_FILE = "type_features/T_TypeSimEntityFeatures.tsv"

USE_LABEL_DESCRIPTION = True


def get_type_features_dict(dataset_str: str) -> Dict:
    """Reads the T_TypeSimEntityFeatures.tsv file into a dict.
    Args:

    Returns:
        A dictionary with types as keys and their features as values in float
        type.
    """
    type_features_dict = {}
    first_line = True
    with open(f"data/{dataset_str}/{TYPE_FEATURES_FILE}") as in_file:
        for line in in_file:
            if first_line:
                first_line = False
                continue
            tokens = line.split("\t")
            type_string = tokens[0]
            features = np.array(tokens[1:])
            type_features_dict[type_string] = features.astype(np.float)
    return type_features_dict


def dump_type_features_for_xbert(
    unique_labels, type_features_dict, dataset_str
):
    """Write type features for the clustering, it should be written in to the
       XBERT_PROC_FOLDER.

    Args:
        out_data_folder: Location where to store the type embeddings for xbert
        unique_labels: List of type labels in the specific order it is
                        written in the data.
        type_features_dict:  Type features dict from get_type_features_dict
    """
    type_features = []
    print(len(type_features_dict))
    count = 0
    for label in unique_labels:
        if label in type_features_dict:
            features = type_features_dict[label].tolist()
        else:
            print(label)
            count += 1
            features = np.zeros(len(type_features_dict) + 1).tolist()
        # print(features)
        type_features.append(features)
    print(f"{count} types have no features")
    # print(type_features)
    np_type_features = np.array(type_features)
    #     np_type_features.reshape(len(type_features), len(type_features))
    print(np_type_features.shape)
    #     print(np_type_features)
    #     row, col = np.where(np_type_features!=0)
    #     values = np.array([np_type_features[i][j] for i,j in zip(row, col)])
    np_type_features_sparse = smat.csr_matrix(np_type_features)
    label_embedding = normalize(np_type_features_sparse, axis=1, norm="l2")
    xbert_proc_folder = (
        f"{XBERT_FOLDER}/save_models/{dataset_str}_full/proc_data/"
    )
    if not os.path.exists(xbert_proc_folder):
        os.makedirs(xbert_proc_folder)
    smat.save_npz(xbert_proc_folder + "L.type-features.npz", label_embedding)


def binarize_labels(data_df, dataset):
    """Converts types into one-hot encoding and adds it to the labels column.

    Args:
        Dataframe containing a 'type' column which needs to be binarized.

    Returns:
        A tuple of a Dataframe with binary labels in 'labels' column and a
        list of unique, classes and the MultiLabelBinarizer, object to be
        pickled for later reverse transform.
    """
    mlb = MultiLabelBinarizer()
    binary_labels_matrix = mlb.fit_transform(data_df["type"])
    data_df["labels"] = list(binary_labels_matrix)
    data_df["ranked_labels"] = copy.deepcopy(list(binary_labels_matrix))
    unique_types = list(mlb.classes_)
    print("unique_types", len(unique_types))
    data_df = add_label_ranking(data_df, unique_types, dataset)
    return data_df, unique_types, mlb


def get_type_hierarchy():
    type_hierarchy_file = (
        "data/smart_dataset/dbpedia/evaluation/dbpedia_types.tsv"
    )
    return load_type_hierarchy(type_hierarchy_file)


def add_label_ranking(data_df, unique_types, dataset: str):
    type_heirarchy, max_depth = get_type_hierarchy()
    for _, row in data_df.iterrows():
        types = row["type"]
        ranked_labels = row["ranked_labels"]
        num_labels = len(types)
        for i, type_label_id in enumerate(types):
            if dataset == "dbpedia":
                rank = type_heirarchy[type_label_id]["depth"]
            elif dataset == "wikidata":
                rank = num_labels - i
            ranked_labels[unique_types.index(type_label_id)] = rank
        row["ranked_labels"] = ranked_labels
    return data_df


def get_tf_idf_vectors(clean_df):
    """Computes tf-idf vectors for the questions field in the data.
    Args:
        clean_df: Dataframe from which 'question' column will be vectorized.

    Returns:
        A 2d-array of tf-idf vectors corresponding to the input text.
    """
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(clean_df["question"])
    tf_transformer = TfidfTransformer(use_idf=True)
    x_train_tf_idf = (
        tf_transformer.fit_transform(x_train_counts).todense().tolist()
    )
    clean_df["TFIDF"] = x_train_tf_idf
    return clean_df, x_train_tf_idf, tf_transformer, count_vect


def get_tf_idf_vectors_from_pretrained(text, tf_transformer, count_vect):
    """Applies the pre-trained tf-idf transformer.

    Args:
        text: text to be converted.
        tf_transformer: Pretrained tf-idf transformer from get_tf_idf_vectors.
                    This is to ensure the IDF is applied from the training data
        count_vect: count vector from get_tf_idf_vectors.

    Returns:
        A 2d-array of tf-idf vectors corresponding to the input text.
    """
    counts = count_vect.transform(np.array(text))
    tf_idf_vector = tf_transformer.transform(counts)
    return tf_idf_vector


def dump_data_id_map(data_df, xbert_data_folder, split):
    """Writes the id of the questions to the order of the data mapping in a
    file. This is needed because the mapping between the question and data in
    xbert_data folder is not stored anywhere"""

    map_file = xbert_data_folder + "/" + split + "_id_map.txt"
    row_num = 0
    with open(map_file, "w") as out_file:
        for index, row in data_df.iterrows():
            q_id = row["id"]
            out_file.write(str(q_id) + "\t" + str(row_num) + "\n")
            row_num = row_num + 1


def prepare_data(
    out_data_folder,
    types_description,
    dataset_str,
    write_test_labels=False,
    use_sample=False,
    use_clean_data=False,
):
    """Prepare the data for XBERT. Creates TF-IDF features for the training
    data and binarizes the labels and write everything in npz format. Also
    write raw texts of the questions and type label descriptions.

    Args:
        out_data_folder: X-bert path to write all the output files to it is
                        under smart/classification/type/X-Transformers/datasets/
                        [dbpedia/wikidata][_full].
        types_description: TypeDescription object to get the textual
                            descriptions of the types.
        dataset_str: DBpedia or Wikidata
        write_test_labels: A boolean flag indicating if the test data
                        contains labels and should be written.
        use_sample: use sample data for testing
        use_clean_data: use cleaned data with typos and errors removed
    Returns:
        None
    """
    test_data_df, train_data_df = read_preprocess_data(
        dataset_str, use_sample, use_clean_data
    )

    print(train_data_df.shape, test_data_df.shape)
    print("generating tfidf features on the training data")
    test_tf_idf, train_tf_idf = get_tf_idf_features(test_data_df, train_data_df)

    # If the data folder doesnt exist create it!
    if not os.path.exists(out_data_folder):
        os.makedirs(out_data_folder)

    dump_data_id_map(train_data_df, out_data_folder, "train")
    dump_data_id_map(test_data_df, out_data_folder, "test")

    print("done generating tfidf features")

    train_data_df["split"] = "train"
    test_data_df["split"] = "test"

    all_df = train_data_df
    all_df = all_df.append(test_data_df)
    full_data_df, unique_labels = binarize_write_labels(
        out_data_folder, all_df, dataset_str
    )
    print(full_data_df.head())
    print('num labels', len(unique_labels))
    train_data_df = full_data_df.loc[full_data_df["split"] == "train"]
    test_data_df = full_data_df.loc[full_data_df["split"] == "test"]

    print("data is split into train_test")
    dump_features_labels_for_xbert(
        out_data_folder,
        test_data_df,
        test_tf_idf,
        train_data_df,
        train_tf_idf,
        write_test_labels,
    )
    dump_raw_texts_for_xbert(
        out_data_folder,
        train_data_df,
        test_data_df,
        types_description,
        unique_labels,
    )
    type_features_dict = get_type_features_dict(dataset_str)
    dump_type_features_for_xbert(unique_labels, type_features_dict, dataset_str)
    RDF2VEC(dataset=dataset_str).encode_rdf2vec_embeddings(
        train_data_df, input_data_dir=out_data_folder
    )
    return test_data_df, train_data_df


def read_preprocess_data(dataset_str, use_sample=False, use_clean_data=False):
    """Read train and test json files and filters to resource category and
    removes n/a questions.

    Args:
       dataset_str: String specifiying which dataset to read.

    Returns:
        test_data_df: Test dataframe.
        train_data_df: Train dataframe.
    """
    # Get the train and test splits
    train_split = "train"
    test_split = "test"
    if use_sample:
        train_split = "train_sample"
        test_split = "test_sample"
    elif use_clean_data:
        print("Using clean data")
        train_split = "train_clean_grammar"
        test_split = "test_clean_grammar"
    train_data_df = Dataset(dataset_str, train_split).get_df()
    test_data_df = Dataset(dataset_str, test_split).get_df()

    # We only use resource category
    train_data_df = train_data_df.loc[train_data_df["category"] == "resource"]
    test_data_df = test_data_df.loc[test_data_df["category"] == "resource"]
    # Use short abstracts as questions for now, if abstract column exists
    train_data_df.rename(columns={"abstract": "question"}, inplace=True)
    test_data_df.rename(columns={"abstract": "question"}, inplace=True)

    return test_data_df, train_data_df


def dump_raw_texts_for_xbert(
    out_data_folder,
    train_data_df,
    test_data_df,
    types_description,
    unique_labels,
):
    """Writes the raw text necessary for xbert training.

    Args:
        out_data_folder: Path where the data to be written.
        train_data_df: train dataframe.
        test_raw_text: test dataframe.
        types_description: type description dictionary.
        unique_labels: Set of unique labels in the same order as the
        binarizer encoding.
    """
    with open(
        out_data_folder + "/" + "train_raw_texts.txt", "w"
    ) as train_txt_file:
        for _, row in train_data_df.iterrows():
            train_txt_file.write(row["question"] + "\n")
    with open(
        out_data_folder + "/" + "test_raw_texts.txt", "w"
    ) as test_txt_file:
        for _, row in test_data_df.iterrows():
            test_txt_file.write(row["question"] + "\n")
    with open(out_data_folder + "/" + "label_map.txt", "w") as label_map_file:
        for label in unique_labels:
            label_str = label.replace("dbo:", "")
            if USE_LABEL_DESCRIPTION:
                type_comment = types_description.get_type_description(label)
                # Adding a delimiter ":" between the label name and description
                if type_comment:
                    label_str = label_str + " : " + type_comment
            label_map_file.write(label_str.replace("\n", "").strip() + "\n")


def dump_features_labels_for_xbert(
    out_data_folder,
    test_data_df,
    test_tf_idf,
    train_df,
    train_tf_idf,
    write_test_labels=False,
):
    """Write the features in npz format.

    Args:
        out_data_folder: Folder to write to.
        test_label: dataframe column of test labels.
        test_tf_idf: dataframe column of test  features.
        train_label: dataframe column of train labels.
        train_tf_idf: dataframe column of train features.
        write_test_labels: Flag to write test labels or not.
    """
    save_npz(f"{out_data_folder}/X.train.npz", csr_matrix(train_tf_idf))
    save_npz(f"{out_data_folder}/X.test.npz", csr_matrix(test_tf_idf))
    train_labels = csr_matrix(np.vstack(train_df.labels))
    print(train_labels[0].nonzero()[1])
    train_ranked_labels = csr_matrix(np.vstack(train_df.ranked_labels))
    save_npz(f"{out_data_folder}/Y.train.npz", train_labels)
    save_npz(f"{out_data_folder}/Y_ranked.train.npz", train_ranked_labels)
    if write_test_labels:
        test_labels = csr_matrix(np.vstack(test_data_df.labels))
        test_ranked_labels = csr_matrix(np.vstack(test_data_df.ranked_labels))
        save_npz(f"{out_data_folder}/Y.test.npz", test_labels)
        save_npz(f"{out_data_folder}/Y_ranked.test.npz", test_ranked_labels)


def binarize_write_labels(out_data_folder, train_data_df, dataset):
    """Binarize the types and writes them to out_data_folder.

    Args:
        out_data_folder: X-bert folder.
        train_data_df: dataframe containing 'type' column.

    Returns:

    """
    full_data_df = train_data_df
    print("generating one hot labels")
    full_data_df, unique_labels, mlb = binarize_labels(full_data_df, dataset)
    pickle.dump(mlb, open(out_data_folder + "/" + "mlb.pkl", "wb"))
    print("done generating one hot labels")
    return full_data_df, unique_labels


def get_tf_idf_features(test_data_df, train_data_df):
    """Compute tf-idf features for a given dataframe with 'question' column.

    Args:
        test_data_df: test dataframe.
        train_data_df: train dataframe.

    Returns:
        test_tf_idf: test tf-idf array.
        train_data_df: train dataframe.
        train_tf_idf: train tf-idf array.
    """
    (
        train_data_df,
        train_tf_idf,
        tf_transformer,
        count_vect,
    ) = get_tf_idf_vectors(train_data_df)
    test_tf_idf = get_tf_idf_vectors_from_pretrained(
        test_data_df.question, tf_transformer, count_vect
    )
    return test_tf_idf, train_tf_idf


def prepare_xbert_data(
    dataset,
    use_sample=False,
    use_clean_data=False,
    use_entity_descriptions=False,
):
    """Meta function to prepare the data."""
    print(dataset)
    type_description = TypeDescription(
        dataset, use_entity_descriptions=use_entity_descriptions
    )

    if not os.path.exists(XBERT_DATA_FOLDER + dataset + "_full/"):
        os.makedirs(XBERT_DATA_FOLDER + dataset + "_full/")

    if not os.path.exists(XBERT_PROC_FOLDER):
        os.makedirs(XBERT_PROC_FOLDER)

    # If doing 5-fold cross validation set the SUBMISSION_MODE to False
    if not SUBMISSION_MODE:
        for split in range(NUM_SPLITS):
            return prepare_data(
                XBERT_DATA_FOLDER + dataset + "_split" + str(split),
                split,
                type_description,
                dataset,
                True,
                use_sample,
                use_clean_data,
            )
    else:
        return prepare_data(
            XBERT_DATA_FOLDER + dataset + "_full/",
            type_description,
            dataset,
            True,
            use_sample,
            use_clean_data,
        )


def arg_parser():
    """Function to parse command line arguments.

    Returns: ArgumentParser object with parsed arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="specify either dbpedia, dbpedia_summaries or wikidata. \
        It can be a list too",
        default="dbpedia",
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--use_clean",
        type=str,
        help="specify whether to use clean data",
        default=False,
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--sample",
        type=str,
        help="specify whether to use sample data",
        default=False,
        const=1,
        nargs="?",
    )
    parser.add_argument(
        "--use_entity_descriptions",
        type=str,
        help="specify whether to use entity descriptions in addition to \
        type descriptions",
        default=False,
        const=1,
        nargs="?",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    print(args)
    dataset = args.dataset
    prepare_xbert_data(
        dataset,
        use_sample=args.sample,
        use_clean_data=args.use_clean,
        use_entity_descriptions=args.use_entity_descriptions,
    )
