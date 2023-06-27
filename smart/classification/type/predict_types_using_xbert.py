"""This module predicts the types for the questions based on the
predictions from the https://github.com/OctoberChang/X-Transformer model.

This module produces the predictions in json format as required by the task
in the following path: data/runs/[DBPedia/Wikidata]/
*.pifa-tfidf_xbert_predictions.json

  Typical usage example:
  python -m smart.classification.type.predict_types_using_xbert dbpedia
  pifa-tfidf xlnet

"""
import argparse
import pickle

import numpy as np
import scipy.sparse as smat
from scipy.sparse.csr import csr_matrix

import smart.utils.file_utils as file_utils
from smart.utils.dataset import Dataset
from smart.utils.tf_rank_to_smart import TFRankOutputReader

# dataset = str(sys.argv[1])
X_TRANSFORMER_FOLDER = "smart/classification/type/X-Transformer/"

SUBMISSION_MODE = True
RESULTS_FOLDER = "data/runs/"


def read_test_texts(raw_text_file):
    """Reads a text file and stores it as array of strings."""
    texts = file_utils.read_text_file(raw_text_file)
    return texts


def read_xbert_predictions(pred_file):
    """Reads the multilabel predictions from xbert into a sparse matrix."""
    print(pred_file)
    preds = smat.load_npz(pred_file)
    preds_sparse = csr_matrix(list(preds.toarray()))
    return preds_sparse


def read_multi_label_binarizer_pickle(label_file):
    """Loads a MultiLabelBinarizer pickle to reverse transform it after
    training."""
    new_mlb = pickle.load(open(label_file, "rb"))
    return new_mlb


def transform_binary_labels_to_str(binary_labels):
    """Inverse transforms the binary labels to string labels using the
    MultiLabelBinarizer which is supposed to be stored when the labels
     were binarized to preserve the mapping."""
    new_mlb = read_multi_label_binarizer_pickle()
    str_labels = new_mlb.inverse_transform(binary_labels)
    return str_labels


def get_top10_predicted_types(texts, pred_binary_labels_sparse, mlb):
    """Reads the predictions from the xbert output and decodes the top-to
    predictions to string labels."""
    top_10_types_dict = {}
    for text, preds in zip(texts, pred_binary_labels_sparse):
        pred_str_labels_sorted = []
        pred_prob = []
        pred_tuples = []
        for (row, col) in zip(*preds.nonzero()):
            val = preds[row, col]
            pred_tuples.append((col, val))
        # Sort the sparse matrix indices according to the predicted scores
        pred_tuples.sort(key=lambda x: -x[1])
        for (col, val) in pred_tuples:
            pred_bin_labels = np.zeros(pred_binary_labels_sparse.shape[1])
            pred_bin_labels[col] = 1
            pred_sparse_labels = csr_matrix([pred_bin_labels])

            pred_str_labels = mlb.inverse_transform(pred_sparse_labels)
            pred_str_labels_sorted.append(pred_str_labels[0][0])
            pred_prob.append(val.item())
        # print(text, pred_str_labels_sorted)
        zip_iterator = zip(pred_str_labels_sorted, pred_prob)
        pred_dicts = dict(zip_iterator)
        top_10_types_dict[text] = (pred_str_labels_sorted, pred_dicts)
    return top_10_types_dict


def get_tf_rank_top10_predictions(texts, new_mlb, test_file, pred_file):
    tfrank_reader = TFRankOutputReader(test_file=test_file, pred_file=pred_file)
    top_10_types_dict = {}
    num_labels = len(list(new_mlb.classes_))
    for idx, question in enumerate(texts):
        pred_str_labels = []
        pred_prob = []
        if idx not in tfrank_reader._rankings:
            continue
        for pred in tfrank_reader.fetch_topk_docs(idx):
            type_id = pred["doc_id"]
            ranked_types_sparse = csr_matrix((1, num_labels), dtype=np.int8)
            ranked_types_sparse[0, type_id] = 1
            pred_str_label = new_mlb.inverse_transform(ranked_types_sparse)
            pred_str_labels.extend(pred_str_label[0])
            pred_prob.append(pred["pred_score"])
        zip_iterator = zip(pred_str_labels, pred_prob)
        pred_dicts = dict(zip_iterator)
        top_10_types_dict[question] = (pred_str_labels, pred_dicts)
    return top_10_types_dict


def dump_predictions(
    cat_result_file,
    raw_text_file,
    pred_file,
    label_file,
    emb,
    dataset,
    use_tf_rank: bool = False,
    tf_rank_pred_file=None,
    tf_rank_test_file=None,
    use_oracle_category_classifier=False,
):
    """Parses top-10 predictions and dumps it in the results json."""
    texts = read_test_texts(raw_text_file)
    pred_binary_labels_sparse = read_xbert_predictions(pred_file)
    new_mlb = read_multi_label_binarizer_pickle(label_file)
    if use_tf_rank:
        predicted_types_dict = get_tf_rank_top10_predictions(
            texts,
            new_mlb,
            pred_file=tf_rank_pred_file,
            test_file=tf_rank_test_file,
        )
    else:
        predicted_types_dict = get_top10_predicted_types(
            texts, pred_binary_labels_sparse, new_mlb
        )
    # cat_predictions = file_utils.read_json(cat_result_file)
    # If oracle is used for category classifier use cat_pred split other use
    # test split which has the gold resource types
    if use_oracle_category_classifier:
        data_split = 'test'
    else:
        data_split = "cat_pred"
    cat_predictions = Dataset(dataset, data_split).get_df()
    submission_ids = []
    submission_category = []
    submission_type = []
    submission_question = []
    category_pred_prob = []
    pred_type_prob = []
    for index, row in cat_predictions.iterrows():
        submission_ids.append(row["id"])
        submission_category.append(row["category"])
        question = row["question"].rstrip("\n")
        question = question.replace("\n", "").strip()
        submission_question.append(question)
        if row["category"] == "resource":
            if question not in predicted_types_dict:
                top_predictions = []
                top_predictions_prob = {}
            else:
                top_predictions, top_predictions_prob = predicted_types_dict[
                    question
                ]
            submission_type.append(top_predictions)
            pred_type_prob.append(top_predictions_prob)
            # print(submission_type)
            # break
        else:
            submission_type.append(row["type"])
            pred_type_prob.append({})
        if "category_pred_prob" in row:
            category_pred_prob.append(row["category_pred_prob"])
            # cat_predictions.at[index, 'predicted_type'] = top_predictions
    print(len(pred_type_prob))
    cat_predictions["type"] = submission_type
    cat_predictions["type_pred_prob"] = pred_type_prob

    print(len(submission_ids), len(submission_category), len(submission_type))
    print(cat_predictions.head())
    file_utils.dump_json(cat_predictions, cat_result_file)


def consolidate_results(dataset):
    """Merges the predictions from all splits.

    Args:
        dataset: String which indicates which dataset to process.
    """
    all_results = []
    category_classifier = "bert"
    for split in range(5):
        print(category_classifier, str(split), dataset)
        result_file = (
            RESULTS_FOLDER
            + "/"
            + dataset
            + "/"
            + category_classifier
            + "/"
            + category_classifier
            + "_results_"
            + dataset
            + "_dev_split"
            + str(split)
            + ".json_xbert_submission.json"
        )
        filename = result_file
        print(filename)
        df = file_utils.read_json(filename)
        print(df.shape)
        #         print(df.head)
        all_results.append(df)
    all_results_df = all_results[0]
    for df in all_results[1:]:
        all_results_df = all_results_df.append(df, ignore_index=True)
    print(all_results_df.shape)
    combined_file = (
        RESULTS_FOLDER
        + "/"
        + dataset
        + "/"
        + category_classifier
        + "/"
        + "/combined.json"
    )
    print(combined_file)
    file_utils.dump_json(all_results_df, combined_file)


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
        "-t",
        "--tfrank",
        type=bool,
        nargs="?",
        default=False,
        const=True,
        help="Use TF-Rank to select top-10 types.",
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
        "--clusters",
        type=int,
        nargs="?",
        default=64,
        const=1,
        help="Number of clusters generated using clustering algo.",
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
        action='store_false',
        help='Sort the clusters by their prediction store. Default = True',
    )

    parser.add_argument(
        "-oracle",
        '--use_oracle_for_cat_classifier',
        action='store_true',
        help='Sort the clusters by their prediction store. Default = False',
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
        "-tl",
        "--tfrank_loss",
        type=str,
        help="TF-Rank loss function",
        default="approx_ndcg_loss",
        const=1,
        nargs="?",
    )
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    """Depending on submission mode SUBMISSION_MODE either trains and
    evaluates  with 5-fold cross-validation or trains on the whole data and
    tests on the test split.
    """

    args = arg_parser()
    print(args)
    dataset = args.dataset
    """TODO: dataset name is inconsistent in xbert and in our dataset make
    it consistent and remove .tolower()"""
    dataset_lower = dataset.lower()
    emb = args.embedding
    xbert_model_name = ""
    if args.model == "roberta":
        xbert_model_name = "roberta-large"
    if args.model == "xlnet":
        xbert_model_name = "xlnet-large-cased"
    if args.model == "bert":
        xbert_model_name = "bert-large-cased"

    if not SUBMISSION_MODE:
        XBERT_PRED_FOLDER = str(
            X_TRANSFORMER_FOLDER + "save_models/" + dataset_lower
        )
        XBERT_DATA_FOLDER = X_TRANSFORMER_FOLDER + "datasets/" + dataset_lower
        for split in range(5):
            # naming convenction is fubar svm_results_dbpedia_dev_split0
            # File names needs to be updated
            category_classifier = "bert"
            result_file = (
                RESULTS_FOLDER
                + "/"
                + dataset
                + "/"
                + category_classifier
                + "/"
                + category_classifier
                + "_results_"
                + dataset
                + "_dev_split"
                + str(split)
                + ".json"
            )
            print(result_file)
            raw_text_file = (
                XBERT_DATA_FOLDER
                + "_split"
                + str(split)
                + "/test_raw_texts.txt"
            )
            pred_file = "{}/test.pred.npz".format(
                XBERT_PRED_FOLDER
                + "_split"
                + str(split)
                + "/"
                + emb
                + "-s0/ranker/roberta-large/"
            )
            label_file = XBERT_DATA_FOLDER + "_split" + str(split) + "/mlb.pkl"
            dump_predictions(
                result_file, raw_text_file, pred_file, label_file, emb, dataset
            )
        consolidate_results(dataset)
    else:
        XBERT_PRED_FOLDER = (
            f'{X_TRANSFORMER_FOLDER}/save_models/{dataset_lower}_full'
        )
        print(XBERT_PRED_FOLDER)
        XBERT_DATA_FOLDER = (
            f'{X_TRANSFORMER_FOLDER}/datasets/{dataset_lower}_full'
        )
        raw_text_file = f'{XBERT_DATA_FOLDER}/test_raw_texts.txt'
        ranker_file = f'{XBERT_PRED_FOLDER}/{emb}-s0/ranker/'
        pred_file = f'{ranker_file}/{xbert_model_name}/test.pred.npz'
        label_file = f'{XBERT_DATA_FOLDER}/mlb.pkl'
        result_file = (
            f'{RESULTS_FOLDER}/{dataset}/{args.model}_{emb}_oracle_'
            f'{args.use_oracle_for_cat_classifier}_clean_{args.use_clean}'
            f'_{args.clusters}_test_pred.json'
        )
        tf_rank_pred_file = None
        tf_rank_test_file = None
        if args.tfrank:
            tf_rank_data_folder = "ranking/data/"
            ng = args.tfrank_ng
            c = args.tfrank_nc
            if args.sort_clusters_for_ng:
                order = "sorted"
            else:
                order = "random"
            loss = args.tfrank_loss
            result_file = (
                f'{RESULTS_FOLDER}/{dataset}/{xbert_model_name}_{emb}'
                f'_ng{ng}_nc{c}_{loss}_{order}_tfrank_test_pred.json'
            )
            # tf_rank_pred_file = (
            #     f'{tf_rank_data_folder}/{dataset}_{emb}_'
            #     f'{xbert_model_name}_ng{ng}_nc{c}_{order}_{loss}_preds.txt'
            # )
            # )

            tf_rank_test_file = (
                f'{tf_rank_data_folder}/{dataset}_{emb}_{xbert_model_name}_test'
                '_use_sparse_False_use_dense_True_ng{ng}_nc{c}_{order}_l2r.txt'
            )
            tf_rank_pred_file = f"{tf_rank_test_file}_pred_scores.txt"
            print(tf_rank_pred_file, tf_rank_test_file)
        print(result_file)
        print(
            "Using oracle use_oracle_for_cat_classifier ",
            args.use_oracle_for_cat_classifier,
        )
        dump_predictions(
            result_file,
            raw_text_file,
            pred_file,
            label_file,
            emb,
            dataset,
            use_tf_rank=args.tfrank,
            tf_rank_pred_file=tf_rank_pred_file,
            tf_rank_test_file=tf_rank_test_file,
            use_oracle_category_classifier=args.use_oracle_for_cat_classifier,
        )
