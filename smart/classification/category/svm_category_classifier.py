"""This file is used for training a SVM model for the category prediction
for both DBPedia and Wikidata. Since the data is similar for the category
classification, both training datasets are merged and predicted for
respective test sets.


Typical usage example:
  python -m smart.classification.category.svm_category_classifier

Author: Vinay Setty
"""
from sklearn.svm import SVC
import smart.classification.category.category_classification_utils as hc

from scipy.sparse.csr import csr_matrix
import pickle
from typing import List

SUBMISSION_MODE = True


def train_svm(X_train, y_train) -> SVC:
    svc_clf = SVC(probability=True, kernel='rbf')
    svc_clf.fit(X_train, y_train)
    return svc_clf


def train_and_eval_svm_with_tfidf(
    train_data_df, dbpedia_data_dev_df, wikidata_data_dev_df, eval_model=True
) -> (float, List[float], float, List[float]):
    """Trains given training df and evaluates or predicts labels for given dev sets.

    Args:
        train_data_df: Train DF
        dbpedia_data_dev_df: DBpedia test DF
        wikidata_data_dev_df: Wikidata test DF
        eval_model: boolean, true if gold label is known false otherwise.

    Returns:
        a tuple of 4 objects:
        dbpedia_f1: f1 score for DBPedia dev set.
        dbpedia_y_pred: List of DBPedia prediction labels.
        wikidata_f1: f1 score for Wikidata dev set.
        wikidata_y_pred: List of Wikidata prediction labels.
    """

    print('Transforming text into tf-idf vectors')

    (
        train_data_df,
        data_tf_idf,
        tf_transformer,
        count_vect,
    ) = hc.get_tf_idf_vectors(train_data_df)
    train_data_df.dropna()
    print(train_data_df.head())
    print('Transforming ttf-idf vectors into sparse matrix')

    x_train: csr_matrix = csr_matrix(data_tf_idf)
    y_train = train_data_df.category
    svm_model = train_svm(x_train, y_train)
    print('evaluating SVM')
    dbpedia_dev_tfidf = hc.get_tf_idf_vectors_from_pretrained(
        dbpedia_data_dev_df.question, tf_transformer, count_vect
    )
    wikidata_dev_tfidf = hc.get_tf_idf_vectors_from_pretrained(
        wikidata_data_dev_df.question, tf_transformer, count_vect
    )
    if eval_model:
        dbpedia_f1, dbpedia_y_pred = hc.eval(
            svm_model, dbpedia_dev_tfidf, dbpedia_data_dev_df.category
        )
        wikidata_f1, wikidata_y_pred = hc.eval(
            svm_model, wikidata_dev_tfidf, wikidata_data_dev_df.category
        )

    else:
        dbpedia_y_pred = hc.eval(svm_model, dbpedia_dev_tfidf)
        wikidata_y_pred = hc.pred(svm_model, wikidata_dev_tfidf)

    return dbpedia_f1, dbpedia_y_pred, wikidata_f1, wikidata_y_pred


def main() -> None:
    if not SUBMISSION_MODE:
        cross_validate_svm()
    else:
        full_train_test_svm()


def full_train_test_svm() -> None:
    """Trains on full task data and predicts tests on the task test data"""
    (
        train_data_df,
        dbpedia_data_dev_df,
        wikidata_data_dev_df,
    ) = hc.get_full_training_test_data()
    print(train_data_df.describe)
    dbpedia_y_pred, wikidata_y_pred, model = train_and_eval_svm_with_tfidf(
        train_data_df,
        dbpedia_data_dev_df,
        wikidata_data_dev_df,
        eval_model=False,
    )
    hc.dump_predictions(
        dbpedia_data_dev_df,
        dbpedia_y_pred,
        'svm',
        'dbpedia',
        SUBMISSION_MODE,
    )
    hc.dump_predictions(
        wikidata_data_dev_df,
        wikidata_y_pred,
        'svm',
        'wikidata',
        SUBMISSION_MODE,
    )
    with open('svm_category_classifier_full_data.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cross_validate_svm():
    """Trains on pre-made train and tests on dev splits."""
    for split in range(hc.NUM_SPLITS):
        (
            train_data_df,
            dbpedia_data_dev_df,
            wikidata_data_dev_df,
        ) = hc.get_split_train_test_data(split)
        print('training svm for split ' + str(split))
        (
            dbpedia_f1,
            dbpedia_y_pred,
            wikidata_f1,
            wikidata_y_pred,
        ) = train_and_eval_svm_with_tfidf(
            train_data_df, dbpedia_data_dev_df, wikidata_data_dev_df
        )
        print('dbpedia split ' + str(split) + ': ' + str(dbpedia_f1))
        print('wikidata split ' + str(split) + ': ' + str(wikidata_f1))
        hc.dump_predictions(
            dbpedia_data_dev_df,
            dbpedia_y_pred,
            'svm',
            'dbpedia',
            SUBMISSION_MODE,
            split=split,
        )
        hc.dump_predictions(
            wikidata_data_dev_df,
            wikidata_y_pred,
            'svm',
            'wikidata',
            SUBMISSION_MODE,
            split=split,
        )


if __name__ == '__main__':
    main()
