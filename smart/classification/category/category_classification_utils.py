"""Utility file for category classification.
Needed for both BERT and SVM classifiers.

Author: Vinay Setty
"""


from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    make_scorer,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from smart.utils.dataset import Dataset
import smart.utils.file_utils as fu

SMARTTASK_DATA_FOLDER = 'data/smart_dataset'
NUM_SPLITS = 5
RESULTS_FOLDER = 'data/runs/'


# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
# predictions, raw_outputs = model.predict(["Some arbitary sentence"])


def eval(model, tf_idf_vector, y_test):
    y_pred = model.predict(tf_idf_vector)
    eval_score = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    print(eval_score)
    report = classification_report(y_test, y_pred)
    return report, y_pred


def pred(model, tf_idf_vector):
    y_pred = model.predict(tf_idf_vector)
    return y_pred


def cross_valide(model, data_df):
    clf = make_pipeline(TfidfVectorizer(), model)
    scorer = make_scorer(f1_score, average='weighted')
    scores = cross_validate(
        clf,
        data_df.question,
        data_df.category,
        scoring=scorer,
        cv=5,
        return_train_score=True,
    )
    return scorer, scores


def print_category_stats(data_df):
    print(data_df.category.unique())
    print(data_df.groupby('category').count())


def get_split_train_test_data(split):
    SMARTTASK_DBPEDIA_TRAIN_JSON = (
        SMARTTASK_DATA_FOLDER
        + 'dbpedia/splits/smarttask_dbpedia_train_split'
        + str(split)
        + '.json'
    )
    SMARTTASK_DBPEDIA_DEV_JSON = (
        SMARTTASK_DATA_FOLDER
        + 'dbpedia/splits/smarttask_dbpedia_dev_split'
        + str(split)
        + '.json'
    )
    SMARTTASK_WIKIDATA_TRAIN_JSON = (
        SMARTTASK_DATA_FOLDER
        + 'wikidata/splits/lcquad2_anstype_wikidata_train_split'
        + str(split)
        + '.json'
    )
    SMARTTASK_WIKIDATA_DEV_JSON = (
        SMARTTASK_DATA_FOLDER
        + 'wikidata/splits/lcquad2_anstype_wikidata_dev_split'
        + str(split)
        + '.json'
    )
    print(SMARTTASK_WIKIDATA_DEV_JSON)
    # Read train and dev files for wikidata and dbpedia
    dbpedia_data_train_df = fu.read_json(SMARTTASK_DBPEDIA_TRAIN_JSON)
    dbpedia_data_dev_df = fu.read_json(SMARTTASK_DBPEDIA_DEV_JSON)

    wikidata_data_train_df = fu.read_json(SMARTTASK_WIKIDATA_TRAIN_JSON)
    wikidata_data_dev_df = fu.read_json(SMARTTASK_WIKIDATA_DEV_JSON)

    train_data_df = dbpedia_data_train_df.append(
        wikidata_data_train_df, ignore_index=True
    )
    train_data_df.dropna()
    # train_data_df = dbpedia_train_data_df
    train_data_df = flatten_literal_category(train_data_df)
    dbpedia_data_dev_df = flatten_literal_category(dbpedia_data_dev_df)
    wikidata_data_dev_df = flatten_literal_category(wikidata_data_dev_df)
    return train_data_df, dbpedia_data_dev_df, wikidata_data_dev_df


def flatten_literal_category(clean_df):
    """Flattens three sub categories of literal category"""
    for i, row in clean_df.iterrows():
        old_cat = row['category']
        if old_cat == 'literal':
            new_cat = old_cat + '_' + str(row['type'][0])
            clean_df.at[i, 'category'] = new_cat
    return clean_df


def de_flatten_literal(df, column_prefix):
    """deflattens the flattened categories to write the results"""
    for i, row in df.iterrows():
        old_cat = row[column_prefix + 'category']
        new_cat = old_cat
        new_type = [row[column_prefix + 'type']]
        if old_cat == "literal_string":
            new_cat = 'literal'
            new_type = ['string']
        if old_cat == "literal_number":
            new_cat = 'literal'
            new_type = ['number']
        if old_cat == "literal_date":
            new_cat = 'literal'
            new_type = ['date']
        if new_cat == 'resource' and column_prefix != '':
            new_type = []
        df.at[i, column_prefix + 'category'] = new_cat
        df.at[i, column_prefix + 'type'] = new_type
    return df


def get_full_training_test_data():
    dbpedia_data_train_df = Dataset('dbpedia', 'train_clean_grammar').get_df()
    wikidata_data_train_df = Dataset('wikidata', 'train').get_df()
    # drop nulls,
    train_data_df = dbpedia_data_train_df.append(
        wikidata_data_train_df, ignore_index=True
    )
    train_data_df.dropna()

    dbpedia_data_test_df = Dataset('dbpedia', 'test_clean_grammar').get_df()
    wikidata_data_test_df = Dataset('wikidata', 'test').get_df()
    train_data_df = flatten_literal_category(train_data_df)
    return train_data_df, dbpedia_data_test_df, wikidata_data_test_df


def dump_predictions(
    df,
    y_pred,
    model_folder,
    model_name,
    dataset,
    SUBMISSION_MODE=False,
    split=-1,
):

    df['category'] = y_pred
    df['type'] = y_pred

    de_flatten_literal(df, '')

    if not SUBMISSION_MODE:
        if split == -1:
            print(
                "If it is not a submission mode split number must be specified"
            )
            return
        PRED_FILE = (
            RESULTS_FOLDER
            + '/'
            + dataset
            + '/'
            + model_name
            + '_results_'
            + dataset
            + '_dev_split'
            + str(split)
            + '.json'
        )
    else:
        PRED_FILE = (
            RESULTS_FOLDER
            + '/'
            + dataset
            + '/'
            + model_name
            + '_results_'
            + dataset
            + '_task_test.json'
        )

    fu.dump_json(df, PRED_FILE)
