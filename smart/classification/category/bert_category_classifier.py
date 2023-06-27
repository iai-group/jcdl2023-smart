"""This file is used for training a BERT model for the category prediction
for both DBPedia and Wikidata. Since the data is similar for the category
classification, both training datasets are merged and predicted for
respective test sets.


Typical usage example:
  python -m smart.classification.category.bert_category_classifier

Author: Vinay Setty
"""
from typing import Tuple

from simpletransformers.classification import ClassificationModel
import smart.classification.category.category_classification_utils as hc
import os
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,5"

TEST_MODE = True
MODEL_NAME = 'roberta-large'
ARCH = 'roberta'
N_EPOCHS = 20


def eval_bert(
    model, X_test_text, y_test, label_mapping
) -> Tuple[str, List[float], List[str]]:
    """Evaluates the bert model using the test data X_test_text with gold
    labels y_test.

    Args:
        model: Trained BERT model to use for evaluating.
        X_test_text: A list of test text questions to evaluate on.
        y_test: List of test labels.
        label_mapping: Mapping of int labels to string labels.

    Returns:
        report: evaluation report from simple transformers.
        eval_score: evaluation score from simple transformers.
        y_pred: Predicted labels for the data in X_test_test.
    """

    eval_df = get_st_data_with_labels(X_test_text, y_test)
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    pred_ind_labels = np.argmax(model_outputs, axis=1)
    y_pred = [label_mapping[idx] for idx in pred_ind_labels]
    # print(y_pred)
    report = classification_report(y_test, y_pred)
    eval_score = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    return report, eval_score, y_pred


def pred_bert(
    model, x_test_text, label_mapping
) -> Tuple[List[str], List[List[float]]]:
    """Predicts using the bert model when the gold labels are unknown.

    Args:
        model: Trained BERT model to use for evaluating.
        x_test_text: A list of test text questions to evaluate on.
        label_mapping: Mapping of int labels to string labels.

    Returns:
        y_pred: A python list of the predictions (0 or 1) for each text.
        raw_outputs: A python list of the raw model outputs for each text.
    """
    predictions, raw_outputs = model.predict(x_test_text)
    y_pred = [label_mapping[idx] for idx in predictions]
    return y_pred, raw_outputs


def train_bert(train_df, num_labels) -> ClassificationModel:
    """Trains a BERT model for the data in train_df Dataframe.

    Args:
        train_df: Training data as a pandas Dataframe.
        num_labels: Number of labels integer.

    Returns:
        model: Trained BERT model.
    """
    print(train_df.head())
    model_dir = './' + MODEL_NAME + '_category_epcohs_' + str(N_EPOCHS) + '/'
    model = ClassificationModel(
        ARCH,
        MODEL_NAME,
        num_labels=num_labels,
        args={
            'reprocess_input_data': True,
            'num_train_epochs': N_EPOCHS,
            'train_batch_size': 64,
            'gradient_accumulation_steps': 2,
            'weight': [2, 1, 1, 2, 10],
            'learning_rate': 3e-6,
            'save_eval_checkpoints': False,
            'save_model_every_epoch': False,
            'show_running_loss': True,
            'overwrite_output_dir': True,
            'best_model_dir': model_dir + '/best_model',
            'n_gpu': 2,
            'output_dir': model_dir,
        },
    )
    model.train_model(train_df)
    return model


def get_st_data_with_labels(
    text, labels
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Converts text data to Dataframe suitable for simpletransformers.

    Args:
        text: List of text to be classified.
        labels: Labels for the catgories.

    Returns:
        df: Dataframe in simpletransformers format.
        mappping: A dict of int to string label.

    """
    train_data = {'text': text, 'labels': labels}
    df = pd.DataFrame(train_data)
    df.labels, mapping = pd.factorize(df.labels, sort=True)
    return df, mapping


def train_and_test_with_bert() -> None:
    """Runs the code in train and test mode. Global flag
    TEST_MODE defines if it is train mode or test mode."""
    if not TEST_MODE:
        cross_validate()
    else:
        full_task_train_test()


def full_task_train_test() -> None:
    """Trains on full task data and predicts tests on the task test data"""
    (
        train_data_df,
        dbpedia_data_dev_df,
        wikidata_data_dev_df,
    ) = hc.get_full_training_test_data()
    print(dbpedia_data_dev_df.head())
    item_counts = train_data_df["category"].value_counts()
    print(item_counts)
    train_df, label_mapping = get_st_data_with_labels(
        train_data_df.question, train_data_df.category
    )
    item_counts = train_df["labels"].value_counts()
    print(item_counts)
    model = train_bert(train_df, len(label_mapping))
    dbpedia_y_pred, _ = pred_bert(
        model, dbpedia_data_dev_df.question, label_mapping
    )
    wikidata_y_pred, _ = pred_bert(
        model, wikidata_data_dev_df.question, label_mapping
    )
    hc.dump_predictions(
        dbpedia_data_dev_df,
        dbpedia_y_pred,
        'bert',
        MODEL_NAME + '_' + str(N_EPOCHS),
        'dbpedia',
        SUBMISSION_MODE=TEST_MODE,
    )
    hc.dump_predictions(
        wikidata_data_dev_df,
        wikidata_y_pred,
        'bert',
        MODEL_NAME + '_' + str(N_EPOCHS),
        'wikidata',
        SUBMISSION_MODE=TEST_MODE,
    )


def cross_validate() -> None:
    """Trains on pre-made train and tests on dev splits."""
    for split in range(hc.NUM_SPLITS):
        print('training bert for split ' + str(split))
        """Jointly train for categories and predict separately
        for wikidata and dbpedia"""
        (
            train_data_df,
            dbpedia_data_dev_df,
            wikidata_data_dev_df,
        ) = hc.get_split_train_test_data(split)
        train_df, label_mapping = get_st_data_with_labels(
            train_data_df.question, train_data_df.category
        )
        model = train_bert(train_df, len(label_mapping))
        (dbpedia_class_report, dbpedia_eval_score, dbpedia_y_pred,) = eval_bert(
            model,
            dbpedia_data_dev_df.question,
            dbpedia_data_dev_df.category,
            label_mapping,
        )
        print('dbpedia split ' + str(split) + ': ' + str(dbpedia_eval_score))
        print(dbpedia_class_report)
        (
            wikidata_class_report,
            wikidata_eval_score,
            wikidata_y_pred,
        ) = eval_bert(
            model,
            wikidata_data_dev_df.question,
            dbpedia_data_dev_df.category,
            label_mapping,
        )
        print('wikidata split ' + str(split) + ': ' + str(wikidata_eval_score))
        print(wikidata_class_report)
        hc.dump_predictions(
            dbpedia_data_dev_df,
            dbpedia_y_pred,
            'bert',
            MODEL_NAME + '_' + str(N_EPOCHS),
            'dbpedia',
            SUBMISSION_MODE=TEST_MODE,
            split=split,
        )
        hc.dump_predictions(
            wikidata_data_dev_df,
            wikidata_y_pred,
            'bert',
            MODEL_NAME + '_' + str(N_EPOCHS),
            'wikidata',
            SUBMISSION_MODE=TEST_MODE,
            split=split,
        )


if __name__ == '__main__':
    train_and_test_with_bert()
