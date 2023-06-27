"""This file is used for predicting the answer categories for MSMARCO
 questions using a BERT model trained on SMART task data on DBPedia
 and Wikidata. This file is currently not needed for this project.


Typical usage example:
  python -m smart.classification.category.msmarco_catgory_classification.

Author: Vinay Setty
"""
from typing import List
import pandas as pd
from simpletransformers.classification import ClassificationModel

import smart.classification.category.bert_category_classifier as bcc
import smart.classification.category.category_classification_utils as hc
import smart.utils.file_utils as fu


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

MSMARCO_QUESTIONS = 'data/msmarco/msmarco-questions-all.tsv'
MODEL_PATH = 'roberta-large_2'
PRED_FILE = 'data/msmarco/msmarco-questions-all-pred.json'
TREC_CAST_FILE = 'data/trec_cast/trec_cast_questions_2020.txt'
TREC_CAST_PRED_FILE = 'data/trec_cast/trec_cast_questions_2020_pred.json'
TREC_CAST_PRED_FILE1 = 'data/smart_dataset/trec/smarttask_trec_test.json'
TREC_CAST_PRED_FILE2 = 'data/smart_dataset/trec/smarttask_trec_cat_pred.json'


def get_data_df(data_file: str) -> pd.DataFrame:
    """Reads the questions file into a pandas Dataframe

    Returns:
        Dataframe containing text questions.
    """
    return pd.read_csv(data_file, sep='\t', header=0)


def load_model(num_labels) -> ClassificationModel:
    """Loads a pre-trained BERT model from MODEL_PATH.

    Returns:
        model: A ClassificationModel object
    """
    args = (
        {
            'reprocess_input_data': True,
            'train_batch_size': 512,
            'eval_batch_size': 512,
            'gradient_accumulation_steps': 2,
            'weight': [2, 1, 1, 2, 10],
            'learning_rate': 3e-6,
            'save_eval_checkpoints': False,
            'save_model_every_epoch': False,
            'show_running_loss': True,
        },
    )
    model = ClassificationModel(
        'roberta', MODEL_PATH, num_labels=num_labels, args=args
    )
    return model


if __name__ == '__main__':
    data_df = get_data_df(TREC_CAST_FILE)
    print(data_df.head())
    (
        train_data_df,
        dbpedia_data_dev_df,
        wikidata_data_dev_df,
    ) = hc.get_full_training_test_data()
    train_df, label_mapping = bcc.get_st_data_with_labels(
        train_data_df.question, train_data_df.category
    )
    keys_list = list(label_mapping)
    model = load_model(len(label_mapping))
    preds, pred_probs = bcc.pred_bert(
        model, list(data_df.question.values), label_mapping
    )
    data_df['category'] = preds
    pred_dicts = []
    all_types: List[List] = []
    for pred in pred_probs:
        zip_iterator = zip(keys_list, pred)
        pred_dicts.append(dict(zip_iterator))
        all_types.append([])
    print(pred_dicts)
    data_df['category_pred_prob'] = pred_dicts
    data_df['type'] = all_types
    print(data_df.head())
    fu.dump_json(data_df, TREC_CAST_PRED_FILE)
    fu.dump_json(data_df, TREC_CAST_PRED_FILE1)
    fu.dump_json(data_df, TREC_CAST_PRED_FILE2)
