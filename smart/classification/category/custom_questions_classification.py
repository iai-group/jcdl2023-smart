"""This file is used for predicting the answer categories for MSMARCO
 questions using a BERT model trained on SMART task data on DBPedia
 and Wikidata. This file is currently not needed for this project.


Typical usage example:
  python -m smart.classification.category.msmarco_catgory_classification.

Author: Vinay Setty
"""

import pandas as pd
import numpy as np

import smart.classification.category.bert_category_classifier as bcc
import smart.classification.category.category_classification_utils as hc
from simpletransformers.classification import ClassificationModel
import smart.utils.file_utils as fu


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

MSMARCO_QUESTIONS = 'data/msmarco/msmarco-questions-all.tsv'
TREC_QUESTIONS = 'data/trec_cast/trec_cast_questions_2021.txt'
MODEL_PATH = 'roberta-large_2'
PRED_FILE = 'data/smart_dataset/trec/smarttask_trec_cat_pred.json'
PRED_FILE1 = 'data/smart_dataset/trec/smarttask_trec_test.json'


def get_data_df(data_file: str) -> pd.DataFrame:
    """Reads the MSMARCO questions file into a pandas Dataframe

    Returns:
        Dataframe containing MSMARCO questions.
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
            'train_batch_size': 1,
            'eval_batch_size': 1,
            'gradient_accumulation_steps': 2,
            'weight': [2, 1, 1, 2, 10],
            'learning_rate': 3e-6,
            'save_eval_checkpoints': False,
            'save_model_every_epoch': False,
            'show_running_loss': True,
        },
    )
    model = ClassificationModel(
        'roberta', MODEL_PATH, num_labels=num_labels, use_cuda=False, args=args
    )
    return model


if __name__ == '__main__':
    # msmarco_df = get_msmarco_df(MSMARCO_QUESTIONS)
    msmarco_df = get_data_df(TREC_QUESTIONS)
    print(msmarco_df.head())
    (
        train_data_df,
        dbpedia_data_dev_df,
        wikidata_data_dev_df,
    ) = hc.get_full_training_test_data()
    train_df, label_mapping = bcc.get_st_data_with_labels(
        train_data_df.question, train_data_df.category
    )
    model = load_model(len(label_mapping))
    print(msmarco_df.question.values)
    preds, pred_probs = bcc.pred_bert(
        model, list(msmarco_df.question.values), label_mapping
    )
    msmarco_df['category'] = preds
    types = []
    for idx, pred in enumerate(preds):
        types.append([])
    msmarco_df['type'] = types
    print(pred_probs)
    pred_pairs = []
    for pred_prob in pred_probs.tolist():
        category_prob = np.column_stack((label_mapping, pred_prob))
        pred_pairs.append(category_prob.tolist())
    print(pred_pairs)
    msmarco_df['category_pred_prob'] = pred_pairs
    print(msmarco_df.head())
    fu.dump_json(msmarco_df, PRED_FILE)
    fu.dump_json(msmarco_df, PRED_FILE1)
