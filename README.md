# Extreme Classification for Answer Type Prediction in Question Answering

# Abstract
Semantic answer type prediction (SMART) is known to be a useful step towards designing effective question answering (QA) systems. The SMART task involves predicting the top-$k$ knowledge graph (KG) types for a given natural language question. This is challenging due to the large number of types in KGs. In this paper, we propose use of extreme multi-label classification using Transformer models (XBERT) by clustering KG types using structural and semantic features based on question text. We specifically improve the clustering stage of the XBERT pipleline using textual and structural features derived from KGs. We show that these features can improve end-to-end performance for the SMART task, and yield state-of-the-art results. 

# Installation
1. Create a conda environment conda create -n jcdl-smart-2023
2. Install the requirements pip install -r requirements.txt
3. Install apex a. Clone apex repository b. install with command pip install -v --disable-pip-version-check --no-cache-dir ./


# Usage

1. On a GPU server run python -m smart.classification.category.bert_category_classifier
2. Modify the bash script scripts/run_experiments.sh to specify appropriate datasets, embedding types and Transformer models.

```
EMB_LIST=( pifa-neural text-emb pifa-tfidf )
dataset_list=( wikidata dbpedia )
model_list=( roberta xlnet )
```

3. Then run ```./scripts/run_experiments.sh```
4. It should run full pipeline with feature generation to model training, ranking, prediction and evaluation using the script. It should print NDCG for DBPedia and MRR for Wikidata.
5. To run with a sample data and for pytest unit tests to pass run ```./scripts/run_experiments.sh True``` The True flag runs it in test mode.
