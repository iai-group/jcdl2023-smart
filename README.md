# Extreme Classification for Answer Type Prediction in Question Answering
# Citation
If you are using the contents of this repository please cite it this way:
```
@article{Setty:2023:Arxiv,
  title={Extreme Classification for Answer Type Prediction in Question Answering},
  author={Setty, Vinay},
  journal={arXiv preprint arXiv:2304.12395},
  year={2023}
}
```

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

# Results

<table>
  <tr>
    <th>Method</th>
    <th colspan="3">DBpedia Type Classification</th>
    <th></th>
    <th colspan="3">DBpedia End-to-End</th>
    <th></th>
    <th>Wikidata Type Classification</th>
    <th>Wikidata End-to-End</th>
  </tr>
  <tr>
    <th></th>
    <th>NDCG@3</th>
    <th>NDCG@5</th>
    <th>NDCG@10</th>
    <th></th>
   <th>NDCG@3</th>
    <th>NDCG@5</th>
    <th>NDCG@10</th>
    <th></th>
    <th>MRR</th>
    <th>MRR</th>
  </tr>
  <tr>
    <td>Question text (TF-IDF)</td>
    <td>0.717</td>
    <td>0.693</td>
    <td>0.650</td>
    <td></td>
    <td>0.824</td>
    <td>0.811</td>
    <td>0.787</td>
    <td></td>
    <td>0.66</td>
    <td>0.76</td>
  </tr>
  <tr>
    <td>KG-TypeSim</td>
    <td>0.725</td>
    <td>0.697</td>
    <td>0.662</td>
    <td></td>
    <td>0.828</td>
    <td>0.813</td>
    <td>0.793</td>
    <td></td>
    <td>0.67</td>
    <td>0.77</td>
  </tr>
  <tr>
    <td>KG-RDF2Vec</td>
    <td>0.729</td>
    <td>0.701</td>
    <td>0.656</td>
    <td></td>
    <td>0.831</td>
    <td>0.815</td>
    <td>0.791</td>
    <td></td>
    <td>0.67</td>
    <td>0.78</td>
  </tr>
  <tr>
    <td>BERT-TypeDesc</td>
    <td>0.727</td>
    <td>0.706</td>
    <td>0.665</td>
    <td></td>
    <td>0.830</td>
    <td>0.818</td>
    <td>0.795</td>
    <td></td>
    <td>0.67</td>
    <td>0.78</td>
  </tr>
  <tr>
    <td>BERT-TypeDesc-FT</td>
    <td>0.734</td>
    <td>0.712</td>
    <td>0.678</td>
    <td></td>
    <td>0.834</td>
    <td>0.822</td>
    <td>0.802</td>
    <td></td>
    <td>0.68</td>
    <td>0.79</td>
  </tr>
</table>

