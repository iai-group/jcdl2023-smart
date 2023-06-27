# Results on the test data of the task for DBPedia
## Evaluation results for Category classification

| Method   | DataSet  |  Test Accuracy |
|----------|:-------------:|------:|
| BERT |  DBPEdia   |  0.978 |


## Evaluation results for Resource Type classification  (DBPedia)

| Method   | Train NDCG@5 | Train NDCG@10 | Test NDCG@3 | Test  NDCG@5 | Test NDCG@10 | Run file |
|----------|------:|------:|------:|------:|------:|------------------------------------:|
| Gold as submission | - | -  | 0.932 | 0.863  | 0.809 | |
| *BERT-XBERT-xlnet (type-features)*  |-| - | 0.809 | *0.812* | *0.800* | roberta-large_10_results_dbpedia_task_test.json_type-features_xbert_predictions.json |
| BERT-XBERT-roberta (text-emb)  |-| - | *0.810* | 0.808 | 0.787 | roberta-large_10_results_dbpedia_task_test.json_text-emb_xbert_predictions.json |
| BERT-XBERT-xlnet (pifa-neural)  |-| - | 0.799 | 0.803 | 0.793 | xlnet-large_10_results_dbpedia_task_test.json_pifa-tfidf_xbert_predictions.json |
| BERT-XBERT-roberta (pifa-tfidf)  | 0.776 | 0.747 | 0.797 | 0.796 | 0.778 | smart_task_test_category_predicted.json_pifa-tfidf_xbert_predictions.json |
| BERT-IR Type-centric (BM25)  | 0.492 | 0.509 | | | |
| BERT-IR Type-centric (LM)  | 0.526 | 0.541 | | | | |
| BERT-IR Entity-centric (BM25, k=20)  | 0.483 | 0.503 | | | | |
| BERT-IR Entity-centric (BM25, k=50)  | 0.448 | 0.474 | | | | |
| BERT-IR Entity-centric (BM25, k=100)  | 0.423 | 0.444 | | | | |
