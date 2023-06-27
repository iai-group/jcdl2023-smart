# List of IR features

Based on (Garigliotti et al., SIGIR'17)

| **#** | **Feature** | **Type** | **Description** | **Value** | **Implementation** |
| -- | -- | -- | -- | -- | -- | 
| *Baseline features* ||||||
| 1-5 | EC_BM25_k(t,q) | QT | Entity-centric type score using BM25 ranker, k={5, 10, 20, 50, 100} | [0..inf) | |
| 6-10 | EC_LM_k(t,q) | QT | Entity-centric type score using LM ranker, k={5, 10, 20, 50, 100} | [0..1] | |
| 11 | TC_BM25(t,q) | QT | Type-centric score using BM25 ranker | [0..inf) | |
| 12 | TC_LM(t,q) | QT | Type-centric score using LM ranker | [0..1] | |
| *Knowledge base features* ||||||
| 13 | DEPTH(t) | T | The hierarchical level of type t, normalized by the taxonomy depth | [0..1] | `hierarchy_features (depth)` |
| 14 | CHILDREN(t) | T | Number of children of type t in the taxonomy | [0..inf) | `hierarchy_features (num_children)` |
| 15 | SIBLINGS(t) | T | Number of siblings of type t in the taxonomy | [0..inf) | `hierarchy_features (num_siblings)` |
| 16 | ENTITIES(t) | T | Number of entities mapped to type t | [0..inf) | `hierarchy_features (num_entities)` |
| *Type label features* ||||||
| 17 | LENGTH(t) | T | Length of (the label of) type t in words | [1..inf) | |
| 18 | IDFSUM(t) | T | Sum of IDF for terms in (the label of) type t | [0..inf) | |
| 19 | IDFAVG(t) | T | Avg of IDF for terms in (the label of) type t | [0..inf) | |
| 20-21 | JTERMS_n(t,q) | QT | Query-type Jaccard similarity for sets of n-grams, for n={1, 2} | [0..1] | |
| 22 | JNOUNS(t,q) | QT | Query-type Jaccard similarity using only nouns| [0..1] | |
| 23 | SIMAGGR(t,q) | QT | Cosine sim. between the q and t *word2vec* vectors aggregated over all terms of their resp. labels | [0..1] | |
| 24 | SIMMAX(t,q) | QT | Max. cosine similarity of *word2vec* vectors between each pair of query (q) and type (t) terms | [0..1] | |
| 25 | SIMAVG(t,q) | QT | Avg. of cosine similarity of *word2vec* vectors between each pair of query (q) and type (t) terms | [0..1] | |

## Generating LTR results for SMART, DBpedia dataset

