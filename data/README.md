# Contents of data folder

## `smart-dataset`

  - Everything under this folder is from official task organizers.
  - Folders are rearranged to have [dbpedia](smart-dataset/dbpedia) and [wikidata](smart-dataset/wikidata) to contain train and test files.
  - [dbpedia/evaluation](smart-dataset/dbpedia/evaluation) and [wikidata/evaluation](smart-dataset/wikidata/evaluation) folders contain evaluation scripts from the organizers.
  - [wikidata/type_labels.tsv](smart-dataset/wikidata/type_labels.tsv) and [wikidata/type_description.tsv](smart-dataset/wikidata/type_description.tsv) contain the Wikidata type labels and type descriptions extracted from the external script.

## `dbpedia`

  - The [dump_sample](dbpedia/dump_sample) folder contains a sample of instance_types and instance_types_transitive files (official samples distributed by DBpedia); these are used for the tests.

## `wikidata`

  - The [dump_sample](wikidata/dump_sample) folder contains a sample of the 2018-05-10 Wikidata dump (`latest-truthy.nt`).
  - For more efficient processing, we extract a small subject-object pairs for selected predicates from this dump. For a more compact representation, the `http://wikidata.org/entity` prefixes are removed and items are represented by their ID (`Qxxx`). Specifically:
    - instanceOf `<http://www.wikidata.org/prop/direct/P31>` => `instance_of.tsv`
      - This file is generated using:
        ```
        grep "/P31> " latest-truthy.nt | cut -d" " -f1,3 | sed 's/<http:\/\/www.wikidata.org\/entity\///g' | sed 's/>//g' | sed 's/ /\t/' > instance_of.tsv
        ```  
    - subclassOf `<http://www.wikidata.org/prop/direct/P279>` => `subclass_of.tsv`
       - This file is created the same way as above, except grepping for P279.
    - label `<http://www.w3.org/2000/01/rdf-schema#label>` => `label.tsv` 
      - This file is generated using the [wikidata_labels_extractor script](../scripts/wikidata_labels_extractor.py).
    - description `<http://schema.org/description>` => `description.tsv`
      - This file is generated using the [wikidata_labels_extractor script](../scripts/wikidata_labels_extractor.py).
  - The full files are on gustav1 at `/data/scratch/smart-task/emnlp2021-smart/data/wikidata`. A sample from these files for the tests is found under [extracts_sample](wikidata/extracts_sample).
  - To analyze the coverage of the SMART types in a given Wikidata dump, run the script
    ```
    python -m smart.utils.wikidata -d $DUMP_DIR -s [train|test]
    ```
  - For the types in the SMART dataset, type label and descriptions were extracted from the Wikidata API `https://www.wikidata.org/w/api.php?action=wbsearchentities&search=type_id` to get `label` and `description` using the external [script](https://github.com/iai-group/iswc-smart-task/blob/master/smart/utils/wikidata/wikidata_entity_extractor.py) and written to following tsv files:
    -  `data/smart_dataset/wikidata/type_labels.tsv`: `type_id` and `search.label` from above API with tab delimiter.
    - `data/smart_dataset/wikidata/type_description.tsv`: `type_id` and `search.description` from above API with tab delimiter.
   
