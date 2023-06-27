import os
from math import floor
import nltk
from random import sample
from typing import Dict, List

import smart.utils.triple_utils as triple_utils
from smart.utils.file_utils import read_file_as_list

DBPEDIA_TYPES_FILE = "data/dbpedia/dbpedia-2016-10/dbpedia_2016-10.nt"
WIKIDATA_TYPES_FILE = "data/wikidata/type_description.tsv"
DBPEDIA_TYPES_FILE_UPDATED = "data/dbpedia/type_description.tsv"
DBPEDIA_PATH = "data/dbpedia/dbpedia-2016-10/core-i18n/en/"
ENTITY_ABSTRACTS_FILE = "short_abstracts_en.ttl.bz2"
ENTITY_TYPES_FILES = [
    "instance_types_en.ttl.bz2",
    "instance_types_transitive_en.ttl.bz2",
]
DP_WD_EQC_FILE = (
    "data/dbpedia/dbpedia-2016-10/embeddings/"
    "dbpedia_wikidata_equivalent_classes.tsv"
)
WD_TYPE_LABELS = "data/wikidata/type_labels.txt"

MAX_BULKING_DOC_SIZE = 10240  # Max doc len when bulking, in chars (20MB)
AVG_SHORT_ABSTRACT_LEN = 216


class TypeDescription:
    def __init__(
        self,
        dataset: str,
        use_entity_descriptions: bool = False,
        generate_type_description: bool = False,
    ) -> None:
        """Instantiates a class to get textual descriptions of types in
        DBPedia and Wikidata.

        Args:
            dataset: String constant specifying "dbpedia" or "wikidata"
            use_entity_descriptions: Flag which specifies if the type
            description should be augmented with corresponding entity
            descriptions.
        """
        self._type_description = {}
        self._dataset = dataset
        self._wikidata_type_description = self._load_type_description_file(
            WIKIDATA_TYPES_FILE
        )
        self._dbpedia_type_desc = self._load_type_description_file(
            DBPEDIA_TYPES_FILE_UPDATED
        )
        if generate_type_description:
            self._wikdata_type_labels = (
                TypeDescription.load_wikidata_type_labels()
            )
            self._eq_classes = (
                TypeDescription.load_dbpedia_wikidata_equivalent_classes()
            )
            print("loading _load_dbpedia_entity_abstracts")
            self._dbpedia_entity_abstracts = (
                self._load_dbpedia_entity_abstracts()
            )
            print(len(self._dbpedia_entity_abstracts))
            print("loading _load_dbpedia_entity_types")
            self._type_entities = self._load_dbpedia_entity_types()
            self._dbpedia_type_desc = self._get_dbpedia_label_descriptions_dict(
                use_entity_descriptions
            )

    def _get_dbpedia_label_descriptions_dict(
        self, use_entity_descriptions: bool = False
    ) -> Dict[str, str]:
        """Extracts comments for types from NT file.

        Returns:
            A dictionary with DBPedia types as keys and their textual
            description as values.
        """
        types_description = {}
        # Read type=>en comments.
        unique_types = set()
        with open(DBPEDIA_TYPES_FILE) as in_file:
            for line in in_file:
                subj, pred, obj = triple_utils.get_triple(line)
                if "comment" in pred and "@en" in obj:
                    types_description[subj] = obj.split("@en")[0].replace(
                        '"', ""
                    )
                unique_types.add(subj)
        print(use_entity_descriptions)

        if use_entity_descriptions:
            for type in unique_types:

                type_without_dbo = type.replace("dbo:", "")
                if type_without_dbo in self._eq_classes:
                    prev_desc = ""
                    if type in types_description:
                        prev_desc = types_description[type]
                    types_description[type] = (
                        prev_desc
                        + " "
                        + self._get_equivalent_wikidata_description(
                            type_without_dbo
                        )
                    )
                abstract_summary = self._make_type_doc(type)
                print(abstract_summary)
                print(type_without_dbo, abstract_summary)
                if type in types_description:
                    prev_desc = types_description[type]
                    types_description[type] = f"{prev_desc} {abstract_summary}"
                else:
                    types_description[type] = abstract_summary
                # break
        print(self._eq_classes["Activity"])
        print(self._wikdata_type_labels["Q1914636"])
        print(self._wikidata_type_description["activity"])
        return types_description

    def _get_equivalent_wikidata_description(
        self, type_without_dbo: str
    ) -> str:
        wd_eq_type = self._eq_classes[type_without_dbo]
        if type == "Activity":
            print(wd_eq_type)
        if wd_eq_type not in self._wikdata_type_labels:
            return ""
        wikidata_type = self._wikdata_type_labels[wd_eq_type]
        # print(wikidata_type)
        if wikidata_type not in self._wikidata_type_description:
            return ""
        return self._wikidata_type_description[wikidata_type].replace("\n", "")

    def _load_type_description_file(
        self, type_description_file: str
    ) -> Dict[str, str]:
        """Get Wikidata comments for types from type_description.tsv.

        Returns:
            A dictionary with Wikdata types as keys and their textual
            description as values.
        """
        type_dict = {}
        with open(type_description_file) as in_file:
            for line in in_file:
                fields = line.split("\t")
                if len(fields) == 2:
                    type_dict[fields[0]] = fields[1]
        return type_dict

    def get_type_description(self, type: str) -> str:
        """Returns the description or summary for a given type.

        Args:
            type: Type in string form for which the description is needed.

        Returns:
            Type description text if type exits in the dictionary otherwise None
        """
        if self._dataset == "dbpedia":
            if type in self._dbpedia_type_desc:
                return self._dbpedia_type_desc[type]
            else:
                return None
        elif self._dataset == "wikidata":
            if type in self._wikidata_type_description:
                return self._wikidata_type_description[type]
            else:
                return None

    def _load_dbpedia_entity_abstracts(self) -> Dict[str, str]:
        num_lines = 0
        entity_abstracts = {}
        for line in read_file_as_list(
            filename=os.sep.join([DBPEDIA_PATH, ENTITY_ABSTRACTS_FILE])
        ):
            try:
                entity, _, abstract = triple_utils.get_triple(
                    line.decode("utf-8")
                )
            except ValueError:
                continue
            if abstract and len(abstract) > 0:  # skip empty objects
                entity_abstracts[entity] = abstract

            num_lines += 1
            if num_lines % 10000 == 0:
                print("  {}K lines processed".format(num_lines // 1000))
                break
        return entity_abstracts

    def _load_dbpedia_entity_types(self) -> Dict[str, List[str]]:
        num_lines = 0
        types_entities = {}
        for types_file in ENTITY_TYPES_FILES:
            filename = os.sep.join([DBPEDIA_PATH, types_file])
            print("Loading entity types from {}".format(filename))
            for line in read_file_as_list(filename):
                try:
                    entity, _, entity_type = triple_utils.get_triple(
                        line.decode("utf-8")
                    )
                except ValueError:
                    continue
                # print(entity, entity_type)
                if type(entity_type) != str:  # Likely result of parsing error
                    continue
                if not entity_type.startswith("dbo:"):
                    # print("  Non-DBpedia type: {}".format(entity_type))
                    continue
                if not entity.startswith("dbr:"):
                    # print("  Invalid entity: {}".format(entity))
                    continue
                if entity_type not in types_entities:
                    types_entities[entity_type] = []
                types_entities[entity_type].append(entity)

                num_lines += 1
                if num_lines % 1000 == 0:
                    print("  {}K lines processed".format(num_lines // 1000))
                    break
            print("  Done.")
            print(len(types_entities))
        return types_entities

    def _make_type_doc(self, type_name):
        """Gets the document representation of a type to be indexed, from its
        entity short abstracts."""
        if type_name not in self._type_entities:
            return ""
        abstracts = []
        for e in self._type_entities.get(type_name):
            if e not in self._dbpedia_entity_abstracts:
                continue
            abst = self._dbpedia_entity_abstracts.get(e)
            # Take one first sentence
            abstracts.append(nltk.sent_tokenize(abst)[0])
        content = "\n".join(abstracts)

        if len(content) > MAX_BULKING_DOC_SIZE:
            print(
                "Type {} has content larger than allowed: {}.".format(
                    type_name, len(content)
                )
            )

            # we randomly sample a subset of Y entity abstracts, s.t.
            # Y * AVG_SHORT_ABSTRACT_LEN <= MAX_BULKING_DOC_SIZE
            num_entities = len(self._type_entities[type_name])
            amount_abstracts_to_sample = min(
                floor(MAX_BULKING_DOC_SIZE / AVG_SHORT_ABSTRACT_LEN),
                num_entities,
            )
            entities_sample = [
                self._type_entities[type_name][i]
                for i in sample(range(num_entities), amount_abstracts_to_sample)
            ]
            content = ""  # reset content
            for entity in entities_sample:
                if entity in self._dbpedia_entity_abstracts:
                    new_content_candidate = "\n".join(
                        [content, self._dbpedia_entity_abstracts.get(entity)]
                    )
                    # we add an abstract only if by doing so it will not exceed
                    # MAX_BULKING_DOC_SIZE
                    if len(new_content_candidate) > MAX_BULKING_DOC_SIZE:
                        break

                    content = new_content_candidate
        content = content.replace("\n", "").replace("\"", "")
        return content

    @staticmethod
    def load_dbpedia_wikidata_equivalent_classes():
        eq_classes = {}
        with open(DP_WD_EQC_FILE, 'r') as f:
            for line in f:
                fields = line.split('\t')
                eq_classes[
                    fields[0].replace("http://dbpedia.org/ontology/", "")
                ] = (
                    fields[1]
                    .replace('\n', '')
                    .replace("http://www.wikidata.org/entity/", "")
                )
        return eq_classes

    @staticmethod
    def load_wikidata_type_labels():
        type_labels = {}
        with open(WD_TYPE_LABELS, 'r') as f:
            for line in f:
                fields = line.split('|')
                # print(fields)
                type_labels[fields[0]] = fields[1].replace('\n', '')
        return type_labels

    def dump_dbpedia_type_descriptions(self):
        with open(DBPEDIA_TYPES_FILE_UPDATED, "w") as f:
            for key, value in self._dbpedia_type_desc.items():
                # print(f"{key}\t{value}")
                f.write(f"{key}\t{value}\n")


if __name__ == "__main__":
    td = TypeDescription("dbpedia", use_entity_descriptions=True)
    td.get_type_description("dbo:Activity", "dbpedia")
    td.dump_dbpedia_type_descriptions()
