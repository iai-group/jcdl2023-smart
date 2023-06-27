"""Utility class for working with the SMART dataset.

The following simple data cleaning steps are performed:
  * Queries with empty question text are ignored.
  * Angle brackets in question text are removed.
"""

import json
import re
from typing import Dict, Set, List

import pandas as pd
from data.smart_dataset.dbpedia.evaluation.evaluate import (
    get_expanded_types,
    load_type_hierarchy,
)


class Dataset:
    def __init__(
        self, dataset: str, split: str, expand_types: bool = False
    ) -> None:
        filename = "data/smart_dataset/"
        self._dataset_str = dataset.lower()
        self._split = split
        if self._dataset_str == "dbpedia":
            filename += f"dbpedia/smarttask_dbpedia_{self._split}.json"
        elif self._dataset_str == "trec":
            filename += f"trec/smarttask_trec_{self._split}.json"
        elif self._dataset_str == "wikidata":
            filename += f"wikidata/lcquad2_anstype_wikidata_{self._split}.json"
        elif self._dataset_str == "dummy":
            filename = None
        else:
            raise Exception("Unknown dataset")
        if self._dataset_str == "dbpedia":
            self.type_hierarchy_file = (
                "data/smart_dataset/dbpedia/evaluation/dbpedia_types.tsv"
            )
            self.type_hierarchy, self.max_depth = load_type_hierarchy(
                self.type_hierarchy_file
            )
            self.expansion_depth = 2
            print(
                "expaning types up to depth ",
                self.expansion_depth,
                expand_types,
            )
        self._load_data(filename, expand_types)

    def _load_data(self, filename: str, expand_types: bool = False) -> None:
        """Loads SMART dataset and performs cleaning steps.

        Args:
            filename: JSON file provided by the SMART Challenge.
        """
        self._data = []
        q_ids = set()
        with open(filename, "r") as f:
            queries = json.load(f)
            for q in queries:
                # Skip queries which have missing fields.
                if "type" not in q or "category" not in q:
                    print(q, " has missing type")
                # Skip queries with empty question text.
                if not self.is_valid_question(q["question"]):
                    continue
                # Skip duplicates.
                if q["id"] in q_ids:
                    continue
                # Clean question text.
                q["question"] = self.clean_question_text(q["question"])
                q_ids.add(q["id"])
                # TODO(vinaysetty): Move expansion to feature generation
                # https://github.com/iai-group/smart/issues/155

                # If type expansion is needed, get extended types within the
                # depth self.expansion_depth of type heirarchy. and appned it to
                # ground-truth types. Note it is currently only supported for
                # dbpedia since type hierarchy is not available for wikidata.
                if (
                    expand_types
                    and self._dataset_str == "dbpedia"
                    and q["category"] == "resource"
                ):
                    expanded_types = get_expanded_types(
                        list(q["type"]), self.type_hierarchy
                    )
                    for expanded_type in expanded_types:
                        if (
                            self.type_hierarchy[expanded_type]["depth"]
                            <= self.expansion_depth
                        ):
                            if expanded_type not in set(q["type"]):
                                q["type"].append(expanded_type)
                self._data.append(q)

    def get_queries(self) -> Dict[str, str]:
        """Returns the set of queries.

        Returns:
            Dictionary with question id as key and text as value.
        """
        return {q["id"]: q["question"] for q in self._data}

    def dump_queries_tsv(self, output_file: str) -> None:
        """Dumps queries to a TSV file (e.g., for Pyserini).

        Args:
            output_file: Path to output TSV file.
        """
        print(f"Writing queries to {output_file}")
        with open(output_file, "w") as f:
            for q_id, query in self.get_queries().items():
                f.write(f"{q_id}\t{query}\n")

    def dump_queries_json(self, output_file: str) -> None:
        """Dumps queries to a JSON file (e.g., for Nordlys).

        Args:
            output_file: Path to output JSON file.
        """
        print(f"Writing queries to {output_file}")
        with open(output_file, "w") as f:
            json.dump(self.get_queries(), f, indent=4)

    @staticmethod
    def dump_json(
        data: List[Dict], output_file: str, start: int = 0, end: int = -1
    ) -> None:
        """Dumps queries to a JSON file (e.g., for Nordlys).

        Args:
            output_file: Path to output JSON file.
        """
        print(f"Writing queries to {output_file}")
        json_str = json.dumps(data[start:end], indent=4, sort_keys=True)
        with open(output_file, 'w') as json_file:
            json_file.write(json_str)

    def get_df(self) -> pd.DataFrame:
        """Returns the whole data as a dataframe.

        Returns:
            pd.DataFrame: Pandas dataframe with full data.
        """
        return pd.DataFrame(self._data)

    def get_unique_types(self, lowercase: bool = False) -> Set[str]:
        """Returns the set of unique types from the dataset (excluding
        owl:Thing).

        Args:
            lowercase: Lowercase type labels (default: False).

        Returns:
            Set with the labels of unique types.
        """
        df = self.get_df()
        answer_types = df[df["category"] == "resource"]
        type_set = set(
            t.lower() if lowercase else t
            for types in answer_types["type"].to_list()
            for t in types
        )
        print("len of unique types", len(type_set))
        type_set.discard("owl:Thing")  # Do not return root type.
        return type_set

    def clean_question_text(self, question_text: str) -> str:
        """Look for obvious errors in the question.

        Args:
            question_text: String of the question.

        Returns:
            clean question string
        """
        question_text = re.sub(' +', ' ', question_text)
        question_text = (
            question_text.replace("{", "")
            .replace("}", "")
            .replace("\"", "")
            .replace("\n", "")
        )
        return question_text

    def is_valid_question(self, question_text: str) -> bool:
        """Checks if the given question text is a valid question. If the
        question contains na, n/a, null or less than certain number of words it
        is invalid question.

        Args:
            question_text: Input question text.

        Returns:
            Returns true if the question is valid else false.
        """
        if not question_text:
            return False
        question_len = len(question_text)
        num_words = len(question_text.split(" "))
        if (
            question_text == "na"
            or question_text == "n/a"
            or question_text == "null"
            or question_len < 10
            or question_len > 300
            or num_words < 3
        ):
            return False
        else:
            return True

    def get_split_name(self) -> str:
        """Get the split type (train or test).

        Returns:
            Split name in string format.
        """
        return self._split

    def get_dataset_name(self) -> str:
        """Get the dataset name such as dbpedia or wikidata.

        Returns:
            Dataset name in string format.
        """
        return self._dataset_str
