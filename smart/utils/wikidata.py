"""Data loader utility class for Wikidata.

For faster processing, some of the functionality relies on extracts from the
Wikidata dump (described in data/README.md). These extracts are tab-separated
item ID and value pairs (where item IDs are in Qxx format, i.e., without the
wd: prefix).

Note that entity-type relationships in Wikidata are noisy. That is, the
subclassOf and instanceOf relationships are not used consistently, which makes
it difficult to differentiate between entities and types. For example, "Linux"
(Q388) is both a type and an entity.

SMART dataset includes the types in the string format such as "Linux". However,
the Wikidata extracts use id format like "Q388". In this module types are
represented with string labels not type ids to keep it consistent with the
SMART dataset.
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import smart.utils.triple_utils as triple_utils
from smart.utils.dataset import Dataset
from smart.utils.type_system import TypeSystem
from tqdm import tqdm

WIKIDATA_SMART_DATASET_DIR = "data/smart_dataset/wikidata/"


class Wikidata(TypeSystem):
    """Utility class for handling Wikidata."""

    DUMP_FILE = "latest-truthy.nt"
    EXTRACTS_FILE_DESCRIPTION = "description.tsv"
    EXTRACTS_FILE_LABEL = "label.tsv"
    EXTRACTS_FILE_INSTANCE_OF = "instance_of.tsv"
    EXTRACTS_FILE_SUBCLASS_OF = "subclass_of.tsv"
    SMART_WIKDATA_LABELS = "type_labels.tsv"
    SMART_WIKDATA_TYPE_DESCRIPTIONS = "type_description.tsv"

    # Number of lines based on 20180510 dump for tdqm progress bar.
    NUM_LINES = {
        DUMP_FILE: 2811805180,
        EXTRACTS_FILE_DESCRIPTION: 35424076,
        EXTRACTS_FILE_LABEL: 35296988,
        EXTRACTS_FILE_INSTANCE_OF: 44572441,
        EXTRACTS_FILE_SUBCLASS_OF: 1896081,
        SMART_WIKDATA_LABELS: 2644,
        SMART_WIKDATA_TYPE_DESCRIPTIONS: 1995,
    }

    def __init__(
        self,
        dump_dir: str = None,
        extracts_dir: str = None,
        smart_dataset_dir: str = WIKIDATA_SMART_DATASET_DIR,
    ) -> None:
        """Initializes the class.

        Args:
            dump_dir: Location of the Wikidata dump (in .nt format).
            extracts_dir: Location of the Wikidata extracts files (in .tsv
            format).
            smart_dataset_dir: Location of the Wikidata SMART dataset extracts
            files (in .tsv format).
        """
        self._dump_file = None
        if dump_dir:
            self._dump_file = os.path.join(dump_dir, self.DUMP_FILE)
            # Check if dump file may be found on dump path.
            if not os.path.isfile(self._dump_file):
                raise FileNotFoundError

        self._extracts_dir = None
        self._extracts = {}
        if extracts_dir:
            self._extracts_dir = extracts_dir
            # Check if all extract files exist.
            for f in self.extracts_files:
                if not os.path.isfile(os.path.join(self._extracts_dir, f)):
                    raise FileNotFoundError
            # Load from dump only if extracts_dir is given
            self._type_entities = defaultdict(
                set
            )  # Set of entities for each type
            self._entity_types = defaultdict(
                set
            )  # Set of types for each entity
            # Load entity-type assignments from dumpfile.
            for (
                entity,
                types,
            ) in self._load_entity_types_from_extracts().items():
                for entity_type in types:
                    self._type_entities[entity_type].add(entity)
                    self._entity_types[entity].add(entity_type)

        self._smart_dataset_extracts_dir = smart_dataset_dir
        # Check if all smart dataset extract files exist.
        for f in self.smart_dataset_extracts_files:
            if not os.path.isfile(
                os.path.join(self._smart_dataset_extracts_dir, f)
            ):
                raise FileNotFoundError

        # Wikidata type id (Q...) to Wikidata string label mapping
        self._type_labels = self.get_extracts(self.EXTRACTS_FILE_LABEL)
        # Wikidata string label to Wikidata type id (Q...) mapping
        self._label_type = {
            type_label: type_id
            for type_id, type_label in self._type_labels.items()
        }
        self._smart_dataset = Dataset(dataset="wikidata", split="train")
        self._types = self._smart_dataset.get_unique_types(lowercase=False)
        print("len(self._types)", len(self._types))

    @property
    def extracts_files(self) -> List[str]:
        """Returns a list of extract files.

        Returns:
            List of extract files.
        """
        return [
            self.EXTRACTS_FILE_LABEL,
            self.EXTRACTS_FILE_DESCRIPTION,
            self.EXTRACTS_FILE_SUBCLASS_OF,
            self.EXTRACTS_FILE_INSTANCE_OF,
        ]

    @property
    def smart_dataset_extracts_files(self) -> List[str]:
        """Returns a list of smart dataset files.

        Returns:
            List of SMART dataset files.
        """
        return [
            self.SMART_WIKDATA_LABELS,
            self.SMART_WIKDATA_TYPE_DESCRIPTIONS,
        ]

    def _load_triples_filtered(self, predicate: str) -> Tuple[str, str]:
        """Yields subject-object pairs from SPO triples that match a given
        predicate.

        Args:
            predicate: Predicate to filter on (without angle brackets).

        Yields:
            Tuple with subject and object.
        """
        if not self._dump_file:
            raise RuntimeError("No dump path provided on class init.")

        print(f"Loading triples from {self._dump_file}...")
        with open(self._dump_file, "r") as f:
            for _, line in enumerate(
                tqdm(f, file=sys.stdout, total=self.NUM_LINES[self.DUMP_FILE])
            ):
                try:
                    subj, pred, obj = triple_utils.get_triple(line)
                except ValueError:
                    continue
                if pred == predicate:
                    yield subj, obj

    def _load_entity_types_from_extracts(self) -> Iterable[Tuple[str, str]]:
        """Loads entity-type assignments from Wikidata dump.
        Works with both uncompressed .ttl and compressed .bz2 files.

        Yields:
            Sequence of entity and type tuples.
            Entities and Types are represented by their Wikidata ID.
        """
        return self.get_extracts(self.EXTRACTS_FILE_INSTANCE_OF)

    def _load_extracts_file(
        self, extracts_file
    ) -> Dict[str, Union[str, List[str]]]:
        """Loads the contents of an extracts file.

        Args:
            extracts_file: Name of the extracts file.

        Return:
            Dict with item IDs as keys; value is either a string (labels and
            descriptions) or a list (subclass_of and instance_of).
        """

        if extracts_file in self.extracts_files:
            filename = os.path.join(self._extracts_dir, extracts_file)
        elif extracts_file in self.smart_dataset_extracts_files:
            filename = os.path.join(
                self._smart_dataset_extracts_dir, extracts_file
            )
        else:
            raise ValueError(f"Unkown extract file: {extracts_file}")
        print(f"Loading extracted data from {filename}...")

        multivalued = extracts_file in [
            self.EXTRACTS_FILE_INSTANCE_OF,
            self.EXTRACTS_FILE_SUBCLASS_OF,
        ]
        # Multivalued predicates are stored as a list.
        contents = defaultdict(list) if multivalued else {}

        with open(filename, "r") as f:
            for _, line in enumerate(
                tqdm(f, file=sys.stdout, total=self.NUM_LINES[extracts_file])
            ):
                try:
                    # item_id, val = line.rstrip().split("\t")
                    item_id, val = re.split(r'\t+', line.rstrip())
                except ValueError as e:
                    print(line, e)
                    raise ValueError
                if multivalued:
                    contents[item_id].append(val)
                else:
                    contents[item_id] = val

        return contents

    def get_extracts(self, extracts_file) -> Dict[str, Union[str, List[str]]]:
        """Returns the contents of an extracts file.

        The method implements caching, i.e., it loads the contents only one.

        Args:
            extracts_file: Name of the extracts file.

        Return:
            Dict with the contents of the extracts file.
        """
        if extracts_file not in self._extracts:
            self._extracts[extracts_file] = self._load_extracts_file(
                extracts_file
            )
        return self._extracts[extracts_file]

    def get_item_predicate_values(self, predicate: str) -> Dict[str, str]:
        """Extracts values of a specific predicates for items from the Wikidata
        dump.

        Args:
            predicate: Predicate for which the values need to be extracted.

        Returns:
            Dictionary with item IDs as keys and predicate values (i.e.,
            objects) as values.
            Item IDs are without wd: prefix.
        """
        item_pred_values = {}
        for subj, obj in self._load_triples_filtered(predicate):
            # Only entities and English descriptions.
            if not subj.startswith("wd:Q") or not obj.endswith("@en"):
                continue
            item_id = subj[3:]  # Remove wd: prefix.
            value = obj[1:-4]  # Remove quotes and @en suffix.
            item_pred_values[item_id] = value
        return item_pred_values

    def get_types(self) -> Set[str]:
        """Returns a set of types in Wikidata datset mentioned in the SMART
        dataset.

        Returns:
            Set of item IDs (without wd: prefix) that are types.
        """
        return self._types

    def get_type_descriptions(self) -> Dict[str, str]:
        """Extracts type descriptions from Wikidata.

        Returns:
            Dictionary with type IDs as keys and descriptions as values.
        """
        descriptions = self.get_extracts(self.SMART_WIKDATA_TYPE_DESCRIPTIONS)
        return {
            t: descriptions.get(self.get_type_id(t), "") for t in self._types
        }

    def get_type_labels(self) -> Dict[str, str]:
        """Extracts type labels from Wikidata.

        Returns:
            Dictionary with type IDs as keys and labels as values.
        """
        return self._type_labels

    def get_type_id(self, type_label: str) -> Optional[str]:
        """Returns Wikdata type id in the "Q..." format for a given string label
        of the Wikidata type.

        Args:
            type_label: Wikidata type in string label format. For example,
                "district of England".

        Returns:
            Wikidata KB id for the given type (For example, Q349084) if found,
                return None otherwise.
        """
        return (
            self._label_type[type_label]
            if type_label in self._label_type
            else None
        )

    def analyze_dataset(self, split: str):
        """Performs an analysis of the Wikidata SMART dataset.

        Args:
            split: Data split ("train" or "test")
        """
        dataset = Dataset("wikidata", split)
        dataset_types = dataset.get_unique_types(lowercase=True)
        wd_types = set(self.get_type_labels().values())
        print(f"#types in dataset:       {len(dataset_types)}")
        print(f"#types in Wikidata dump: {len(wd_types)}")

        # Check whether all types in the dataset may be found.
        missing_types = dataset_types - wd_types
        found_types = dataset_types.intersection(wd_types)

        # Found/missing types.
        print(f"#types found:   {len(found_types)}")
        print(f"#types missing: {len(missing_types)}")
        if len(missing_types) > 0:
            print(f"  For example: {', '.join(list(missing_types)[:10])} ...")

        # TODO(KB): Check where types lie in the hierarchy

    def is_type(self, type_id: str) -> bool:
        """Checks if a type exists in the type system.

        Args:
            type_id: Type ID.

        Returns:
            True if the type exists in the type system, otherwise False.
        """
        return True if type_id in self._types else False

    def get_depth(self, type_id) -> int:
        """Returns the depth level of a given type.

        Args:
            type_id: Type ID.

        Returns:
            Depth in the hierarchy counted from root (which has a depth of 0).
        """
        raise NotImplementedError

    def get_max_depth(self) -> int:
        """Returns the maximum depth of the type hierarchy."""
        raise NotImplementedError

    def get_entity_types(self, entity_id: str) -> Set[str]:
        """Return the set of types assigned to a given entity.

        Arsg:
            entity_id: Entity ID in the format "Q2".

        Returns:
            Set of type IDs (empty set if there are no types assigned).
        """
        return self._entity_types.get(entity_id, set())

    def get_type_entities(self, type_label: str) -> Set[str]:
        """Returns the set of entities belonging to a given type (string label
        format). (It considers only direct type-entity assignments, and
        transitive ones via sub-types.)

        Args:
            type_label: Type in the form of string label.

        Returns:
            Set of entity IDs (empty set if no entities for the type).
        """
        type_id = self.get_type_id(type_label)
        return self._type_entities.get(type_id, set()) if type_id else set()

    def get_supertype(self, type_id) -> Optional[str]:
        """Returns the parent type of a given type.

        Args:
            type_id: Type ID.

        Returns:
            ID of the parent type of None (if type_id is ROOT).
        """
        raise NotImplementedError

    def get_subtypes(self, type_id) -> Set[str]:
        """Returns the set of (direct) subtypes for a given type.

        Args:
            type_id: Type ID.

        Returns:
            Set of type IDs (empty if leaf type).
        """
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dir", type=str, help="Path to Wikidata extracts"
    )
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        help="Split (train or test)",
        choices=["train", "test"],
    )
    args = parser.parse_args()

    if args.dir:
        wd = Wikidata(extracts_dir=args.dir)
        wd.analyze_dataset(args.split)
