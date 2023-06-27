"""Data loader utility class for DBpedia, implementing the TypeSystem interface.

Note that `instance_types_en.ttl` contains only the most specific types, while
`instance_types_transitive_en.ttl` contains only the path to the path to the
most specific types.  Therefore, both files need to be loaded.

Additionally, the DBpedia type hierarchy is assumed to be provided as a tsv
file.

Also, there are some data discrepancies around equivalence classes that are
fixed in the loader:
  - dbo:Location => dbo:Place
  - dbo:Wikidata:Q11424 => dbo:Film
"""

import bz2
import errno
import os
import sys

from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Iterable, Optional, Set, Tuple

from smart.utils.type_system import TypeSystem
import smart.utils.triple_utils as triple_utils


class DBpedia(TypeSystem):

    ROOT_TYPE = "owl:Thing"

    DBPEDIA_TYPE_FILES = [
        "instance_types_en.ttl",
        "instance_types_transitive_en.ttl",
    ]

    DBPEDIA_HIERARCHY_FILE = "data/dbpedia/dbpedia_types.tsv"

    def __init__(
        self,
        dbpedia_dump_path: str,
        dbpedia_hierarchy_file: str = DBPEDIA_HIERARCHY_FILE,
    ) -> None:
        """Handles DBpedia type data.

        Args:
            dbpedia_dump_path: Location of the DBpedia dump where type files are
                located (expected in .ttl or .ttl.bz2 format).
            dbpedia_hierarchy_file: File path of the DBpedia hierarchy in tsv
                format.
        """
        self._dump_path = dbpedia_dump_path
        # Check if the files we need may be found on dump path.
        for filename in self.DBPEDIA_TYPE_FILES:
            filepath = self._get_dump_file_path(filename)
            if not filepath:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename
                )

        self._max_depth = 0  # Maximum depth of the hierarchy
        self._types = defaultdict(dict)  # Type depth, subtypes, supertypes
        self._load_types_from_hierarchy_file(dbpedia_hierarchy_file)
        self._type_entities = defaultdict(set)  # Set of entities for each type
        self._entity_types = defaultdict(set)  # Set of types for each entity
        # Load entity-type assignments from dumpfile.
        for entity, entity_type in self._load_entity_types_from_dump():
            self._type_entities[entity_type].add(entity)
            self._entity_types[entity].add(entity_type)

    def _load_types_from_hierarchy_file(self, filepath: str) -> None:
        """Loads the DBpedia type hierarchy from file.

        Args:
            filepath: Path of the hierarchy tsv file.
        """
        self._types[self.ROOT_TYPE] = {"depth": 0, "subtypes": set()}
        with open(filepath, "r") as tsv_file:
            next(tsv_file)  # Skip header row
            for line in tsv_file:
                fields = line.rstrip().split("\t")
                type_name, depth, parent_type = (
                    fields[0],
                    int(fields[1]),
                    fields[2],
                )
                self._types[type_name]["supertype"] = parent_type
                self._types[type_name]["depth"] = depth
                if "subtypes" not in self._types[parent_type]:
                    self._types[parent_type]["subtypes"] = set()
                self._types[parent_type]["subtypes"].add(type_name)
                self._max_depth = max(depth, self._max_depth)

    def _get_dump_file_path(self, filename: str) -> Optional[str]:
        """Returns full path for a given dump file, with both original filename
        and .bz2 extension checked.

        Args:
            filename: Dump file name (with .ttl extension).

        Returns: Full file path.
        """
        filepath = os.path.join(self._dump_path, filename)
        print(f"Checking {filepath}")
        if os.path.isfile(filepath):
            return filepath
        if os.path.isfile(filepath + ".bz2"):
            return filepath + "bz2"
        return None

    def _load_entity_types_from_dump(self) -> Iterable[Tuple[str, str]]:
        """Loads entity-type assignments from DBpedia dump.
        Works with both uncompressed .ttl and compressed .bz2 files.

        Yields:
            Sequence of entity and type tuples.
            Entities are represented by their DBpedia ID, without prefixing
            (e.g., 'Stavanger', 'Albert_Einstein').
        """
        for filename in self.DBPEDIA_TYPE_FILES:
            print(f"Loading entity-type assignments from {filename}...")
            filepath = self._get_dump_file_path(filename)
            is_bz2 = filepath.endswith(".bz2")
            f = bz2.BZ2File(filepath, "r") if is_bz2 else open(filepath, "r")

            with f:
                # Total value (for progress bar) hardcoded for DBpedia-2016-10.
                total = 31254272 if 'transitive' in filename else 5150434
                for i, line in enumerate(tqdm(f, total=total, file=sys.stdout)):
                    if is_bz2:
                        line = line.decode("utf-8")
                    if not line.startswith("<http://dbpedia.org"):
                        continue
                    try:
                        subj, pred, obj = triple_utils.get_triple(line)
                    except ValueError:
                        continue

                    # Only consider types from DBpedia Ontology.
                    if not obj.startswith("dbo:"):
                        continue
                    # Replacing equivalence classes to canonical ones.
                    if obj == "dbo:Location":
                        obj = "dbo:Place"
                    elif obj == "dbo:Wikidata:Q11424":
                        obj = "dbo:Film"

                    # Entities are represented by their DBpedia ID (without dbr:
                    # prefix).
                    subj = subj.split(':')[-1]

                    yield subj, obj

    def get_types(self) -> Set[str]:
        """Returns the set of types in DBpedia.

        Returns:
            Set of type IDs.
        """
        return self._types.keys()

    def is_type(self, type_id: str) -> bool:
        """Checks if a type exists in the type system.

        Args:
            type_id: Type ID.

        Returns:
            True if the type exists in the type system, otherwise False.
        """
        return type_id in self._types

    def get_depth(self, type_id) -> int:
        """Returns the depth level of a given type.

        Args:
            type_id: Type ID.

        Returns:
            Depth in the hierarchy counted from root (which has a depth of 0).
        """
        return self._types.get(type_id, {}).get("depth")

    def get_max_depth(self) -> int:
        """Returns the maximum depth of the type hierarchy."""
        return self._max_depth

    def get_entity_types(self, entity_id: str) -> Set[str]:
        """Return the set of types assigned to a given entity.

        Arsg:
            entity_id: Entity ID.

        Returns:
            Set of type IDs (empty set if there are no types assigned).
        """
        return self._entity_types.get(entity_id, set())

    def get_type_entities(self, type_id: str) -> Set[str]:
        """Returns the set of entities belonging to a given type.
        (It considers only direct type-entity assignments, and transitive ones
        via sub-types.)

        Args:
            type_id: Type ID.

        Returns:
            Set of entity IDs (empty set if no entities for the type).
        """
        return self._type_entities.get(type_id, set())

    def get_supertype(self, type_id) -> Optional[str]:
        """Returns the parent type of a given type.

        Args:
            type_id: Type ID.

        Returns:
            ID of the parent type of None (if type_id is ROOT).
        """
        return self._types.get(type_id, {}).get("supertype")

    def get_subtypes(self, type_id) -> Set[str]:
        """Returns the set of (direct) subtypes for a given type.

        Args:
            type_id: Type ID.

        Returns:
            Set of type IDs (empty if leaf type).
        """
        return self._types.get(type_id, {}).get("subtypes")
