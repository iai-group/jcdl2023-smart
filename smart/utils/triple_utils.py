"""Utilities for working with URIs and NTriple files."""

from typing import Tuple


def prefix_uri(uri: str) -> str:
    """Prefixes an URI with a few common namespaces.

    URI may be provided either with or without angle brackets.

    Args:
        uri: URI.

    Returns: prefixed URI.
    """
    mapping = {
        "http://dbpedia.org/ontology/": "dbo",
        "http://dbpedia.org/resource/": "dbr",
        "http://dbpedia.org/property/": "dbp",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
        "http://www.w3.org/2002/07/owl#": "owl",
        "http://schema.org/": "schema",
        "http://www.wikidata.org/entity/": "wd",  # Wikidata items
        "http://www.wikidata.org/prop/direct/": "wdt",  # Wikidata properties
    }
    for prefix, shortened in mapping.items():
        offset = 1 if uri.startswith("<") else 0
        if uri[offset:].startswith(prefix):
            return f"{uri[:offset]}{shortened}:{uri[len(prefix)+offset:]}"
    return uri


def get_triple(
    line: str, remove_angle_brackets: bool = True
) -> Tuple[str, str, str]:
    """Extracts a SPO triple from a line from a TTL file.

    URIs are prefixed and are enclosed in angle brackets.
    Literals are enclosed in "..." followed optionally by a language tag
    (e.g., @en) or type information (e.g., ^^<http://...>).

    Args:
        line: Line in either "<subject> <predicate> <object> ." or
            "<subject> <predicate> "object"[optional] ." format.
            Line might contain trailing new line character.
        remove_angle: Whether to remove angle brackets around URLs
            (default: True).

    Returns: (subject, predicate, object) tuple, with URIs prefixed.

    Raises:
        ValueError: If triple cannot be extracted from line.
    """
    line = line.rstrip()
    if not line.startswith("<") or not line.endswith("."):
        raise ValueError

    # Find all spaces in the line; there should be at least 3.
    # <subject> <predicate> object .
    sep = [i for i in range(len(line)) if line[i] == " "]
    if len(sep) < 3:
        raise ValueError

    subj = prefix_uri(line[: sep[0]])
    pred = prefix_uri(line[sep[0] + 1 : sep[1]])
    obj = line[sep[1] + 1 : sep[-1]]
    if obj.startswith("<"):
        obj = prefix_uri(obj)

    # Remove angle brackets.
    if remove_angle_brackets:
        subj = subj[1:-1]
        pred = pred[1:-1]
        if obj.startswith("<"):
            obj = obj[1:-1]

    return subj, pred, obj
