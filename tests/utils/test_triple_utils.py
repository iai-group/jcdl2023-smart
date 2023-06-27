"""Tests for URI utils."""

import pytest

from smart.utils.triple_utils import get_triple, prefix_uri


@pytest.mark.parametrize(
    "uri,prefixed_uri",
    [
        ("http://dbpedia.org/ontology/Athlete", "dbo:Athlete"),
        ("<http://dbpedia.org/ontology/Athlete>", "<dbo:Athlete>"),
        (
            "http://dbpedia.org/ontology/PopulatedPlace/areaMetro",
            "dbo:PopulatedPlace/areaMetro",
        ),
        ("http://www.w3.org/2000/01/rdf-schema#subClassOf", "rdfs:subClassOf"),
        ("<http://www.wikidata.org/entity/Q6256>", "<wd:Q6256>"),
        ("http://www.wikidata.org/prop/direct/P31", "wdt:P31"),
    ],
)
def test_prefix_uri(uri, prefixed_uri):
    assert prefix_uri(uri) == prefixed_uri


@pytest.mark.parametrize(
    "line,s,p,o",
    [
        ("<a> <b> <c> .", "<a>", "<b>", "<c>"),
        ("<a> <b> \"c\"@en .", "<a>", "<b>", "\"c\"@en"),
        ("<a> <b> \"c\"^^<type> .", "<a>", "<b>", "\"c\"^^<type>"),
        (
            (
                "<http://dbpedia.org/resource/Autism> "
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                "<http://dbpedia.org/ontology/Disease> ."
            ),
            "<dbr:Autism>",
            "<rdf:type>",
            "<dbo:Disease>",
        ),
        (
            (
                "<http://dbpedia.org/resource/A> "
                "<http://dbpedia.org/property/map1char> "
                "\"81\"^^<http://www.w3.org/2001/XMLSchema#integer> ."
            ),
            "<dbr:A>",
            "<dbp:map1char>",
            "\"81\"^^<http://www.w3.org/2001/XMLSchema#integer>",
        ),
        (
            (
                "<http://www.wikidata.org/entity/Q865> "
                "<http://www.wikidata.org/prop/direct/P31> "
                "<http://www.wikidata.org/entity/Q6256> ."
            ),
            "<wd:Q865>",
            "<wdt:P31>",
            "<wd:Q6256>",
        ),
        (
            (
                "<http://www.wikidata.org/entity/Q26> "
                "<http://schema.org/name> \"Northern Ireland\"@en ."
            ),
            "<wd:Q26>",
            "<schema:name>",
            "\"Northern Ireland\"@en",
        ),
    ],
)
def test_get_triple_keep_angle_brackets(line, s, p, o):
    assert get_triple(line, remove_angle_brackets=False) == (s, p, o)


@pytest.mark.parametrize(
    "line,s,p,o",
    [
        ("<a> <b> <c> .\n", "a", "b", "c"),
        ("<a> <b> \"c\"@en .", "a", "b", "\"c\"@en"),
        ("<a> <b> \"c\"^^<type> .", "a", "b", "\"c\"^^<type>"),
    ],
)
def test_get_triple_remove_angle_brackets(line, s, p, o):
    assert get_triple(line) == (s, p, o)


@pytest.mark.parametrize("line", ["#comment", "<a> <b>", "<a> <b> <c>"])
def test_get_triple_non_parseable(line):
    with pytest.raises(ValueError):
        get_triple(line)
