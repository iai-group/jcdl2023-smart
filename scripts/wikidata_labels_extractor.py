"""This script dumps Wikidata item ids and their description and labels in
tsv files. It is assumed to be run on gustav1."""

import argparse

from smart.utils.wikidata import Wikidata

WIKIDATA_DUMP_SAMPLE_DIR = "data/wikidata/dump_sample"
WIKIDATA_EXTRACTS_SAMPLE_DIR = "data/wikidata/extracts_sample"

# Full Wikidata dump (20210519) and extracts on gustav1
WIKIDATA_DUMP_DIR = "/data/collections/wikidata/20210519/"
WIKIDATA_EXTRACTS_DIR = "/data/collections/wikidata/20210519/extracts"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        '--full',
        action='store_true',
        help="Use Wikidata full dump folder.",
    )
    args = parser.parse_args()

    if args.full:
        dump_dir = WIKIDATA_DUMP_DIR
        extracts_dir = WIKIDATA_EXTRACTS_DIR
    else:
        dump_dir = WIKIDATA_DUMP_SAMPLE_DIR
        extracts_dir = WIKIDATA_EXTRACTS_SAMPLE_DIR

    wikidata = Wikidata(
        dump_dir=dump_dir,
        extracts_dir=extracts_dir,
    )

    descriptions = wikidata.get_item_predicate_values("schema:description")
    with open(f"{extracts_dir}/{Wikidata.EXTRACTS_FILE_DESCRIPTION}", "w") as f:
        for item, description in descriptions.items():
            f.write(f"{item}\t{description}\n")

    labels = wikidata.get_item_predicate_values("rdfs:label")
    with open(f"{extracts_dir}/{Wikidata.EXTRACTS_FILE_LABEL}", "w") as f:
        for item, description in labels.items():
            f.write(f"{item}\t{description}\n")
