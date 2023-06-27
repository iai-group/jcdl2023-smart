"""This module has file utils and json utils. For reading and dumping json or
text files."""

import json
import pandas as pd
import os
import bz2
import gzip


def read_json(file):
    """Reads task json into a pandas dataframe."""

    df = pd.read_json(file)
    clean_df = df.dropna()
    return clean_df


def dump_json(data, file_name):
    """Dumps dataframe into pretty formatted json format similar to the task
    data."""

    df_json_str = data.to_dict('records')
    json_str = json.dumps(df_json_str, indent=4)
    with open(file_name, 'w') as json_file:
        json_file.write(json_str)


def read_text_file(file):
    """Reads text file and returns as an array."""

    texts = []
    with open(file, 'r') as f:
        for line in f:
            texts.append(line.replace('\n', ''))
    return texts


def open_file_by_type(file_name, mode="r"):
    """Opens file (gz/text) and returns the handler.
    :param file_name: NTriples file
    :return: handler to the file
    """
    file_name = os.path.expanduser(
        file_name
    )  # expands '~' to the absolute home dir
    if file_name.endswith("bz2"):
        return bz2.open(file_name, mode)
    elif file_name.endswith("gz"):
        return gzip.open(file_name, mode, encoding="utf-8")
    else:
        return open(file_name, mode, encoding="utf-8")


def read_file_as_list(filename):
    """Reads in non-empty lines from a textfile (which may be gzipped/bz2ed)
        and returns it as a list.
    Args filename:
    """
    with open_file_by_type(filename) as f:
        return [ll for ll in (line.strip() for line in f) if ll]
