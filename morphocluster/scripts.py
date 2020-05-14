import fnmatch
import os.path
import zipfile

import click
import pandas as pd

from morphocluster.processing.extract_features import extract_features


@click.group()
def main():
    """
    MorphoCluster preprocessing.

    Archive format
    ==============
    MorphoCluster expects a ZIP archive with an `index.csv`
    with columns `object_id` and `path`.
    `path` is the name of the corresponding image in the archive.
    """
    pass


def ecotaxa_fix_types(dataframe):
    first_row = dataframe.iloc[0]

    num_cols = []
    for c, v in first_row.items():
        if v == "[f]":
            num_cols.append(c)
        elif v == "[t]":
            continue
        else:
            # If the first row contains other values than [f] or [t],
            # it is not a type header and the dataframe doesn't need to be changed.
            return dataframe

    dataframe = dataframe.iloc[1:]

    dataframe[num_cols] = dataframe[num_cols].apply(
        pd.to_numeric, errors="coerce", axis=1
    )

    return dataframe


import chardet, csv


@main.command()
@click.argument(
    "archive_fn",
    type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
)
def fix_ecotaxa(archive_fn):
    """Fix EcoTaxa-style archives to be processable by MorphoCluster."""

    with zipfile.ZipFile(archive_fn, "a") as zf:
        if "index.csv" in zf.namelist():
            print("Archive already contains index.csv")
            return

        # Find index file
        index_pat = "ecotaxa_*"
        index_fns = fnmatch.filter(zf.namelist(), index_pat)

        if not index_fns:
            raise ValueError(
                "No archive member matches the pattern '{}'".format(index_pat)
            )

        index = []
        for index_fn in index_fns:
            index_dir = os.path.dirname(index_fn)

            if index_dir:
                index_dir = index_dir + "/"

            print(f"Loading {index_fn}...")

            with zf.open(index_fn) as fp:
                sample = fp.read(8000)
                encoding = chardet.detect(sample)["encoding"]
                sample = sample.decode(encoding)
                dialect = csv.Sniffer().sniff(sample, [",", "\t", ";"])
                print("delimiter", repr(dialect.delimiter))

            with zf.open(index_fn) as fp:
                dataframe = pd.read_csv(
                    fp,
                    encoding=encoding,
                    dtype=str,
                    dialect=dialect,
                    usecols=["object_id", "img_file_name"],
                )

            dataframe = ecotaxa_fix_types(dataframe)
            dataframe["img_file_name"] = index_dir + dataframe["img_file_name"]

            index.append(dataframe)

        # Concatenate and rename column
        index = pd.concat(index).rename(columns={"img_file_name": "path"})

        # Write to archive
        print("Writing result...")
        zf.writestr("index.csv", index.to_csv(index=False))


@main.command()
@click.argument("model_fn", type=click.Path(exists=True, dir_okay=False))
@click.argument("archive_fn", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_fn", type=click.Path(exists=False, dir_okay=False))
@click.option("--normalize/--no-normalize", default=True)
def features(model_fn, archive_fn, output_fn, normalize):
    """
    Extract features from an EcoTaxa export (or compatible) archive.
    """

    extract_features(model_fn, archive_fn, output_fn, normalize)


@main.command()
def cluster():
    ...
