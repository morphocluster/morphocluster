import csv
import fnmatch
import os.path
from typing import Optional
import zipfile

import chardet
import click
import pandas as pd

from morphocluster.processing.extract_features import extract_features
from morphocluster.processing.recluster import Recluster


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


@main.command()
@click.argument(
    "archive_fn",
    type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
)
@click.option(
    "--encoding"
)
@click.option(
    "--delimiter"
)
def fix_ecotaxa(archive_fn, encoding, delimiter: Optional[str]):
    """Fix EcoTaxa-style archives to be processable by MorphoCluster."""

    if delimiter is not None:
        delimiter = delimiter.replace("\\t", "\t")

    with zipfile.ZipFile(archive_fn, "a") as zf:
        if "index.csv" in zf.namelist():
            print("Archive already contains index.csv")
            return

        # Find index file
        index_pat = "ecotaxa_*"
        index_fns = [
            fn
            for fn in zf.namelist()
            if fnmatch.fnmatch(os.path.basename(fn), index_pat)
        ]

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

            if encoding is None or delimiter is None:
                with zf.open(index_fn) as fp:
                    sample = fp.read(8000)
                    if encoding is None:
                        encoding = chardet.detect(sample)["encoding"]
                        print("Detected encoding:", encoding)
                    sample = sample.decode(encoding)
                    if delimiter is None:
                        dialect = csv.Sniffer().sniff(sample, [",", "\t", ";"])
                        delimiter = dialect.delimiter
                        print("Detected delimiter:", repr(delimiter))

            with zf.open(index_fn) as fp:
                dataframe = pd.read_csv(
                    fp,
                    encoding=encoding,
                    delimiter=delimiter,
                    dtype=str,
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
@click.option("--batch-size", type=int, default=512)
# @click.option("--num-workers", type=int, default=0)
def features(model_fn, archive_fn, output_fn, normalize, batch_size):
    """
    Extract features from an EcoTaxa export (or compatible) archive.
    """

    extract_features(model_fn, archive_fn, output_fn, normalize, batch_size)


@main.command()
@click.argument(
    "features_fns", type=click.Path(exists=True, readable=True), nargs=-1, required=True
)
@click.argument("result_fn", type=click.Path(exists=False, writable=True), nargs=1)
@click.option(
    "--tree", "tree_fn", type=click.Path(exists=True, readable=True), default=None
)
@click.option("--min-cluster-size", type=int, default=128)
@click.option("--min-samples", type=int, default=1)
@click.option("--method", type=click.Choice(["eom", "leaf"]), default="leaf")
@click.option("--sample-size", type=int, default=None)
@click.option("--pca", type=int, default=None)
def cluster(
    features_fns, result_fn, tree_fn, min_cluster_size, min_samples, method, sample_size, pca
):
    rc = Recluster()

    for fn in features_fns:
        rc.load_features(fn)

    if tree_fn is not None:
        rc.load_tree(tree_fn)

    rc.cluster(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        sample_size=sample_size,
        pca=pca,
    )

    rc.save(result_fn)
