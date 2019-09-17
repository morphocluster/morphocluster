import itertools

import h5py
import pandas as pd
import tqdm
from sqlalchemy.sql.expression import bindparam

from morphocluster import models
from morphocluster.extensions import database


def load_object_collection(collection, batch_size=1000):
    """
    Load a collection of objects into the database.
    """

    conn = database.get_connection()

    if not isinstance(collection, pd.DataFrame):
        print("Reading {}...".format(collection))
        collection = pd.read_csv(
            collection,
            header=None,
            names=["object_id", "path", "label"],
            usecols=["object_id", "path"],
            dtype=str
        )

    with conn.begin():
        collection_iter = collection.itertuples()
        progress = tqdm.tqdm(total=len(collection))
        while True:
            chunk = tuple(itertools.islice(collection_iter, batch_size))
            if not chunk:
                break
            conn.execute(
                models.objects.insert(),  # pylint: disable=no-value-for-parameter
                [row._asdict() for row in chunk]
            )

            progress.update(len(chunk))
        progress.close()
        print("Done.")


def load_object_features(features_fns, batch_size=1000):
    """
    Load object features from an HDF5 file.
    """
    for features_fn in features_fns:
        print("Loading {}...".format(features_fn))
        with h5py.File(features_fn, "r", libver="latest") as f_features, database.engine.begin() as conn:
            object_ids = f_features["objids"]
            vectors = f_features["features"]

            stmt = (
                models.objects.update()  # pylint: disable=no-value-for-parameter
                .where(models.objects.c.object_id == bindparam('_object_id'))
                .values({
                        'vector': bindparam('vector')
                        })
            )

            progress = tqdm.tqdm(total=len(object_ids))
            obj_iter = iter(zip(object_ids, vectors))
            while True:
                chunk = tuple(itertools.islice(obj_iter, batch_size))
                if not chunk:
                    break
                conn.execute(stmt, [{"_object_id": str(object_id), "vector": vector} for (
                    object_id, vector) in chunk])

                progress.update(len(chunk))
            progress.close()
            print("Done.")
