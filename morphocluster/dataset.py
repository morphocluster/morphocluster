import itertools
import os.path
import shutil
import zipfile

import h5py
import pandas as pd
import tqdm
from flask import current_app
from sqlalchemy.sql.expression import bindparam

from morphocluster import models
from morphocluster.extensions import database
from morphocluster.models import datasets
from morphocluster.project import Project
from sqlalchemy.engine import ResultProxy


class Dataset:
    """An abstraction for a dataset."""

    path_fmt = "{data_dir}/{dataset_id:d}"

    @staticmethod
    def create(name, owner):
        """Create a dataset.

        This creates a new row in datasets as well as a new partition objects_{dataset_id}.
        """

        connection = database.get_connection()

        with connection.begin():
            stmt = datasets.insert({"name": name, "owner": owner})

            result = connection.execute(stmt)
            dataset_id = result.inserted_primary_key[0]

            # Create partition for objects
            stmt = "CREATE TABLE objects_{dataset_id} PARTITION OF objects FOR VALUES IN ({dataset_id})".format(
                dataset_id=dataset_id
            )
            result = connection.execute(stmt)

        return Dataset(dataset_id)

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

    @property
    def path(self):
        return self.path_fmt.format(
            data_dir=current_app.config["DATA_DIR"], dataset_id=self.dataset_id
        )

    def load_objects(self, archive_fn, batch_size=1000):
        """Load an archive of objects into the database."""

        conn = database.get_connection()

        dst_root = self.path

        print("Loading {}...".format(archive_fn))
        with conn.begin(), zipfile.ZipFile(archive_fn) as zf:
            index = pd.read_csv(zf.open("index.csv"), usecols=["object_id", "path"])
            index_iter = index.itertuples()
            progress = tqdm.tqdm(total=len(index))
            while True:
                chunk = tuple(
                    row._asdict() for row in itertools.islice(index_iter, batch_size)
                )
                if not chunk:
                    break
                conn.execute(
                    models.objects.insert(),  # pylint: disable=no-value-for-parameter
                    [dict(row, dataset_id=self.dataset_id) for row in chunk],
                )

                for row in chunk:
                    dst_fn = os.path.join(dst_root, row["path"])
                    dst_dir = os.path.dirname(dst_fn)
                    os.makedirs(dst_dir, exist_ok=True)
                    zf.extract(row["path"], dst_fn)

                progress.update(len(chunk))
            progress.close()
            print("Done.")

    def load_object_features(self, features_fn, batch_size=1000):
        """Load object features from an HDF5 file."""

        conn = database.get_connection()

        print("Loading {}...".format(features_fn))
        with h5py.File(features_fn, "r") as f_features, conn.begin():
            object_ids = f_features["object_id"]
            vectors = f_features["features"]

            stmt = (
                models.objects.update()  # pylint: disable=no-value-for-parameter
                .where(
                    (models.objects.c.object_id == bindparam("_object_id"))
                    & (models.objects.c.dataset_id == self.dataset_id)
                )
                .values({"vector": bindparam("vector")})
            )

            progress = tqdm.tqdm(total=len(object_ids))
            obj_iter = iter(zip(object_ids, vectors))
            while True:
                chunk = tuple(itertools.islice(obj_iter, batch_size))
                if not chunk:
                    break
                result: ResultProxy = conn.execute(
                    stmt,
                    [
                        {"_object_id": str(object_id), "vector": vector}
                        for (object_id, vector) in chunk
                    ],
                )

                progress.update(len(chunk))
            progress.close()
            print("Done.")

    def create_project(self, name) -> Project:
        return Project.create(name, self.dataset_id)

    def remove(self):
        """Remove the dataset and all belonging entries from the database and filesystem."""
        connection = database.get_connection()

        with connection.begin():

            # Drop partition for objects
            stmt = "DROP TABLE objects_{dataset_id} CASCADE".format(
                dataset_id=self.dataset_id
            )
            connection.execute(stmt)

            # Delete entry
            stmt = datasets.delete(datasets.c.dataset_id == self.dataset_id)
            connection.execute(stmt)

            # Delete filesystem data
            shutil.rmtree(self.path)

        self.dataset_id = None
