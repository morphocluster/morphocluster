#!/usr/bin/env python3

import h5py
import pandas as pd

features_fn = "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-02-06-12-39-56/split-2/collection_unlabeled.h5"
#features_1M_fn = "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-02-06-12-39-56/split-2/collection_unlabeled_1M.h5"
features_1k_fn = "/data1/mschroeder/NoveltyDetection/Results/CrossVal/2018-02-06-12-39-56/split-2/collection_unlabeled_1k.h5"
objids_fn = "/data1/mschroeder/NoveltyDetection/Results/CV-Clustering/2018-09-10-12-08-49/split-2/objids.csv"

with h5py.File(features_fn, "r", libver="latest") as f_features:
    dataset = {t: f_features[t][:]
               for t in ("features", "objids", "targets")}

objids = pd.read_csv(objids_fn, index_col=False, header=None, dtype="int32", squeeze=True)

objids = objids.sample(1000)

dataset_objids = pd.Series(dataset["objids"])
dataset_selector = dataset_objids.isin(objids)

with h5py.File(features_1k_fn, "w", libver="earliest") as f_features:
    for t in ("features", "objids", "targets"):
        f_features.create_dataset(t, data=dataset[t][dataset_selector])
