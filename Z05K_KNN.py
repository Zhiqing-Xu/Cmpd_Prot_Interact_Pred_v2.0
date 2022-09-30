#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#--------------------------------------------------#


###################################################################################################################
###################################################################################################################
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
#--------------------------------------------------#
import torch
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#--------------------------------------------------#
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
#--------------------------------------------------#
import pickle
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#



###################################################################################################################
###################################################################################################################
import json
import math
import argparse
import itertools

from tqdm import tqdm
from scipy import stats
from typing import List
from pathlib import Path
from functools import partial
#--------------------------------------------------#
import multiprocessing as mp
#--------------------------------------------------#
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
#--------------------------------------------------#
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
#--------------------------------------------------#
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

###################################################################################################################
###################################################################################################################
class KMERFeaturizer:
    """KMERFeaturizer."""

    def __init__(self,
                 ngram_min: int = 2,
                 ngram_max: int = 4,
                 unnormalized: bool = False,):

        # __init__.Args:
        # ngram_min (int): ngram_min
        # ngram_max (int): ngram_max
        # unnormalized (bool): normalize
        # kwargs: kwargs

        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.normalize = not (unnormalized)

        self.vectorizer = CountVectorizer(ngram_range=(self.ngram_min, self.ngram_max), analyzer="char")

        # If false, fit, otherwise just run
        self.is_fit = False

    def featurize(self, seqs_list: List[str]) -> List[np.ndarray]:

        # On first settles on fixed kmer components
        # Args   :  seqs_list (List[str]): seqs_list containing strings of smiles
        # Returns:  np.ndarray: of features

        if not self.is_fit:
            self.vectorizer.fit(seqs_list)
            self.is_fit = True
        output = self.vectorizer.transform(seqs_list)
        output = np.asarray(output.todense())

        # If this is true, normalize the sequence
        if self.normalize:
            output = output / output.sum(1).reshape(-1, 1)

        return list(output)


class MorganFeaturizer:
    """MorganFeaturizer."""

    def __init__(self):
        pass

    def _mol_to_fp(self, mol: Chem.Mol) -> np.ndarray:
        """Convert mol to fingerprint"""
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def featurize(self, mol_list: List[str]) -> List[np.ndarray]:
        """featurize."""
        ret = []
        for mol in tqdm(mol_list):
            mol = Chem.MolFromSmiles(mol)
            ret.append(self._mol_to_fp(mol))
        return ret


class KNNModel:
    """KNNModel"""

    def __init__(self, n=3, seqs_dist_weight=5, comp_dist_weight=1, **kwargs):
        self.n = n
        self.seqs_dist_weight = seqs_dist_weight
        self.comp_dist_weight = comp_dist_weight

    def cosine_dist(self, train_objs, test_objs):
        """compute cosine_dist"""
        numerator = train_objs[:, None, :] * test_objs[None, :, :]
        numerator = numerator.sum(-1)

        norm = lambda x: (x**2).sum(-1)**(0.5)

        denominator = norm(train_objs)[:, None] * norm(test_objs)[None, :]
        denominator[denominator == 0] = 1e-12
        cos_dist = 1 - numerator / denominator

        return cos_dist

    def fit(self, train_seqs, train_cmpd, train_vals, val_seqs, val_cmpd,
            val_vals) -> None:
        self.train_seqs = train_seqs
        self.train_cmpd = train_cmpd
        self.train_vals = train_vals

    def predict(self, test_seqs, test_cmpd) -> np.ndarray:
        # Compute test dists
        test_seqs_dists = self.cosine_dist(self.train_seqs, test_seqs)
        test_comp_dists = self.cosine_dist(self.train_cmpd, test_cmpd)
        total_dists = (self.seqs_dist_weight * test_seqs_dists +
                       self.comp_dist_weight * test_comp_dists)

        smallest_dists = np.argsort(total_dists, 0)

        top_n = smallest_dists[:self.n, :]
        ref_vals = self.train_vals[top_n]
        mean_preds = np.mean(ref_vals, 0)
        return mean_preds


def shuffle_dataset(dataset, seed):
    """shuffle_dataset."""
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    """split_dataset."""
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def single_trial(model_params, train_data, dev_data, test_data):
    """Conduct a single trial with given model params"""

    # Unpack data
    train_seqs_feats, train_sub_feats, train_vals = train_data
    dev_seqs_feats, dev_sub_feats, dev_vals = dev_data
    test_seqs_feats, test_sub_feats, test_vals = test_data

    # Create model
    knn_model = KNNModel(**model_params)

    knn_model.fit(
        train_seqs_feats,
        train_sub_feats,
        train_vals,
        dev_seqs_feats,
        dev_sub_feats,
        dev_vals,
    )

    # Conduct analysis on val and test set
    outputs = {}
    outputs.update(model_params)
    for dataset, seqs_feats, sub_feats, targs in zip(
        ["val", "test"],
        [dev_seqs_feats, test_seqs_feats],
        [dev_sub_feats, test_sub_feats],
        [dev_vals, test_vals],
    ):

        inds = np.arange(len(seqs_feats))
        num_splits = min(50, len(inds))
        ars = np.array_split(inds, num_splits)
        ar_vec = []
        for ar in ars:
            test_preds = knn_model.predict(seqs_feats[ar], sub_feats[ar])
            ar_vec.append(test_preds)

        test_preds = np.concatenate(ar_vec)

        # Evaluation
        true_vals_corrected = np.log10(np.power(2, targs))
        predicted_vals_corrected = np.log10(np.power(2, test_preds))
        
        SAE = np.abs(predicted_vals_corrected - true_vals_corrected)
        MAE = np.mean(SAE)
        r2 = r2_score(predicted_vals_corrected, true_vals_corrected)
        RMSE = np.sqrt((SAE**2).mean())

        results = {
            f"{dataset}_mae": MAE,
            f"{dataset}_RMSE": RMSE,
            f"{dataset}_r2": r2
        }

        outputs.update(results)
    return outputs



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == "__main__":
    print()
