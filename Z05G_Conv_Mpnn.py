#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
from tokenize import Double
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
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
        try:
            os.chdir(os.path.dirname(__file__))
            print('CurrentDir: ', os.getcwd())
        except:
            print("Problems with navigating to the file dir.")
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
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.weight_norm import weight_norm
#--------------------------------------------------#
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
#--------------------------------------------------#
import pickle
import random
import threading
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection, Iterator
#--------------------------------------------------#
from functools import reduce



#--------------------------------------------------#
from Z05_utils import *
from Z05_split_data import *

#------------------------------
from ZX02_nn_utils import StandardScaler
from ZX02_nn_utils import initialize_weights
from ZX02_nn_utils import get_activation_function
from ZX02_nn_utils import build_optimizer, build_lr_scheduler
from ZX02_nn_utils import *

from ZX03_nn_args import TrainArgs
from ZX04_funcs import onek_encoding_unk
from ZX05_loss_functions import get_loss_func

#------------------------------
from Z05G_Cpd_Data import *


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM"""Yb.      db  MMP""MM""YMM  db     `7MMF'        .g8""8q.      db     `7MM"""Yb. `7MM"""YMM  `7MM"""Mq.                                M      #
#    MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM        .dP'    `YM.   ;MM:      MM    `Yb. MM    `7    MM   `MM.                               M      #
#    MM     `Mb  ,V^MM.     MM     ,V^MM.     MM        dM'      `MM  ,V^MM.     MM     `Mb MM   d      MM   ,M9                                M      #
#    MM      MM ,M  `MM     MM    ,M  `MM     MM        MM        MM ,M  `MM     MM      MM MMmmMM      MMmmdM9                             `7M'M`MF'  #
#    MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM      , MM.      ,MP AbmmmqMA    MM     ,MP MM   Y  ,   MM  YM.                               VAM,V    #
#    MM    ,dP'A'     VML   MM   A'     VML   MM     ,M `Mb.    ,dP'A'     VML   MM    ,dP' MM     ,M   MM   `Mb.                              VVV     #
#  .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMMmmmmMMM   `"bmmd"'.AMA.   .AMMA.JMMmmmdP' .JMMmmmmMMM .JMML. .JMM.                              V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Get Dataset.
class SQemb_CPmpn_dataset(data.Dataset):
    # SQemb_CPmpn_dataset is for Processing Sequences (Padding) and organizing data that being sent to the model. 
    # In order to process molecules, use Molecule Dataloader to load molecule data (in Z05G_Cpd_Data).
    def __init__(self, seqs_embeddings : List            , 
                       cmpd_dataset    : MoleculeDataset , 
                       target          : List            , 
                       max_len         : int             ,
                       args            : TrainArgs       ,):


    #====================================================================================================#
        super().__init__()
        self.seqs_embeddings = seqs_embeddings
        self.cmpd_dataset    = cmpd_dataset
        self.target          = target
        self.max_len         = max_len


    #====================================================================================================#
    def __len__(self):
        return len(self.seqs_embeddings)

    def __getitem__(self, idx):
        return self.seqs_embeddings[idx],            \
               self.cmpd_dataset[idx],               \
               self.target[idx]                 

    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:


        seqs_embeddings,            \
        cmpd_dataset,               \
        target = zip(*batch)

        batch_size = len(seqs_embeddings)
        embedding_dim = seqs_embeddings[0].shape[1]
        seqs_emb_padded = np.full([batch_size, self.max_len, embedding_dim], 0.0)

        # Padded the sequences embeddings in the batch.
        for seqs_padded, seqs_emb in zip(seqs_emb_padded, seqs_embeddings):
            seqs_slice = tuple(slice(dim) for dim in seqs_emb.shape)
            seqs_padded[seqs_slice] = seqs_emb


        return {'seqs_embeddings' : torch.from_numpy(seqs_emb_padded)               , # torch.LongTensor
                'cmpd_dataset'    : construct_molecule_batch(list(cmpd_dataset))    , # MoleculeDataset
                'y_property'      : torch.tensor(np.array(list(target)))            , # torch.LongTensor
                }


###################################################################################################################
###################################################################################################################
def get_SQemb_CPmpn_pairs(X_tr_seqs, X_tr_smiles_dataset, y_tr,
                        X_va_seqs, X_va_smiles_dataset, y_va,
                        X_ts_seqs, X_ts_smiles_dataset, y_ts,
                        seqs_max_len, batch_size, args):

    X_y_tr = SQemb_CPmpn_dataset(list(X_tr_seqs), X_tr_smiles_dataset, y_tr, seqs_max_len, args)
    X_y_va = SQemb_CPmpn_dataset(list(X_va_seqs), X_va_smiles_dataset, y_va, seqs_max_len, args)
    X_y_ts = SQemb_CPmpn_dataset(list(X_ts_seqs), X_ts_smiles_dataset, y_ts, seqs_max_len, args)

    train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn=X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn=X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn=X_y_ts.collate_fn)
    
    return train_loader, valid_loader, test_loader

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM"""Yb.      db  MMP""MM""YMM  db     `7MMF'        .g8""8q.      db     `7MM"""Yb. `7MM"""YMM  `7MM"""Mq.                                A      #
#    MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM        .dP'    `YM.   ;MM:      MM    `Yb. MM    `7    MM   `MM.                              MMM     #
#    MM     `Mb  ,V^MM.     MM     ,V^MM.     MM        dM'      `MM  ,V^MM.     MM     `Mb MM   d      MM   ,M9                              MMMMM    #
#    MM      MM ,M  `MM     MM    ,M  `MM     MM        MM        MM ,M  `MM     MM      MM MMmmMM      MMmmdM9                             ,MA:M:AM.  #
#    MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM      , MM.      ,MP AbmmmqMA    MM     ,MP MM   Y  ,   MM  YM.                                 M      #
#    MM    ,dP'A'     VML   MM   A'     VML   MM     ,M `Mb.    ,dP'A'     VML   MM    ,dP' MM     ,M   MM   `Mb.                               M      #
#  .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMMmmmmMMM   `"bmmd"'.AMA.   .AMMA.JMMmmmdP' .JMMmmmmMMM .JMML. .JMM.                              M      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#










#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    .g8"""bgd `7MMM.     ,MMF'`7MM"""Mq. `7MM"""Yb.            `7MM        `7MMM.     ,MMF'`7MM"""Mq. `7MN.   `7MF'`7MN.   `7MF'               M      #
#  .dP'     `M   MMMb    dPMM    MM   `MM.  MM    `Yb.            MM          MMMb    dPMM    MM   `MM.  MMN.    M    MMN.    M                 M      #
#  dM'       `   M YM   ,M MM    MM   ,M9   MM     `Mb       ,M""bMM          M YM   ,M MM    MM   ,M9   M YMb   M    M YMb   M                 M      #
#  MM            M  Mb  M' MM    MMmmdM9    MM      MM     ,AP    MM          M  Mb  M' MM    MMmmdM9    M  `MN. M    M  `MN. M             `7M'M`MF'  #
#  MM.           M  YM.P'  MM    MM         MM     ,MP     8MI    MM  mmmmm   M  YM.P'  MM    MM         M   `MM.M    M   `MM.M               VAM,V    #
#  `Mb.     ,'   M  `YM'   MM    MM         MM    ,dP'     `Mb    MM          M  `YM'   MM    MM         M     YMM    M     YMM                VVV     #
#    `"bmmmd'  .JML. `'  .JMML..JMML.     .JMMmmmdP'        `Wbmd"MML.      .JML. `'  .JMML..JMML.     .JML.    YM  .JML.    YM                 V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Model: (Cautions: Need to specify mpn_shared in the arguments for compound protein interaction training.)

class Cmpd_d_MPNN(nn.Module):
    # Class Cmpd_d_MPNN is a directed message passing neural network with,
    #   -  "ECFP6 Count Encodings of molecular structures" as compound features.
    #   -  "Morgan Count Encoding of molecular substructures" as atom features. 
    #   -  "Atom rdkit profiles" as atom features.
    #   -  "Bond rdkit profiles" as bond features.
    #   -  ("Compound profiles" as EXTRA compound features.)

    def __init__(self, 
                 args: TrainArgs, 
                 atom_fdim: int, 
                 bond_fdim: int, 
                 hidden_size: int = None,
                 bias: bool = None, 
                 depth: int = None):
        

        # args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        # atom_fdim: Atom feature vector dimension.
        # bond_fdim: Bond feature vector dimension.
        # hidden_size: Hidden layers dimension
        # bias: Whether to add bias to linear layers
        # depth: Number of message passing steps


        super(Cmpd_d_MPNN, self).__init__()

        # Get model hyperparams from TrainArgs:
        self.hidden_size      = hidden_size        or args.hidden_size
        self.bias             = bias               or args.bias
        self.depth            = depth              or args.depth
        self.device           = args.device
        self.dropout          = args.dropout
        self.undirected       = args.undirected
        self.atom_messages    = args.atom_messages
        self.aggregation      = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        self.layers_per_message = 1

        # For Constructing the model
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim


        # Dropout
        self.dropout_layer = nn.Dropout(p = self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad = False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias = self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias = self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)



    ###################################################################################################################
    ###################################################################################################################
    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        #====================================================================================================#





        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).double().to(self.device)




        f_atoms    , \
        f_bonds    , \
        a2b        , \
        b2a        , \
        b2revb     , \
        a_scope    , \
        b_scope    , \
                       = mol_graph.get_components(atom_messages = self.atom_messages)





        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.double().to(self.device), a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)






        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    .g8"""bgd `7MMM.     ,MMF'`7MM"""Mq. `7MM"""Yb.            `7MM        `7MMM.     ,MMF'`7MM"""Mq. `7MN.   `7MF'`7MN.   `7MF'               A      #
#  .dP'     `M   MMMb    dPMM    MM   `MM.  MM    `Yb.            MM          MMMb    dPMM    MM   `MM.  MMN.    M    MMN.    M                MMM     #
#  dM'       `   M YM   ,M MM    MM   ,M9   MM     `Mb       ,M""bMM          M YM   ,M MM    MM   ,M9   M YMb   M    M YMb   M               MMMMM    #
#  MM            M  Mb  M' MM    MMmmdM9    MM      MM     ,AP    MM          M  Mb  M' MM    MMmmdM9    M  `MN. M    M  `MN. M             ,MA:M:AM.  #
#  MM.           M  YM.P'  MM    MM         MM     ,MP     8MI    MM  mmmmm   M  YM.P'  MM    MM         M   `MM.M    M   `MM.M                 M      #
#  `Mb.     ,'   M  `YM'   MM    MM         MM    ,dP'     `Mb    MM          M  `YM'   MM    MM         M     YMM    M     YMM                 M      #
#    `"bmmmd'  .JML. `'  .JMML..JMML.     .JMMmmmdP'        `Wbmd"MML.      .JML. `'  .JMML..JMML.     .JML.    YM  .JML.    YM                 M      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#












#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#                                                             
#   .g8"""bgd `7MMM.     ,MMF'`7MM"""Mq. `7MM"""Yb.        `7MMM.     ,MMF'               `7MM           `7MM                                   M      #
# .dP'     `M   MMMb    dPMM    MM   `MM.  MM    `Yb.        MMMb    dPMM                   MM             MM                                   M      #
# dM'       `   M YM   ,M MM    MM   ,M9   MM     `Mb        M YM   ,M MM   ,pW"Wq.    ,M""bMM   .gP"Ya    MM                                   M      #
# MM            M  Mb  M' MM    MMmmdM9    MM      MM        M  Mb  M' MM  6W'   `Wb ,AP    MM  ,M'   Yb   MM                               `7M'M`MF'  #
# MM.           M  YM.P'  MM    MM         MM     ,MP        M  YM.P'  MM  8M     M8 8MI    MM  8M""""""   MM                                 VAMAV    #
# `Mb.     ,'   M  `YM'   MM    MM         MM    ,dP'        M  `YM'   MM  YA.   ,A9 `Mb    MM  YM.    ,   MM                                  VVV     #
#   `"bmmmd'  .JML. `'  .JMML..JMML.     .JMMmmmdP'        .JML. `'  .JMML. `Ybmd9'   `Wbmd"MML. `Mbmmd' .JMML.                                 V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class Cmpd_Model(nn.Module):
    # Cmpd_Model calls Cmpd_d_MPNN to learn molecule representations and send to the main model.

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        
        # args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        # atom_fdim: Atom feature vector dimension.
        # bond_fdim: Bond feature vector dimension.

        super(Cmpd_Model, self).__init__()

        #====================================================================================================#
        # For learning reaction representations of the molecules.
        # Replace the solvent here to learn protein-compound interactions.
        # Reaction defaults are False, need to switch to True.
        self.reaction         = args.reaction
        self.reaction_solvent = args.reaction_solvent


        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                     is_reaction=(self.reaction or self.reaction_solvent))

        self.bond_fdim = bond_fdim or get_bond_fdim(overwrite_default_atom=args.overwrite_default_atom_features,
                                                    overwrite_default_bond=args.overwrite_default_bond_features,
                                                    atom_messages=args.atom_messages,
                                                    is_reaction=(self.reaction or self.reaction_solvent))

        print("self.atom_fdim: ", self.atom_fdim)
        print("self.bond_fdim: ", self.bond_fdim)

        #====================================================================================================#
        # 
        self.device = args.device

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features

        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features


        #====================================================================================================#
        # 
        self.CMPD_MODEL = nn.ModuleList([Cmpd_d_MPNN(args, self.atom_fdim, self.bond_fdim)])

        if self.features_only: # Use only the additional features in an FFN, no graph network.
            return

                                        # cmpd_graph_list, 
                                        # cmpd_features_list, 
                                        # cmpd_atom_descriptors_list,
                                        # cmpd_atom_features_list, 
                                        # cmpd_bond_features_list




    def forward(self,
                cmpd_graph_list        :  List[BatchMolGraph]       ,
                cmpd_features_list     :  List[np.ndarray] = None   ,
                atom_descriptors_batch :  List[np.ndarray] = None   ,
                atom_features_batch    :  List[np.ndarray] = None   ,
                bond_features_batch    :  List[np.ndarray] = None   ,) -> torch.FloatTensor:

        # Encodes a batch of molecules.

        # batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      # list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      # The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      # the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        # features_batch: A list of numpy arrays containing additional features.
        # atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        # atom_features_batch: A list of numpy arrays containing additional atom features.
        # bond_features_batch: A list of numpy arrays containing additional bond features.
        # return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.



        if self.use_input_features:
            cmpd_features_list = torch.from_numpy(np.stack(cmpd_features_list)).double().to(self.device)
            if self.features_only:
                return cmpd_features_list

        if self.atom_descriptors == 'descriptor' and len(cmpd_graph_list) > 1:
            raise NotImplementedError('Atom descriptors are currently only supported with one molecule per input (i.e., number_of_molecules = 1).')


        #====================================================================================================#
        # Concatenate MPNN encodings with CMPD Features that were input separately.
        cmpd_enc = [CMPD_MODEL_X(cmpd_graph) for CMPD_MODEL_X, cmpd_graph in zip(self.CMPD_MODEL, cmpd_graph_list)]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), cmpd_enc)

        if self.use_input_features:
            if len(cmpd_features_list.shape) == 1:
                cmpd_features_list = cmpd_features_list.view(1, -1)

            output = torch.cat([output, cmpd_features_list], dim = 1)

        #====================================================================================================#
        return output



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    .g8"""bgd `7MMM.     ,MMF'`7MM"""Mq. `7MM"""Yb.        `7MMM.     ,MMF'               `7MM           `7MM                                  A      #
#  .dP'     `M   MMMb    dPMM    MM   `MM.  MM    `Yb.        MMMb    dPMM                   MM             MM                                 MMM     #
#  dM'       `   M YM   ,M MM    MM   ,M9   MM     `Mb        M YM   ,M MM   ,pW"Wq.    ,M""bMM   .gP"Ya    MM                                MMMMM    #
#  MM            M  Mb  M' MM    MMmmdM9    MM      MM        M  Mb  M' MM  6W'   `Wb ,AP    MM  ,M'   Yb   MM                              ,MA:M:AM.  #
#  MM.           M  YM.P'  MM    MM         MM     ,MP        M  YM.P'  MM  8M     M8 8MI    MM  8M""""""   MM                                  M      #
#  `Mb.     ,'   M  `YM'   MM    MM         MM    ,dP'        M  `YM'   MM  YA.   ,A9 `Mb    MM  YM.    ,   MM                                  M      #
#    `"bmmmd'  .JML. `'  .JMML..JMML.     .JMMmmmdP'        .JML. `'  .JMML. `Ybmd9'   `Wbmd"MML. `Mbmmd' .JMML.                                M      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#



















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#     MMP""MM""YMM   .g8""8q.   `7MM"""Mq.     `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                                     M      #
#     P'   MM   `7 .dP'    `YM.   MM   `MM.      MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                                       M      #
#          MM      dM'      `MM   MM   ,M9       M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                                       M      #
#          MM      MM        MM   MMmmdM9        M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                                   `7M'M`MF'  #
#          MM      MM.      ,MP   MM             M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,                              VAM,V    #
#          MM      `Mb.    ,dP'   MM             M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M                               VVV     #
#        .JMML.      `"bmmd"'   .JMML.         .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM                                V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

class SQembConv_CPmpn_Model(nn.Module):
    # A class SQemb_CONV_CPmpn_MPN is a model which contains: 
    #  -  a NN (Conv1D/sATT/cATT) for reading sequences embeddings, 
    #  -  a Message Passing Network for parsing molecule structures, 
    #  -  a feed-forward network for predicting interactions.
    
    def __init__(self, 
                 args      : TrainArgs  ,
                 in_dim    : int = None ,
                 hid_dim   : int = 256  ,
                 kernal_1  : int = 3    ,
                 out_dim   : int = 1    ,
                 kernal_2  : int = 3    ,
                 max_len   : int = None ,
                 cmpd_dim  : int = None ,
                 last_hid  : int = 1024 ,
                 dropout   : Double = 0.1, ):
        
        super(SQembConv_CPmpn_Model, self).__init__()
        #--------------------------------------------------#
        # Get training essentials.
        self.device             = args.device
        self.loss_function      = args.loss_function

        #--------------------------------------------------#
        # Classification parameters for training.
        self.multiclass         = args.dataset_type == 'multiclass'
        self.classification     = args.dataset_type == 'classification'
        # Classification parameters for neural network.
        self.output_size        = args.num_tasks
        self.output_size       *= args.multiclass_num_classes if self.multiclass else 1
        self.sigmoid            = nn.Sigmoid() if self.classification else None
        self.multiclass_softmax = nn.Softmax(dim=2) if self.multiclass else None

        #--------------------------------------------------#
        # Other parameters for neural networks.
        self.output_size        = args.num_tasks

        #--------------------------------------------------#
        # Select cmpd enc to use and adjust cmpd_dim accordingly.
        self.get_cmpd_encodings(args)

        # cmpd_encodings_dim:
        if args.features_only:
            #cmpd_dim = args.features_size #?
            cmpd_dim = cmpd_dim
        else:
            mpn_enc_dim = args.hidden_size * args.number_of_molecules
            cmpd_dim = mpn_enc_dim + cmpd_dim if args.use_input_features else mpn_enc_dim

        print("cmpd_dim: ", cmpd_dim)

        #====================================================================================================#
        # Top Model Layers
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Layer 1
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout1 = nn.Dropout(dropout, inplace = False)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Layer 2: 
        # 2-1 highway that skips layer #3; 
        # 2-2 layer that goes to layer #3.
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace = False)
        #------------------------------
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace = False)
        #------------------------------
        self.fc_early = nn.Linear(max_len * hid_dim + cmpd_dim, 1)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Layer 3
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout3 = nn.Dropout(dropout, inplace = False)
        #self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # FFN Layers
        self.fc_1 = nn.Linear(int( 2 * max_len * out_dim + cmpd_dim), last_hid)
        self.fc_2 = nn.Linear(last_hid, last_hid)
        self.fc_3 = nn.Linear(last_hid, 1)
        self.cls = nn.Sigmoid()
        #====================================================================================================#



    def get_cmpd_encodings(self, args: TrainArgs) -> None:
        #--------------------------------------------------#
        # Creates the A Encoder for getting molecule encodings.
        self.Cmpd_Enc_Model = Cmpd_Model(args)


        #--------------------------------------------------#
        # Load pre-trained parameters to the model.
        if args.checkpoint_frzn is not None:
            #------------------------------
            if args.freeze_first_only:
            # Freeze only the first encoder.
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            #------------------------------
            else: 
            # Freeze all encoders.
                for param in self.encoder.parameters():
                    param.requires_grad = False



    def forward(self, 
                seqs_embeddings : torch.LongTensor   ,  
                cmpd_dataset    : MoleculeDataset    ,
                ) -> torch.FloatTensor:

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # features_batch         : List[np.ndarray] = None, A list of numpy arrays containing additional features.
        # atom_descriptors_batch : List[np.ndarray] = None, A list of numpy arrays containing additional atom descriptors.
        # atom_features_batch    : List[np.ndarray] = None, A list of numpy arrays containing additional atom features.
        # bond_features_batch    : List[np.ndarray] = None, A list of numpy arrays containing additional bond features.
        # return                 : The output contains a list of property predictions
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Data Final Preparations:
        #------------------------------
        # seqs_embeddings shall be sent to cuda device now.
        # ...
        #------------------------------
        # cmpd_encodings shall be converted to MolGraph List.
        cmpd_graph_list            = cmpd_dataset.batch_graph()        #-> List[BatchMolGraph]
        cmpd_features_list         = cmpd_dataset.features()           #-> List[np.ndarray]
        cmpd_atom_descriptors_list = cmpd_dataset.atom_descriptors()   #-> List[np.ndarray]
        cmpd_atom_features_list    = cmpd_dataset.atom_features()      #-> List[np.ndarray]
        cmpd_bond_features_list    = cmpd_dataset.bond_features()      #-> List[np.ndarray]


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # CMPD Encodings.
        cmpd_encodings = self.Cmpd_Enc_Model(cmpd_graph_list, 
                                             cmpd_features_list, 
                                             cmpd_atom_descriptors_list,
                                             cmpd_atom_features_list, 
                                             cmpd_bond_features_list)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Layer 1
        output = seqs_embeddings.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout1(output)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Layer 2
        output_1 = nn.functional.relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        #------------------------------
        output_2 = nn.functional.relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Layer 3
        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        #------------------------------
        output = torch.cat((output_1,output_2),1)
        #output = self.pooling(output)
        #------------------------------
        output = torch.cat( (torch.flatten(output,1), cmpd_encodings) ,1)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # FFN Layers
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)

        return output, cmpd_encodings

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#      .g8"""bgd                           `7MM      `7MMF'                           `7MM           OO                                                #
#    .dP'     `M                             MM        MM                               MM           88                                                #
#    dM'       `   ,pW"Wq.   ,pW"Wq.    ,M""bMM        MM        `7MM  `7MM   ,p6"bo    MM  ,MP'     ||                                                #
#    MM           6W'   `Wb 6W'   `Wb ,AP    MM        MM          MM    MM  6M'  OO    MM ;Y        ||                                                #
#    MM.    `7MMF'8M     M8 8M     M8 8MI    MM        MM      ,   MM    MM  8M         MM;Mm        ''                                                #
#    `Mb.     MM  YA.   ,A9 YA.   ,A9 `Mb    MM        MM     ,M   MM    MM  YM.    ,   MM `Mb.      __                                                #
#      `"bmmmdPY   `Ybmd9'   `Ybmd9'   `Wbmd"MML.    .JMMmmmmMMM   `Mbod"YML. YMbmd'  .JMML. YA.     MM                                                #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == "__main__":


    Step_code = "Z05G"
    print("*" * 50)
    print(Step_code + " Done!") 






#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#   `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'       `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'  #
#     VAMAV          VAMAV          VAMAV          VAMAV          VAMAV           VAMAV          VAMAV          VAMAV          VAMAV          VAMAV    #
#      VVV            VVV            VVV            VVV            VVV             VVV            VVV            VVV            VVV            VVV     #
#       V              V              V              V              V               V              V              V              V              V      #

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
###################################################################################################################
###################################################################################################################
#====================================================================================================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------#
#------------------------------

#                                                                                                                                                          
#      `MM.              `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.       
#        `Mb.              `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.     
# MMMMMMMMMMMMD     MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD   
#         ,M'               ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'     
#       .M'               .M'              .M'              .M'              .M'              .M'              .M'              .M'              .M'       
#                                                                                                                                                          

#------------------------------
#--------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#====================================================================================================#
###################################################################################################################
###################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #








