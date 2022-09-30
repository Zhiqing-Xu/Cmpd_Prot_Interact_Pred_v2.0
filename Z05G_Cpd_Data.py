#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
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
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns

#--------------------------------------------------#
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.weight_norm import weight_norm

#--------------------------------------------------#
import matplotlib.pyplot as plt
#--------------------------------------------------#
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
#--------------------------------------------------#
from itertools import zip_longest
from collections import OrderedDict


#--------------------------------------------------#
import threading

#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection, Iterator


#--------------------------------------------------#
from AP_convert import Get_Unique_SMILES, MolFromSmiles_ZX


from ZX02_nn_utils import StandardScaler
from ZX02_nn_utils import build_optimizer, build_lr_scheduler
from ZX03_nn_args import TrainArgs
from ZX04_funcs import onek_encoding_unk
from ZX05_loss_functions import get_loss_func



GetUnqSmi = Get_Unique_SMILES(isomericSmiles = True, kekuleSmiles = False, canonical = True, SMARTS_bool = False)

###################################################################################################################
###################################################################################################################







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#          MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION            #
#          MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION            #
#          MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION            #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """
    def __init__(self) -> None:

        # Atom feature sizes
        self.MAX_ATOMIC_NUM = 100 # This shall NOT limit the molecule size. 
        self.ATOM_FEATURES = {  'atomic_num'    : list(range(self.MAX_ATOMIC_NUM)),
                                'degree'        : [0, 1, 2, 3, 4, 5],
                                'formal_charge' : [-1, -2, 1, 2, 0],
                                'chiral_tag'    : [0, 1, 2, 3],
                                'num_Hs'        : [0, 1, 2, 3, 4],
                                'hybridization' : [Chem.rdchem.HybridizationType.SP,
                                                   Chem.rdchem.HybridizationType.SP2,
                                                   Chem.rdchem.HybridizationType.SP3,
                                                   Chem.rdchem.HybridizationType.SP3D,
                                                   Chem.rdchem.HybridizationType.SP3D2 ], }

        # Distance feature sizes
        self.PATH_DISTANCE_BINS    = list(range(10))
        self.THREE_D_DISTANCE_MAX  = 20
        self.THREE_D_DISTANCE_STEP = 1
        self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM       = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM       = 14
        self.EXTRA_BOND_FDIM = 0
        self.REACTION_MODE   = None
        self.EXPLICIT_H      = False
        self.REACTION        = False
        self.ADDING_H        = False
        # print("Featurization_parameters->self.ATOM_FEATURES: ", self.ATOM_FDIM)

#====================================================================================================#
def atom_features(atom: Chem.rdchem.Atom, 
                  functional_groups: List[int] = None, 
                  PARAMS: Featurization_parameters = Featurization_parameters()) -> List[Union[bool, int, float]]:

    # Builds a feature vector for an atom.
    #--------------------------------------------------#
    # param atom: An RDKit atom.
    # param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    # return: A list containing the atom features.
    #--------------------------------------------------#
    # Parameter object for reference throughout this module
    #PARAMS = Featurization_parameters()
    #--------------------------------------------------#
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features =  onek_encoding_unk(atom.GetAtomicNum() - 1       ,     PARAMS.ATOM_FEATURES['atomic_num'])        + \
                    onek_encoding_unk(atom.GetTotalDegree()         ,     PARAMS.ATOM_FEATURES['degree'])            + \
                    onek_encoding_unk(atom.GetFormalCharge()        ,     PARAMS.ATOM_FEATURES['formal_charge'])     + \
                    onek_encoding_unk(int(atom.GetChiralTag())      ,     PARAMS.ATOM_FEATURES['chiral_tag'])        + \
                    onek_encoding_unk(int(atom.GetTotalNumHs())     ,     PARAMS.ATOM_FEATURES['num_Hs'])            + \
                    onek_encoding_unk(int(atom.GetHybridization())  ,     PARAMS.ATOM_FEATURES['hybridization'])     + \
                    [1 if atom.GetIsAromatic() else 0]                                                               + \
                    [atom.GetMass() * 0.01]  # scaled to about the same range as other features

        if functional_groups is not None: # Default to None.
            features += functional_groups

    return features

#====================================================================================================#
def bond_features(bond: Chem.rdchem.Bond,
                  PARAMS: Featurization_parameters = Featurization_parameters()) -> List[Union[bool, int, float]]:
    # Builds a feature vector for a bond.
    #--------------------------------------------------#
    # param bond: An RDKit bond.
    # return: A list containing the bond features.
    #--------------------------------------------------#
    # Parameter object for reference throughout this module
    #PARAMS = Featurization_parameters()
    if bond is None:
        fbond = [1] + [0] * (PARAMS.BOND_FDIM - 1) 
    else:
        bt = bond.GetBondType()
        fbond = [0,  # bond is not None
                 bt == Chem.rdchem.BondType.SINGLE,
                 bt == Chem.rdchem.BondType.DOUBLE,
                 bt == Chem.rdchem.BondType.TRIPLE,
                 bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

#====================================================================================================#
# get atom fearture dimension.
def get_atom_fdim(overwrite_default_atom: bool = False, 
                  is_reaction           : bool = False, 
                  PARAMS: Featurization_parameters = Featurization_parameters() ) -> int:

    # Gets the dimensionality of the atom feature vector.
    # param 'overwrite_default_atom': Whether to overwrite the default atom descriptors
    # param 'is_reaction': Whether to add :code:'EXTRA_ATOM_FDIM' for reaction input when code 'REACTION_MODE' is not None
    # return: The dimensionality of the atom feature vector.

    #PARAMS = Featurization_parameters()
    return (not overwrite_default_atom) * PARAMS.ATOM_FDIM + PARAMS.EXTRA_ATOM_FDIM

#====================================================================================================#
# get bond fearture dimension.
def get_bond_fdim(atom_messages         : bool = False,
                  overwrite_default_bond: bool = False,
                  overwrite_default_atom: bool = False,
                  is_reaction           : bool = False, 
                  PARAMS: Featurization_parameters = Featurization_parameters()) -> int:

    # Gets the dimensionality of the bond feature vector.

    # param atom_messages: Whether atom messages are being used. 
                          # If atom messages are used,
                          # then the bond feature vector only contains bond features.
                          # Otherwise it contains both atom and bond features.
    # param overwrite_default_bond: Whether to overwrite the default bond descriptors
    # param overwrite_default_atom: Whether to overwrite the default atom descriptors
    # param is_reaction: Whether to add :code:'EXTRA_BOND_FDIM' for reaction input when :code:'REACTION_MODE:' is not None
    # return: The dimensionality of the bond feature vector.

    #PARAMS = Featurization_parameters()
    return (not overwrite_default_bond) * PARAMS.BOND_FDIM +  PARAMS.EXTRA_BOND_FDIM + \
           (not atom_messages) * get_atom_fdim(overwrite_default_atom=overwrite_default_atom, is_reaction=is_reaction)


#====================================================================================================#
def set_extra_atom_fdim(extra, PARAMS: Featurization_parameters = Featurization_parameters()):
    """Change the dimensionality of the atom feature vector."""
    PARAMS.EXTRA_ATOM_FDIM = extra
    return


#====================================================================================================#
def set_extra_bond_fdim(extra, PARAMS: Featurization_parameters = Featurization_parameters()):
    """Change the dimensionality of the bond feature vector."""
    PARAMS.EXTRA_BOND_FDIM = extra
    return



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#          MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION            #
#          MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION            #
#          MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION                    MOL FEATURIZATION            #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               MolGraph                    MolGraph                    MolGraph                    MolGraph                    MolGraph               #
#               MolGraph                    MolGraph                    MolGraph                    MolGraph                    MolGraph               #
#               MolGraph                    MolGraph                    MolGraph                    MolGraph                    MolGraph               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MolGraph:

    # Class 'MolGraph' represents the graph structure and featurization of a single molecule.
    # A MolGraph computes the following attributes:

    # vars 'n_atoms'       : The number of atoms in the molecule.
    # vars 'n_bonds'       : The number of bonds in the molecule.
    # vars 'f_atoms'       : A mapping from an atom index to a list of atom features.
    # vars 'f_bonds'       : A mapping from a bond index to a list of bond features.
    # vars 'a2b'           : A mapping from an atom index to a list of incoming bond indices.
    # vars 'b2a'           : A mapping from a bond index to the index of the atom the bond originates from.
    # vars 'b2revb'        : A mapping from a bond index to the index of the reverse bond.
    # vars 'is_mol'        : A boolean whether the input is a molecule.
    # vars 'is_reaction'   : A boolean whether the molecule is a reaction.
    # vars 'is_explicit_h' : A boolean whether to retain explicit Hs (for reaction mode)
    # vars 'is_adding_hs'  : A boolean whether to add explicit Hs (not for reaction mode)
    # vars 'overwrite_default_atom_features': A boolean to overwrite default atom descriptors.
    # vars 'overwrite_default_bond_features': A boolean to overwrite default bond descriptors.


    def __init__(self, mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                 atom_features_extra: np.ndarray = None,
                 bond_features_extra: np.ndarray = None,
                 overwrite_default_atom_features: bool = False,
                 overwrite_default_bond_features: bool = False):

        # param mol: A SMILES or an RDKit molecule.
        # param atom_features_extra: A list of 2D numpy array containing additional atom features to featurize the molecule
        # param bond_features_extra: A list of 2D numpy array containing additional bond features to featurize the molecule
        # param overwrite_default_atom_features: Boolean to overwrite default atom features by atom_features instead of concatenating
        # param overwrite_default_bond_features: Boolean to overwrite default bond features by bond_features instead of concatenating

        #====================================================================================================#
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            Chem.MolFromSmiles_ZX(mol, bad_ss_dict    = {}   , 
                                       isomericSmiles = True , 
                                       kekuleSmiles   = False, 
                                       canonical      = True , 
                                       SMARTS_bool    = False, ) 

        self.n_atoms = 0   # number of atoms
        self.n_bonds = 0   # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b     = []  # mapping from atom index to incoming bond indices                           , old name: " a 2 b "
        self.b2a     = []  # mapping from bond index to the index of the atom the bond is coming from   , old name: " b 2 a "
        self.b2revb  = []  # mapping from bond index to the index of the reverse bond                   , old name: " b 2 revb "
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        #====================================================================================================#
        # Get atom features
        self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()]

        if atom_features_extra is not None:
            if overwrite_default_atom_features:
                self.f_atoms = [descs.tolist() for descs in atom_features_extra]
            else:
                self.f_atoms = [f_atoms + descs.tolist() for f_atoms, descs in zip(self.f_atoms, atom_features_extra)]

        #====================================================================================================#
        # Get number of atoms
        self.n_atoms = len(self.f_atoms)
        if atom_features_extra is not None and len(atom_features_extra) != self.n_atoms:
            raise ValueError(f'The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of the extra atom features.')

        #====================================================================================================#
        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        #====================================================================================================#
        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                if bond_features_extra is not None:
                    descr = bond_features_extra[bond.GetIdx()].tolist()
                    if overwrite_default_bond_features:
                        f_bond = descr
                    else:
                        f_bond += descr

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

        if bond_features_extra is not None and len(bond_features_extra) != self.n_bonds / 2:
            raise ValueError(f'The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of the extra bond features.')
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               MolGraph                    MolGraph                    MolGraph                    MolGraph                    MolGraph               #
#               MolGraph                    MolGraph                    MolGraph                    MolGraph                    MolGraph               #
#               MolGraph                    MolGraph                    MolGraph                    MolGraph                    MolGraph               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                  #
#                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                  #
#                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class BatchMolGraph:

    # A class 'BatchMolGraph' represents the graph structure and featurization of a batch of molecules.
    # A BatchMolGraph contains the attributes of a :class:'MolGraph' plus:

    # vars 'atom_fdim': The dimensionality of the atom feature vector.
    # vars 'bond_fdim': The dimensionality of the bond feature vector (technically the combined atom/bond features).
    # vars 'a_scope': A list of tuples indicating the start and end atom indices for each molecule.
    # vars 'b_scope': A list of tuples indicating the start and end bond indices for each molecule.
    # vars 'max_num_bonds': The maximum number of bonds neighboring an atom in this batch.
    # vars 'b2b': (Optional) A mapping from a bond index to incoming bond indices.
    # vars 'a2a': (Optional): A mapping from an atom index to neighboring atom indices.

    def __init__(self, mol_graphs: List[MolGraph]):
        # param mol_graphs: A list of :class:'MolGraph'\ s from which to construct the :class:'BatchMolGraph'.

        self.overwrite_default_atom_features = mol_graphs[0].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[0].overwrite_default_bond_features

        self.atom_fdim = get_atom_fdim(overwrite_default_atom=self.overwrite_default_atom_features)

        self.bond_fdim = get_bond_fdim(overwrite_default_bond=self.overwrite_default_bond_features,
                                       overwrite_default_atom=self.overwrite_default_atom_features)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:

        # Returns the components of the :class:'BatchMolGraph'.

        # The returned components are, in order:

        # vars 'f_atoms'
        # vars 'f_bonds'
        # vars 'a2b'
        # vars 'b2a'
        # vars 'b2revb'
        # vars 'a_scope'
        # vars 'b_scope'

        # param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
                             # vector to contain only bond features rather than both atom and bond features.
        # return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
                # and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).

        if atom_messages:
            f_bonds = self.f_bonds[:, -get_bond_fdim(atom_messages=atom_messages,
                                                     overwrite_default_atom=self.overwrite_default_atom_features,
                                                     overwrite_default_bond=self.overwrite_default_bond_features):]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """
        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each atom index to all the neighboring atom indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                  #
#                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                  #
#                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                    BatchMolGraph                  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


#====================================================================================================#
def mol2graph(mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
              atom_features_batch: List[np.array] = (None,),
              bond_features_batch: List[np.array] = (None,),
              overwrite_default_atom_features: bool = False,
              overwrite_default_bond_features: bool = False
              ) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy array containing additional atom features to featurize the molecule
    :param bond_features_batch: A list of 2D numpy array containing additional bond features to featurize the molecule
    :param overwrite_default_atom_features: Boolean to overwrite default atom descriptors by atom_descriptors instead of concatenating
    :param overwrite_default_bond_features: Boolean to overwrite default bond descriptors by bond_descriptors instead of concatenating
    :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
    """
    return BatchMolGraph([MolGraph(mol, af, bf,
                                   overwrite_default_atom_features=overwrite_default_atom_features,
                                   overwrite_default_bond_features=overwrite_default_bond_features)
                          for mol, af, bf
                          in zip_longest(mols, atom_features_batch, bond_features_batch)])






























#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#             MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint               #
#             MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint               #
#             MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MoleculeDatapoint:
    # MoleculeDatapoint contains a single molecule and its associated features and targets.

    ###################################################################################################################
    ###################################################################################################################
    def __init__(self,
                 smiles           : List[str]          ,
                 row              : OrderedDict = None,
                 features         : np.ndarray  = None ,
                 phase_features   : List[float] = None ,
                 atom_features    : np.ndarray  = None ,
                 atom_descriptors : np.ndarray  = None ,
                 bond_features    : np.ndarray  = None ,
                 overwrite_default_atom_features : bool = False,
                 overwrite_default_bond_features : bool = False):

        #====================================================================================================#
        # smiles: A list of the SMILES strings for the molecules.
        # row: The raw CSV row containing the information for this molecule.
        # features: A numpy array containing additional features (e.g., Morgan fingerprint).
        # phase_features: A one-hot vector indicating the phase of the data, as used in spectra data.
        # atom_features: A numpy array containing additional atom features to featurize the molecule
        # atom_descriptors: A numpy array containing additional atom descriptors to featurize the molecule
        # bond_features: A numpy array containing additional bond features to featurize the molecule

        self.smiles = smiles # self.smiles is a list!

        self.row = row
        self.features = features
        self.phase_features = phase_features
        self.atom_descriptors = atom_descriptors
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features        

        #====================================================================================================#
        #self.is_explicit_h_list = [is_explicit_h(x) for x in self.is_mol_list]
        #self.is_adding_hs_list = [is_adding_hs(x) for x in self.is_mol_list]

        #====================================================================================================#
        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Fix nans in atom_descriptors
        if self.atom_descriptors is not None:
            self.atom_descriptors = np.where(np.isnan(self.atom_descriptors), replace_token, self.atom_descriptors)

        # Fix nans in atom_features
        if self.atom_features is not None:
            self.atom_features = np.where(np.isnan(self.atom_features), replace_token, self.atom_features)

        # Fix nans in bond_descriptors
        if self.bond_features is not None:
            self.bond_features = np.where(np.isnan(self.bond_features), replace_token, self.bond_features)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features = self.features

        self.raw_atom_descriptors = self.atom_descriptors
        self.raw_atom_features    = self.atom_features
        self.raw_bond_features    = self.bond_features


    ###################################################################################################################
    ###################################################################################################################
    # Get RDKIT mol object from SMILES input string.
    @property
    def mol(self) -> List[Union[Chem.Mol, Tuple[Chem.Mol, Chem.Mol]]]:
        # Gets the corresponding list of RDKit molecules for the corresponding SMILES list.
        def MolFromSmilesList_ZX(smiles_list, bad_ss_dict    = {}    , 
                                              isomericSmiles = True  ,
                                              kekuleSmiles   = False ,  
                                              canonical      = True  ,
                                              SMARTS_bool    = False , ):
            return [MolFromSmiles_ZX(one_smiles, bad_ss_dict, isomericSmiles, kekuleSmiles, canonical, SMARTS_bool) for one_smiles in smiles_list]
        mol = MolFromSmilesList_ZX(self.smiles)
        return mol

    #====================================================================================================#
    @property
    def number_of_molecules(self) -> int:
        # Gets the number of molecules in the class MoleculeDatapoint.
        return len(self.smiles)

    ###################################################################################################################
    ###################################################################################################################
    def set_features(self, features: np.ndarray) -> None:
        # Sets the features of the molecule.
        # :features: A 1D numpy array of features for the molecule.
        self.features = features

    #====================================================================================================#
    def set_atom_descriptors(self, atom_descriptors: np.ndarray) -> None:
        # Sets the atom descriptors of the molecule.
        # atom_descriptors: A 1D numpy array of features for the molecule.
        self.atom_descriptors = atom_descriptors

    #====================================================================================================#
    def set_atom_features(self, atom_features: np.ndarray) -> None:
        """
        Sets the atom features of the molecule.
        :atom_features: A 1D numpy array of features for the molecule.
        """
        self.atom_features = atom_features

    #====================================================================================================#
    def set_bond_features(self, bond_features: np.ndarray) -> None:
        """
        Sets the bond features of the molecule.
        :bond_features: A 1D numpy array of features for the molecule.
        """
        self.bond_features = bond_features

    #====================================================================================================#
    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.
        :features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(self.features, features) if self.features is not None else features

    #====================================================================================================#
    def num_tasks(self) -> int:
        # For Future Use.
        """
        Returns the number of prediction tasks.
        :return: The number of tasks.
        """
        return len(self.targets)

    #====================================================================================================#
    def set_targets(self, targets: List[Optional[float]]):
        # For Future Use.
        """
        Sets the targets of a molecule.
        :targets: A list of floats containing the targets.
        """
        self.targets = targets
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#             MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint               #
#             MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint               #
#             MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint                  MoleculeDatapoint               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#




















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               MoleculeDataset                    MoleculeDataset                    MoleculeDataset                    MoleculeDataset               #
#               MoleculeDataset                    MoleculeDataset                    MoleculeDataset                    MoleculeDataset               #
#               MoleculeDataset                    MoleculeDataset                    MoleculeDataset                    MoleculeDataset               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MoleculeDataset(Dataset):
    # class MoleculeDataset contains a list of classes MoleculeDatapoint's with access to their attributes.
    ###################################################################################################################
    ###################################################################################################################
    def __init__(self, data: List[MoleculeDatapoint]):
        # data: A list of classes MoleculeDatapoint's.
        self._data = data
        self._batch_graph = None
        self._random = random.Random()
    #====================================================================================================#
    def smiles(self, flatten_bool: bool = False) -> Union[List[str], List[List[str]]]:
        # Returns a list containing the SMILES list associated with each class MoleculeDatapoint.
        # param 'flatten': Whether to flatten the returned SMILES to a list instead of a list of lists.
        # return: A list of SMILES or a list of lists of SMILES, depending on vars 'flatten'.
        if flatten_bool:
            return [smiles for d in self._data for smiles in d.smiles]
        return [d.smiles for d in self._data]
    #====================================================================================================#
    def mols(self, flatten_bool: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]], List[Tuple[Chem.Mol, Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]]]:
        # Returns a list of the RDKit molecules associated with each :class:'MoleculeDatapoint'.
        # param 'flatten': Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        # return: A list of SMILES or a list of lists of RDKit molecules, depending on code 'flatten'.
        if flatten_bool:
            return [mol for d in self._data for mol in d.mol]
        return [d.mol for d in self._data]
    #====================================================================================================#
    @property
    def number_of_molecules(self) -> int:
        # Gets the number of molecules in each :class:'MoleculeDatapoint'.
        return self._data[0].number_of_molecules if len(self._data) > 0 else None
    ###################################################################################################################
    ###################################################################################################################
    def features(self) -> List[np.ndarray]:
        # Returns the features associated with each molecule (if they exist).
        # return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        if len(self._data) == 0 or self._data[0].features is None:
            return None
        return [d.features for d in self._data]


    def phase_features(self) -> List[np.ndarray]:
        """
        Returns the phase features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the phase features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].phase_features is None:
            return None

        return [d.phase_features for d in self._data]

    def atom_features(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_features is None:
            return None

        return [d.atom_features for d in self._data]

    def atom_descriptors(self) -> List[np.ndarray]:
        """
        Returns the atom descriptors associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the atom descriptors
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].atom_descriptors is None:
            return None

        return [d.atom_descriptors for d in self._data]

    def bond_features(self) -> List[np.ndarray]:
        """
        Returns the bond features associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the bond features
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].bond_features is None:
            return None

        return [d.bond_features for d in self._data]


    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    def atom_descriptors_size(self) -> int:
        """
        Returns the size of custom additional atom descriptors vector associated with the molecules.

        :return: The size of the additional atom descriptor vector.
        """
        return len(self._data[0].atom_descriptors[0]) \
            if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None

    def atom_features_size(self) -> int:
        """
        Returns the size of custom additional atom features vector associated with the molecules.

        :return: The size of the additional atom feature vector.
        """
        return len(self._data[0].atom_features[0]) \
            if len(self._data) > 0 and self._data[0].atom_features is not None else None

    def bond_features_size(self) -> int:
        """
        Returns the size of custom additional bond features vector associated with the molecules.

        :return: The size of the additional bond feature vector.
        """
        return len(self._data[0].bond_features[0]) \
            if len(self._data) > 0 and self._data[0].bond_features is not None else None
    #====================================================================================================#
    def normalize_features(self, 
                           scaler                 : StandardScaler = None, 
                           replace_nan_token      : int            = 0,
                           scale_atom_descriptors : bool           = False, 
                           scale_bond_features    : bool           = False, ) -> StandardScaler:
        
        # Normalizes the features of the dataset using a class 'nn_utils.StandardScalar'.

        # The class chemprop.data.StandardScaler subtracts the mean and divides by the standard deviation
        # for each feature independently.

        # If a class 'nn_utils.StandardScalar' is provided, it is used to perform the normalization.
        # Otherwise, a class 'nn_utils.StandardScalar' is first fit to the features in this dataset
        # and is then used to perform the normalization.

        # param scaler: A fitted class 'nn_utils.StandardScalar'. If it is provided it is used,
        #               otherwise a new class 'nn_utils.StandardScalar' is first fitted to this
        #               data and is then used.
        # param replace_nan_token: A token to use to replace NaN entries in the features.
        # param scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
        # param scale_bond_features: If the features that need to be scaled are bond descriptors rather than molecule.

        # return: A fitted class 'nn_utils.StandardScalar'. If a class 'nn_utils.StandardScalar'
        #         is provided as a parameter, this is the same class 'nn_utils.StandardScalar'. Otherwise,
        #         this is a new class 'nn_utils.StandardScalar' that has been fit on this dataset.

        if len(self._data) == 0 or (self._data[0].features is None and not scale_bond_features and not scale_atom_descriptors):
            return None
        if scaler is None:
            if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
                features = np.vstack([d.raw_atom_descriptors for d in self._data])
            elif scale_atom_descriptors and not self._data[0].atom_features is None:
                features = np.vstack([d.raw_atom_features for d in self._data])
            elif scale_bond_features:
                features = np.vstack([d.raw_bond_features for d in self._data])
            else:
                features = np.vstack([d.raw_features for d in self._data])
            scaler = StandardScaler(replace_nan_token=replace_nan_token)
            scaler.fit(features)

        if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
            for d in self._data:
                d.set_atom_descriptors(scaler.transform(d.raw_atom_descriptors))
        elif scale_atom_descriptors and not self._data[0].atom_features is None:
            for d in self._data:
                d.set_atom_features(scaler.transform(d.raw_atom_features))
        elif scale_bond_features:
            for d in self._data:
                d.set_bond_features(scaler.transform(d.raw_bond_features))
        else:
            for d in self._data:
                d.set_features(scaler.transform(d.raw_features.reshape(1, -1))[0])
        return scaler

    ###################################################################################################################
    ###################################################################################################################
    def batch_graph(self) -> List[BatchMolGraph]:

        # Constructs a :class:'~chemprop.features.BatchMolGraph' with the graph featurization of all the molecules.

        #   note::
        #   The class '~chemprop.features.BatchMolGraph' is cached in after the first time it is computed
        #   and is simply accessed upon subsequent calls to :meth:'batch_graph'. This means that if the underlying
        #   set of :class:'MoleculeDatapoint'\ s changes, then the returned :class:'~chemprop.features.BatchMolGraph'
        #   will be incorrect for the underlying data.

        # return: A list of :class:'~chemprop.features.BatchMolGraph' containing the graph featurization of all the
        #         molecules in each :class:'MoleculeDatapoint'.

        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):

                    if len(d.smiles) > 1 and (d.atom_features is not None or d.bond_features is not None):
                        raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                                    'per input (i.e., number_of_molecules = 1).')

                    mol_graph = MolGraph(m, d.atom_features, d.bond_features,
                                            overwrite_default_atom_features=d.overwrite_default_atom_features,
                                            overwrite_default_bond_features=d.overwrite_default_bond_features)

                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [BatchMolGraph([g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
            # print("len(self._batch_graph): ", len(self._batch_graph))  # len = 1



        return self._batch_graph

    ###################################################################################################################
    ###################################################################################################################
    def __len__(self) -> int:
        # Returns the length of the dataset (i.e., the number of molecules).
        return len(self._data)
    #====================================================================================================#
    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        # Gets one or more class MoleculeDatapoint's via an index or slice.
        # param 'item': An index (int) or a slice object.
        # return: A class MoleculeDatapoint if an int is provided or a list of class MoleculeDatapoint if a slice is provided.
        return self._data[item]


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               MoleculeDataset                    MoleculeDataset                    MoleculeDataset                    MoleculeDataset               #
#               MoleculeDataset                    MoleculeDataset                    MoleculeDataset                    MoleculeDataset               #
#               MoleculeDataset                    MoleculeDataset                    MoleculeDataset                    MoleculeDataset               #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#           MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs          #
#           MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs          #
#           MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs          #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    # Filters out invalid SMILES.
    
    # param data: A :class:`~chemprop.data.MoleculeDataset`.
    # return: A :class:`~chemprop.data.MoleculeDataset` with only the valid molecules.

    return MoleculeDataset([datapoint for datapoint in data
                            if all(s != '' for s in datapoint.smiles) and all(m is not None for m in datapoint.mol)
                            and all(m.GetNumHeavyAtoms() > 0 for m in datapoint.mol if not isinstance(m, tuple))
                            and all(m[0].GetNumHeavyAtoms() + m[1].GetNumHeavyAtoms() > 0 for m in datapoint.mol if isinstance(m, tuple))])



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#           MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs          #
#           MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs          #
#           MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs               MoleculeDataset Funcs          #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMM.     ,MMF'  .g8""8q.   `7MMF'           .M"""bgd       db      `7MMM.     ,MMF'`7MM"""Mq. `7MMF'      `7MM"""YMM  `7MM"""Mq.  
#    MMMb    dPMM  .dP'    `YM.   MM            ,MI    "Y      ;MM:       MMMb    dPMM    MM   `MM.  MM          MM    `7    MM   `MM. 
#    M YM   ,M MM  dM'      `MM   MM            `MMb.         ,V^MM.      M YM   ,M MM    MM   ,M9   MM          MM   d      MM   ,M9  
#    M  Mb  M' MM  MM        MM   MM              `YMMNq.    ,M  `MM      M  Mb  M' MM    MMmmdM9    MM          MMmmMM      MMmmdM9   
#    M  YM.P'  MM  MM.      ,MP   MM      ,     .     `MM    AbmmmqMA     M  YM.P'  MM    MM         MM      ,   MM   Y  ,   MM  YM.   
#    M  `YM'   MM  `Mb.    ,dP'   MM     ,M     Mb     dM   A'     VML    M  `YM'   MM    MM         MM     ,M   MM     ,M   MM   `Mb. 
#  .JML. `'  .JMML.  `"bmmd"'   .JMMmmmmMMM     P"Ybmmd"  .AMA.   .AMMA..JML. `'  .JMML..JMML.     .JMMmmmmMMM .JMMmmmmMMM .JMML. .JMM.
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = random.Random(seed)

        if self.class_balance:
            indices = np.arange(len(dataset))
            has_active = np.array([any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices), len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMM.     ,MMF'  .g8""8q.   `7MMF'           .M"""bgd       db      `7MMM.     ,MMF'`7MM"""Mq. `7MMF'      `7MM"""YMM  `7MM"""Mq.  
#    MMMb    dPMM  .dP'    `YM.   MM            ,MI    "Y      ;MM:       MMMb    dPMM    MM   `MM.  MM          MM    `7    MM   `MM. 
#    M YM   ,M MM  dM'      `MM   MM            `MMb.         ,V^MM.      M YM   ,M MM    MM   ,M9   MM          MM   d      MM   ,M9  
#    M  Mb  M' MM  MM        MM   MM              `YMMNq.    ,M  `MM      M  Mb  M' MM    MMmmdM9    MM          MMmmMM      MMmmdM9   
#    M  YM.P'  MM  MM.      ,MP   MM      ,     .     `MM    AbmmmqMA     M  YM.P'  MM    MM         MM      ,   MM   Y  ,   MM  YM.   
#    M  `YM'   MM  `Mb.    ,dP'   MM     ,M     Mb     dM   A'     VML    M  `YM'   MM    MM         MM     ,M   MM     ,M   MM   `Mb. 
#  .JML. `'  .JMML.  `"bmmd"'   .JMMmmmmMMM     P"Ybmmd"  .AMA.   .AMMA..JML. `'  .JMML..JMML.     .JMMmmmmMMM .JMMmmmmMMM .JMML. .JMM.
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#






#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMM.     ,MMF'  .g8""8q.   `7MMF'          `7MM"""Yb.              mm             `7MMF'                              `7MM                        #
#    MMMb    dPMM  .dP'    `YM.   MM              MM    `Yb.            MM               MM                                  MM                        #
#    M YM   ,M MM  dM'      `MM   MM              MM     `Mb  ,6"Yb.  mmMMmm   ,6"Yb.    MM         ,pW"Wq.   ,6"Yb.    ,M""bMM   .gP"Ya  `7Mb,od8     #
#    M  Mb  M' MM  MM        MM   MM              MM      MM 8)   MM    MM    8)   MM    MM        6W'   `Wb 8)   MM  ,AP    MM  ,M'   Yb   MM' "'     #
#    M  YM.P'  MM  MM.      ,MP   MM      ,       MM     ,MP  ,pm9MM    MM     ,pm9MM    MM      , 8M     M8  ,pm9MM  8MI    MM  8M""""""   MM         #
#    M  `YM'   MM  `Mb.    ,dP'   MM     ,M       MM    ,dP' 8M   MM    MM    8M   MM    MM     ,M YA.   ,A9 8M   MM  `Mb    MM  YM.    ,   MM         #
#  .JML. `'  .JMML.  `"bmmd"'   .JMMmmmmMMM     .JMMmmmdP'   `Moo9^Yo.  `Mbmo `Moo9^Yo..JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML. `Mbmmd' .JMML.       #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )

        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )


    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()


###################################################################################################################
###################################################################################################################
def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~chemprop.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMM.     ,MMF'  .g8""8q.   `7MMF'          `7MM"""Yb.              mm             `7MMF'                              `7MM                        #
#    MMMb    dPMM  .dP'    `YM.   MM              MM    `Yb.            MM               MM                                  MM                        #
#    M YM   ,M MM  dM'      `MM   MM              MM     `Mb  ,6"Yb.  mmMMmm   ,6"Yb.    MM         ,pW"Wq.   ,6"Yb.    ,M""bMM   .gP"Ya  `7Mb,od8     #
#    M  Mb  M' MM  MM        MM   MM              MM      MM 8)   MM    MM    8)   MM    MM        6W'   `Wb 8)   MM  ,AP    MM  ,M'   Yb   MM' "'     #
#    M  YM.P'  MM  MM.      ,MP   MM      ,       MM     ,MP  ,pm9MM    MM     ,pm9MM    MM      , 8M     M8  ,pm9MM  8MI    MM  8M""""""   MM         #
#    M  `YM'   MM  `Mb.    ,dP'   MM     ,M       MM    ,dP' 8M   MM    MM    8M   MM    MM     ,M YA.   ,A9 8M   MM  `Mb    MM  YM.    ,   MM         #
#  .JML. `'  .JMML.  `"bmmd"'   .JMMmmmmMMM     .JMMmmmdP'   `Moo9^Yo.  `Mbmo `Moo9^Yo..JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML. `Mbmmd' .JMML.       #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#




















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  MMM"""AMV             M******   .g8"""bgd        .g8"""bgd `7MM"""Mq.`7MM"""Yb.      `7MM"""Yb.      db  MMP""MM""YMM  db                    M      #
#  M'   AMV             .M       .dP'     `M      .dP'     `M   MM   `MM. MM    `Yb.      MM    `Yb.   ;MM: P'   MM   `7 ;MM:                   M      #
#  '   AMV    ,pP""Yq.  |bMMAg.  dM'       `      dM'       `   MM   ,M9  MM     `Mb      MM     `Mb  ,V^MM.     MM     ,V^MM.                  M      #
#     AMV    6W'    `Wb      `Mb MM               MM            MMmmdM9   MM      MM      MM      MM ,M  `MM     MM    ,M  `MM              `7M'M`MF'  #
#    AMV   , 8M      M8       jM MM.    `7MMF'    MM.           MM        MM     ,MP      MM     ,MP AbmmmqMA    MM    AbmmmqMA               VAM,V    #
#   AMV   ,M YA.    ,A9 (O)  ,M9 `Mb.     MM      `Mb.     ,'   MM        MM    ,dP'      MM    ,dP'A'     VML   MM   A'     VML               VVV     #
#  AMVmmmmMM  `Ybmmd9'   6mmm9     `"bmmmdPY        `"bmmmd'  .JMML.    .JMMmmmdP'      .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.              V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Format smiles set and compound features into class MoleculeDataset.



def Z05G_Cpd_Data(X_tr_smiles : List, X_ts_smiles : List, X_va_smiles : List,
                  X_tr_cmpd   : List, X_ts_cmpd   : List, X_va_cmpd   : List,
                  args : TrainArgs
                  ):

    X_tr_smiles_dataset = MoleculeDataset([ MoleculeDatapoint(smiles           = [smiles,],
                                                              row              = OrderedDict({'smiles': smiles}),
                                                              features         = X_tr_cmpd[i],
                                                              phase_features   = None,
                                                              atom_features    = None,
                                                              atom_descriptors = None,
                                                              bond_features    = None)  
                                                              for i, smiles in enumerate(X_tr_smiles)  ])

    X_ts_smiles_dataset = MoleculeDataset([ MoleculeDatapoint(smiles           = [smiles,],
                                                              row              = OrderedDict({'smiles': smiles}),
                                                              features         = X_ts_cmpd[i],
                                                              phase_features   = None,
                                                              atom_features    = None,
                                                              atom_descriptors = None,
                                                              bond_features    = None)  
                                                              for i, smiles in enumerate(X_ts_smiles)  ])

    X_va_smiles_dataset = MoleculeDataset([ MoleculeDatapoint(smiles           = [smiles,],
                                                              row              = OrderedDict({'smiles': smiles}),
                                                              features         = X_va_cmpd[i],
                                                              phase_features   = None,
                                                              atom_features    = None,
                                                              atom_descriptors = None,
                                                              bond_features    = None)  
                                                              for i, smiles in enumerate(X_va_smiles)  ])    



    #====================================================================================================#
    # Validate the datasets.
    original_data_len = len(X_tr_smiles_dataset)
    X_tr_smiles_dataset = filter_invalid_smiles(X_tr_smiles_dataset)
    if len(X_tr_smiles_dataset) < original_data_len:
        print(f'Warning: {original_data_len - len(X_tr_smiles_dataset)} SMILES are invalid.')

    original_data_len = len(X_ts_smiles_dataset)
    X_ts_smiles_dataset = filter_invalid_smiles(X_ts_smiles_dataset)
    if len(X_ts_smiles_dataset) < original_data_len:
        print(f'Warning: {original_data_len - len(X_ts_smiles_dataset)} SMILES are invalid.')

    original_data_len = len(X_va_smiles_dataset)
    X_va_smiles_dataset = filter_invalid_smiles(X_va_smiles_dataset)
    if len(X_va_smiles_dataset) < original_data_len:
        print(f'Warning: {original_data_len - len(X_va_smiles_dataset)} SMILES are invalid.')

    #====================================================================================================#
    # Update args.

    args.features_size = X_tr_smiles_dataset.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = X_tr_smiles_dataset.atom_descriptors_size()
        args.ffn_hidden_size += args.atom_descriptors_size 
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = X_tr_smiles_dataset.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
    if args.bond_features_path is not None:
        args.bond_features_size = X_tr_smiles_dataset.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)


    if args.features_scaling:
        features_scaler = X_tr_smiles_dataset.normalize_features(replace_nan_token=0)
        X_va_smiles_dataset.normalize_features(features_scaler)
        X_ts_smiles_dataset.normalize_features(features_scaler)
    else:
        features_scaler = None


    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = X_tr_smiles_dataset.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        X_va_smiles_dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        X_ts_smiles_dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None


    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = X_tr_smiles_dataset.normalize_features(replace_nan_token=0, scale_bond_features=True)
        X_va_smiles_dataset.normalize_features(bond_feature_scaler, scale_bond_features=True)
        X_ts_smiles_dataset.normalize_features(bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None


    args.train_data_size = len(X_tr_smiles_dataset)


    #====================================================================================================#
    # Validate the size of arguments.
    print("len(X_tr_smiles_dataset.batch_graph()): ", len(X_tr_smiles_dataset.batch_graph()))
    print("X_tr_smiles_dataset.batch_graph()[0].f_atoms: ", X_tr_smiles_dataset.batch_graph()[0].f_atoms)
    print("X_tr_smiles_dataset.batch_graph()[0].f_atoms.size(): ", X_tr_smiles_dataset.batch_graph()[0].f_atoms.size())
    print("X_tr_smiles_dataset.features_size(): ", X_tr_smiles_dataset.features_size())                   # None
    print("X_tr_smiles_dataset.atom_features_size(): ", X_tr_smiles_dataset.atom_features_size())         # None
    print("X_tr_smiles_dataset.bond_features_size(): ", X_tr_smiles_dataset.bond_features_size())         # None
    print("args.train_data_size: ", args.train_data_size )

    return X_tr_smiles_dataset, X_ts_smiles_dataset, X_va_smiles_dataset












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
if __name__ == "__main__":
    print()


#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #