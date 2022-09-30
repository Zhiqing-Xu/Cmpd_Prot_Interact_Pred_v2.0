#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
from logging import raiseExceptions
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
import re
import sys
import time
import scipy
import torch
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
#--------------------------------------------------#
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from Bio import SeqIO
from tqdm import tqdm
from tpot import TPOTRegressor
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
from Z05_split_data import *

#--------------------------------------------------#
from ZX01_PLOT import *
from ZX02_nn_utils import StandardScaler, normalize_targets







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.       db      `7MM"""Mq.        db      MMP""MM""YMM `7MMF'  .g8""8q.   `7MN.   `7MF'    #
#   __,           MM   `MM.  MM   `MM.   MM    `7    MM   `MM.     ;MM:       MM   `MM.      ;MM:     P'   MM   `7   MM  .dP'    `YM.   MMN.    M      #
#  `7MM           MM   ,M9   MM   ,M9    MM   d      MM   ,M9     ,V^MM.      MM   ,M9      ,V^MM.         MM        MM  dM'      `MM   M YMb   M      #
#    MM           MMmmdM9    MMmmdM9     MMmmMM      MMmmdM9     ,M  `MM      MMmmdM9      ,M  `MM         MM        MM  MM        MM   M  `MN. M      #
#    MM           MM         MM  YM.     MM   Y  ,   MM          AbmmmqMA     MM  YM.      AbmmmqMA        MM        MM  MM.      ,MP   M   `MM.M      #
#    MM  ,,       MM         MM   `Mb.   MM     ,M   MM         A'     VML    MM   `Mb.   A'     VML       MM        MM  `Mb.    ,dP'   M     YMM      #
#  .JMML.db     .JMML.     .JMML. .JMM..JMMmmmmMMM .JMML.     .AMA.   .AMMA..JMML. .JMM..AMA.   .AMMA.   .JMML.    .JMML.  `"bmmd"'   .JML.    YM      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
## Create Temp Folder for Saving Results
def Create_Temp_Folder(results_folder, 
                       encoding_file, 
                       embedding_file, 
                       dataset_nme, 
                       Step_code, 
                       NN_type, 
                       split_type, 
                       screen_bool, 
                       log_value, 
                       classification_threshold_type,
                       MSA_bool):
    #====================================================================================================#
    print("="*80)
    print("\n\n\n>>> Creating temporary subfolder and clear past empty folders... ")
    print("="*80)
    now = datetime.now()
    #d_t_string = now.strftime("%Y%m%d_%H%M%S")
    d_t_string = now.strftime("%m%d-%H%M%S")
    #====================================================================================================#
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_folder_contents = os.listdir(results_folder)
    count_non_empty_folder = 0
    for item in results_folder_contents:
        if os.path.isdir(results_folder / item):
            num_files = len(os.listdir(results_folder/item))
            if num_files in [1,2]:
                try:
                    for idx in range(num_files):
                        os.remove(results_folder / item / os.listdir(results_folder/item)[0])
                    os.rmdir(results_folder / item)
                    print("Remove empty folder " + item + "!")
                except:
                    print("Cannot remove empty folder " + item + "!")
            elif num_files == 0:
                try:
                    os.rmdir(results_folder / item)
                    print("Remove empty folder " + item + "!")
                except:
                    print("Cannot remove empty folder " + item + "!")
            else:
                count_non_empty_folder += 1
    print("Found " + str(count_non_empty_folder) + " non-empty folders: " + "!")
    print("="*80)
    #====================================================================================================#
    encoding_code=encoding_file.replace("X02A_" + dataset_nme + "_", "")
    encoding_code=encoding_code.replace("_encodings_dict.p", "")
    #====================================================================================================#
    embedding_code=embedding_file.replace("X03_" + dataset_nme + "_embedding_", "")
    embedding_code=embedding_code.replace(".p", "")
    #====================================================================================================#
    temp_folder_name = Step_code 
    temp_folder_name += d_t_string + "_"
    temp_folder_name += dataset_nme + "_"
    temp_folder_name += embedding_code.replace("_","") + "_"
    temp_folder_name += encoding_code + "_"
    temp_folder_name += NN_type.upper() + "_"
    temp_folder_name += "splt" + str(split_type) + "_"
    temp_folder_name += "scrn" + str(screen_bool)[0] + "_"
    temp_folder_name += "lg" + str(log_value)[0] + "_"
    temp_folder_name += "thrhld" + str(classification_threshold_type) + "_"
    temp_folder_name += "MSA" + str(MSA_bool)[0]
    #====================================================================================================#
    results_sub_folder=Path("X_DataProcessing/" + Step_code + "intermediate_results/" + temp_folder_name +"/")
    if not os.path.exists(results_sub_folder):
        os.makedirs(results_sub_folder)
    return results_sub_folder


###################################################################################################################
###################################################################################################################
# Modify the print function AND print all interested info.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                .g8""8q.   `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM    `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM #
#              .dP'    `YM.   MM       M  P'   MM   `7   MM   `MM.  MM       M  P'   MM   `7      MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 #
#  pd*"*b.     dM'      `MM   MM       M       MM        MM   ,M9   MM       M       MM           MM   ,M9   MM   ,M9    MM    M YMb   M       MM      #
# (O)   j8     MM        MM   MM       M       MM        MMmmdM9    MM       M       MM           MMmmdM9    MMmmdM9     MM    M  `MN. M       MM      #
#     ,;j9     MM.      ,MP   MM       M       MM        MM         MM       M       MM           MM         MM  YM.     MM    M   `MM.M       MM      #
#  ,-='    ,,  `Mb.    ,dP'   YM.     ,M       MM        MM         YM.     ,M       MM           MM         MM   `Mb.   MM    M     YMM       MM      #
# Ammmmmmm db    `"bmmd"'      `bmmmmd"'     .JMML.    .JMML.        `bmmmmd"'     .JMML.       .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
 
def output_print(dataset_nme,
                 results_sub_folder,
                 embedding_file,
                 encoding_file,
                 log_value,
                 screen_bool,
                 clf_thrhld_type,
                 MSA_bool,
                 split_type,
                 epoch_num,
                 batch_size,
                 learning_rate,
                 NN_type,
                 seed,
                 hyperparameters_dict):
    #====================================================================================================#
    orig_stdout = sys.stdout
    f = open(results_sub_folder / 'print_out.txt', 'w')
    sys.stdout = Tee(sys.stdout, f)
    #--------------------------------------------------#
    print("\n\n\n>>> Initializing hyperparameters and settings... ")
    print("="*80)
    #--------------------------------------------------#
    print("dataset_nme           : ", dataset_nme)
    print("embedding_file        : ", embedding_file)
    print("encoding_file         : ", encoding_file)
    #--------------------------------------------------#
    print("log_value             : ", log_value,   " (Whether to use log values of Y.)")
    print("screen_bool           : ", screen_bool, " (Whether to remove zeroes.)")
    print("clf_thrhld_type       : ", clf_thrhld_type, " (Type 2: 1e-5, Type 3: 1e-2)")
    print("MSA_bool              : ", MSA_bool, " (Whether to use aligned sequences.)")
    #--------------------------------------------------#
    print("split_type            : ", split_type)
    #--------------------------------------------------#
    print("NN_type               : ", NN_type)
    print("Random Seed           : ", seed)
    print("epoch_num             : ", epoch_num)
    print("batch_size            : ", batch_size)
    print("learning_rate         : ", learning_rate)
    #--------------------------------------------------#
    print("-"*80)
    for one_hyperpara in hyperparameters_dict:
        print(one_hyperpara, " "*(21-len(one_hyperpara)), ": ", hyperparameters_dict[one_hyperpara])
    print("="*80)
    return 







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   pd""b.          `7MM"""Yb.      db  MMP""MM""YMM  db           .M"""bgd `7MM"""Mq.`7MMF'      `7MMF'MMP""MM""YMM                                   #
#  (O)  `8b           MM    `Yb.   ;MM: P'   MM   `7 ;MM:         ,MI    "Y   MM   `MM. MM          MM  P'   MM   `7                                   #
#       ,89           MM     `Mb  ,V^MM.     MM     ,V^MM.        `MMb.       MM   ,M9  MM          MM       MM                                        #
#     ""Yb.           MM      MM ,M  `MM     MM    ,M  `MM          `YMMNq.   MMmmdM9   MM          MM       MM                                        #
#        88           MM     ,MP AbmmmqMA    MM    AbmmmqMA       .     `MM   MM        MM      ,   MM       MM                                        #
#  (O)  .M'   ,,      MM    ,dP'A'     VML   MM   A'     VML      Mb     dM   MM        MM     ,M   MM       MM                                        #
#   bmmmd'    db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.    P"Ybmmd"  .JMML.    .JMMmmmmMMM .JMML.   .JMML.                                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  


# Prepare Train/Test/Validation Dataset for NN model.
def tr_ts_va_for_NN(dataset_nme,
                    data_folder,
                    results_folder,
                    results_sub_folder,
                    seqs_fasta_file,
                    encoding_file,
                    embedding_file,
                    properties_file,
                    MSA_info_file,
                    MSA_bool,
                    cstm_splt_file,
                    screen_bool,
                    log_value,
                    split_type,
                    NN_type,
                    clf_thrhld_type,
                    seed):







    ###################################################################################################################
    #                               `7MM"""YMM `7MMF'`7MMF'      `7MM"""YMM   .M"""bgd                                #
    #                                 MM    `7   MM    MM          MM    `7  ,MI    "Y                                #
    #                                 MM   d     MM    MM          MM   d    `MMb.                                    #
    #                                 MM""MM     MM    MM          MMmmMM      `YMMNq.                                #
    #                                 MM   Y     MM    MM      ,   MM   Y  , .     `MM                                #
    #                                 MM         MM    MM     ,M   MM     ,M Mb     dM                                #
    #                               .JMML.     .JMML..JMMmmmmMMM .JMMmmmmMMM P"Ybmmd"                                 #
    ###################################################################################################################
    # Get Input files
    print("\n\n\n>>> Getting all input files and splitting the data... ")
    print("="*80)
    #====================================================================================================#
    # Get Compound Encodings from X02 pickles.
    with open( data_folder / encoding_file, 'rb') as cmpd_encodings:
        X_cmpd_encodings_dict = pickle.load(cmpd_encodings)
    #====================================================================================================#
    # Get Sequence Embeddings from X03 pickles.
    with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
        seqs_embeddings_pkl = pickle.load(seqs_embeddings)
    try: 
        X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens']
    except:
        X_seqs_all_hiddens_list = seqs_embeddings_pkl['seqs_all_hiddens']
    del(seqs_embeddings_pkl)
    #====================================================================================================#
    # Get Sequences file (.fasta) as well.
    with open(data_folder / seqs_fasta_file) as f:
        lines = f.readlines()
    one_sequence = ""
    X_all_seqs_list = []
    seqs_nme = ""
    for line_idx, one_line in enumerate(lines):
        if ">seq" in one_line:
            if one_sequence != "":
                X_all_seqs_list.append(one_sequence)
            # new sequence start from here
            one_sequence = ""
            seqs_nme = one_line.replace("\n", "")
        if ">seq" not in one_line:
            one_sequence = one_sequence + one_line.replace("\n", "")
        if line_idx == len(lines) - 1:
            X_all_seqs_list.append(one_sequence)

    #====================================================================================================#
    # Get cmpd_properties_list.
    # [[one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    with open( data_folder / properties_file, 'rb') as cmpd_properties:
        cmpd_properties_list = pickle.load(cmpd_properties) 
    #====================================================================================================#
    customized_idx_list = []
    customized_idx_dict = dict([])
    # split_type = 0, train/test split completely randomly selected
    # split_type = 1, train/test split contains different seq-cmpd pairs
    # split_type = 2, train/test split contains different seqs
    # split_type = 3, train/test split contains different cmpd
    # split_type = 4, train/test split contains different CD-hit seqs, train contains non-CD-hit seqs.
    # split_type = 5, train/test split contains different CD-hit seqs
    # split_type = 6, train/test split completely randomly selected, with non-CD-hit seqs data all contained in train.
    # split_type = 7, train/test split completely randomly selected, with non-CD-hit seqs data being left out.
    if split_type in [4,5,6,7]: 
        with open( data_folder / cstm_splt_file, 'rb') as cstm_splt:
            customized_idx_list = pickle.load(cstm_splt)
        
        cstm_splt_clstr_file = str(cstm_splt_file).replace("splt", "splt_clstr")

        with open( data_folder / cstm_splt_clstr_file, 'rb') as cstm_splt_clstr:
            customized_idx_dict = pickle.load(cstm_splt_clstr)








    ###################################################################################################################
    #                                    `7MMM.     ,MMF' .M"""bgd       db                                           #
    #                                      MMMb    dPMM  ,MI    "Y      ;MM:                                          #
    #                                      M YM   ,M MM  `MMb.         ,V^MM.                                         #
    #                                      M  Mb  M' MM    `YMMNq.    ,M  `MM                                         #
    #                                      M  YM.P'  MM  .     `MM    AbmmmqMA                                        #
    #                                      M  `YM'   MM  Mb     dM   A'     VML                                       #
    #                                    .JML. `'  .JMML.P"Ybmmd"  .AMA.   .AMMA.                                     #
    ###################################################################################################################
    '''
    # Introduce MSA information to the model.
    #====================================================================================================#
    def Get_Embeddings_w_MSA(MSA_info_dict, X_seqs_all_hiddens_list, aligned_seqs_len):
        # Input: X_seqs_all_hiddens_list: length of this list = number of sequences
        # Input: X_seqs_all_hiddens_list: [(), (), (seqs_len, Emb_dim), ...]
        # Output: X_seqs_all_hiddens_list_MSA : [(), (), (aligned_seqs_len, Emb_dim), ...]
        X_seqs_all_hiddens_list_MSA = []
        for i in range(len(MSA_info_dict)):
            #--------------------------------------------------#
            # Step 1: Verify if it is operating on the right seq using seqs_len.
            assert aligned_seqs_len == (MSA_info_dict[">seq"+str(i+1)]["seqs_length"] + len(MSA_info_dict[">seq"+str(i+1)]["gaps"]))
            #--------------------------------------------------#
            # Step 2: 
            # MSA_seqs_emb = initialize a numpy array here (based on aligned_seqs_len and seqs_emb_dimension)!
            emb_dim = X_seqs_all_hiddens_list[0].shape[1]
            MSA_seqs_emb = np.zeros((aligned_seqs_len, emb_dim), )
            #print(X_seqs_all_hiddens_list[i].shape)
            #print(MSA_info_dict[">seq"+str(i+1)])
            #--------------------------------------------------#
            count_amino_acids = 0
            for one_location in range(aligned_seqs_len):
                if one_location in MSA_info_dict[">seq"+str(i+1)]["gaps"]:
                    # fill it with zeros for one_location in MSA_seqs_emb 
                    MSA_seqs_emb[one_location] = np.zeros(emb_dim)
                else:
                    # MSA_seqs_emb takes the embeddings @ count_amino_acids location.
                    MSA_seqs_emb[one_location] = X_seqs_all_hiddens_list[i][count_amino_acids]
                    count_amino_acids += 1
            #--------------------------------------------------#
            X_seqs_all_hiddens_list_MSA.append(MSA_seqs_emb)
            #--------------------------------------------------#
        return X_seqs_all_hiddens_list_MSA #(Same format as X_seqs_all_hiddens_list)
    

    #====================================================================================================#
    # Replace X_seqs_all_hiddens_list with X_seqs_all_hiddens_list_MSA to include MSA info
    if MSA_bool == True:
        #--------------------------------------------------#
        # Get MSA_info_dict.
        with open( data_folder / MSA_info_file, 'rb') as MSA_info:
            MSA_info_dict = pickle.load(MSA_info) 
        #--------------------------------------------------#
        aligned_seqs_len = MSA_info_dict[">seq1"]["seqs_length"] + len(MSA_info_dict[">seq1"]["gaps"])
        print("aligned_seqs_len: ", aligned_seqs_len)
        #--------------------------------------------------#
        X_seqs_all_hiddens_list_MSA = Get_Embeddings_w_MSA(MSA_info_dict, X_seqs_all_hiddens_list, aligned_seqs_len)
        #print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
        #print("len(X_seqs_all_hiddens_list_MSA): ", len(X_seqs_all_hiddens_list_MSA))
        #print("X_seqs_all_hiddens_list[0].shape: ", X_seqs_all_hiddens_list[0].shape)   
        #print("X_seqs_all_hiddens_list_MSA[0].shape: ", X_seqs_all_hiddens_list_MSA[0].shape)
        X_seqs_all_hiddens_list = X_seqs_all_hiddens_list_MSA
        del X_seqs_all_hiddens_list_MSA
        '''







    ###################################################################################################################
    #                     `7MMF'`7MN.   `7MF'`7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                          #
    #                       MM    MMN.    M    MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y                          #
    #                       MM    M YMb   M    MM   ,M9   MM       M       MM      `MMb.                              #
    #                       MM    M  `MN. M    MMmmdM9    MM       M       MM        `YMMNq.                          #
    #                       MM    M   `MM.M    MM         MM       M       MM      .     `MM                          #
    #                       MM    M     YMM    MM         YM.     ,M       MM      Mb     dM                          #
    #                     .JMML..JML.    YM  .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"                           #
    ###################################################################################################################
    # Get embeddings and encodings
    #====================================================================================================#
    if NN_type == "Reg":
        X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, seqs_cmpd_idx_book = \
            Get_represented_X_y_data(X_seqs_all_hiddens_list, 
                                     X_all_seqs_list,
                                     cmpd_properties_list,
                                     X_cmpd_encodings_dict,
                                     screen_bool, 
                                     clf_thrhld_type)



    #====================================================================================================#
    if NN_type == "Clf":
        X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, seqs_cmpd_idx_book = \
            Get_represented_X_y_data_clf(X_seqs_all_hiddens_list, 
                                         X_all_seqs_list,
                                         cmpd_properties_list,
                                         X_cmpd_encodings_dict,
                                         clf_thrhld_type)








    ###################################################################################################################
    #                      `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM  .M"""bgd                           #
    #                        MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 ,MI    "Y                           #
    #                        MM   ,M9   MM   ,M9    MM    M YMb   M       MM      `MMb.                               #
    #                        MMmmdM9    MMmmdM9     MM    M  `MN. M       MM        `YMMNq.                           #
    #                        MM         MM  YM.     MM    M   `MM.M       MM      .     `MM                           #
    #                        MM         MM   `Mb.   MM    M     YMM       MM      Mb     dM                           #
    #                      .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    P"Ybmmd"                            #
    ###################################################################################################################
    # Save seqs_embeddings and cmpd_encodings
    #====================================================================================================#
    #save_dict=dict([])
    #save_dict["X_seqs_all_hiddens"] = X_seqs_all_hiddens
    #save_dict["X_cmpd_encodings"] = X_cmpd_encodings
    #save_dict["y_data"] = y_data
    #pickle.dump( save_dict , open( results_folder / output_file_0, "wb" ) )
    #print("Done getting X_seqs_all_hiddens, X_cmpd_encodings and y_data!")
    print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens), ", len(X_cmpd_encodings): ", len(X_cmpd_encodings), ", len(y_data): ", len(y_data) )



    #====================================================================================================#
    # Get size of some interested parameters.
    #====================================================================================================#
    X_seqs_all_hiddens_dim = [ max([ X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list)) ]), X_seqs_all_hiddens_list[0].shape[1], ]
    X_cmpd_encodings_dim = len(X_cmpd_encodings[0])
    X_seqs_num = len(X_seqs_all_hiddens_list)
    X_cmpd_num = len(cmpd_properties_list)
    print("seqs, cmpd dimensions: ", X_seqs_all_hiddens_dim, ", ", X_cmpd_encodings_dim)
    print("seqs, cmpd counts: ", X_seqs_num, ", ", X_cmpd_num)

    seqs_max_len = max([  X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list))  ])
    print("seqs_max_len: ", seqs_max_len)

    NN_input_dim=X_seqs_all_hiddens_dim[1]
    print("NN_input_dim: ", NN_input_dim)


    # Print the total number of data points.
    #====================================================================================================#
    count_y = 0
    for one_cmpd_properties in cmpd_properties_list:
        for one_y_data in one_cmpd_properties[1]:
            if one_y_data != None:
                count_y+=1
    print("Number of Data Points (#y-values): ", count_y)








    ###################################################################################################################
    #                                  .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM                             #
    #                                 ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7                             #
    #                                 `MMb.       MM   ,M9   MM          MM       MM                                  #
    #                                   `YMMNq.   MMmmdM9    MM          MM       MM                                  #
    #                                 .     `MM   MM         MM      ,   MM       MM                                  #
    #                                 Mb     dM   MM         MM     ,M   MM       MM                                  #
    #                                 P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.                                #
    ###################################################################################################################
    # Get Separate SEQS index and SUBS index.
    #====================================================================================================#
    if split_type == 1:
        tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_idx(X_seqs_num, train_split = 0.7, test_split = 0.2, random_state=seed)
        tr_idx_cmpd, ts_idx_cmpd, va_idx_cmpd = split_idx(X_cmpd_num, train_split = 0.7, test_split = 0.2, random_state=seed)
    elif split_type in [2, 3]:
        tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_idx(X_seqs_num, train_split = 0.8, test_split = 0.1, random_state=seed)
        tr_idx_cmpd, ts_idx_cmpd, va_idx_cmpd = split_idx(X_cmpd_num, train_split = 0.8, test_split = 0.1, random_state=seed)
    elif split_type in [4, 5, 6, 7, 0]:
        tr_idx_seqs, ts_idx_seqs, va_idx_seqs = [], [], []
        tr_idx_cmpd, ts_idx_cmpd, va_idx_cmpd = [], [], []
    else:
        raise Exception("WRONG SPLIT TYPE BRO~")

    #====================================================================================================#
    # Customized SEQ split.

    if split_type == 4:
        #--------------------------------------------------#
        # Get splits.
        tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_seqs_idx_custom(X_seqs_num, 
                                                                      customized_idx_list, 
                                                                      customized_idx_dict, 
                                                                      train_split  = 0.8, 
                                                                      test_split   = 0.1, 
                                                                      random_state = seed, 
                                                                      split_type   = split_type)
    
    if split_type == 5:
        #--------------------------------------------------#
        # Get splits.
        tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_seqs_idx_custom(X_seqs_num, 
                                                                      customized_idx_list, 
                                                                      customized_idx_dict,  
                                                                      train_split  = 0.8, 
                                                                      test_split   = 0.1, 
                                                                      random_state = seed, 
                                                                      split_type   = split_type)
    
    #====================================================================================================#
    # When split_type = 1, 2, 3, 4, 5, use the following tr:ts:va = 8:1:1 split_idx for splitting.

    print("-"*50)
    print("Use tr:ts:va = 8:1:1 splitting on both SEQs & SUBs.")
    print("These are only for split_type = 1, 2, 3, 4, 5 : ")
    print("len(tr_idx_seqs): ", len(tr_idx_seqs) if len(tr_idx_seqs) != 0 else "N/A")
    print("len(ts_idx_seqs): ", len(ts_idx_seqs) if len(ts_idx_seqs) != 0 else "N/A")
    print("len(va_idx_seqs): ", len(va_idx_seqs) if len(va_idx_seqs) != 0 else "N/A")
    print("len(tr_idx_cmpd): ", len(tr_idx_cmpd) if len(tr_idx_cmpd) != 0 else "N/A")
    print("len(ts_idx_cmpd): ", len(ts_idx_cmpd) if len(ts_idx_cmpd) != 0 else "N/A")
    print("len(va_idx_cmpd): ", len(va_idx_cmpd) if len(va_idx_cmpd) != 0 else "N/A")




    #====================================================================================================#
    # Get splitted index of the entire combined dataset.

    X_train_idx, X_test_idx, X_valid_idx = split_seqs_cmpd_idx_book(tr_idx_seqs, 
                                                                    ts_idx_seqs, 
                                                                    va_idx_seqs, 
                                                                    tr_idx_cmpd, 
                                                                    ts_idx_cmpd, 
                                                                    va_idx_cmpd,
                                                                    X_seqs_num,
                                                                    X_cmpd_num, 
                                                                    y_data, 
                                                                    customized_idx_list, # representative sequences
                                                                    customized_idx_dict, # clusters dictionary
                                                                    seqs_cmpd_idx_book, 
                                                                    split_type, 
                                                                    random_state = seed,
                                                                    train_split  = 0.8,  # Used for split_type = 0, 6, 7
                                                                    test_split   = 0.1,  # Used for split_type = 0, 6, 7
                                                                    )

    print("-"*50)
    print("Split of the seqs_cmpd_idx_book: ")
    print("   __________________________________________________ ")
    print("  | seqs_cmpd_idx_book contains all seq-sub-pairs    |")
    print("  | after screening zeros but contains None values,  |")
    print("  | which is removed after splitting the train/test. |")
    print("  |__________________________________________________|")

    seqs_cmpd_pairs_count = len(seqs_cmpd_idx_book)
    print("Number of SEQ&SUB pairs after screening: ", seqs_cmpd_pairs_count)
    print("len(X_train_idx): ", len(X_train_idx))
    print("len(X_test_idx):  ", len(X_test_idx))
    print("len(X_valid_idx): ", len(X_valid_idx))







    ###################################################################################################################
    #          `7MMF'  `7MMF'`7MM"""YMM        db      MMP""MM""YMM `7MMM.     ,MMF'      db      `7MM"""Mq.          #
    #            MM      MM    MM    `7       ;MM:     P'   MM   `7   MMMb    dPMM       ;MM:       MM   `MM.         #
    #            MM      MM    MM   d        ,V^MM.         MM        M YM   ,M MM      ,V^MM.      MM   ,M9          #
    #            MMmmmmmmMM    MMmmMM       ,M  `MM         MM        M  Mb  M' MM     ,M  `MM      MMmmdM9           #
    #            MM      MM    MM   Y  ,    AbmmmqMA        MM        M  YM.P'  MM     AbmmmqMA     MM                #
    #            MM      MM    MM     ,M   A'     VML       MM        M  `YM'   MM    A'     VML    MM                #
    #          .JMML.  .JMML..JMMmmmmMMM .AMA.   .AMMA.   .JMML.    .JML. `'  .JMML..AMA.   .AMMA..JMML.              #
    ###################################################################################################################
    # Visualize Data Split.
    #====================================================================================================#
    print("Getting a categorical heatmap...")
    #====================================================================================================#
    # Stupid Version of getting seqs_cmpd_setlabel_list.


    # num_seqs_plot = 50
    # num_cmpd_plot = 50

    # seqs_cmpd_setlabel_list = []
    # for idx in X_train_idx:
    #     seqs_cmpd_setlabel_list.append([seqs_cmpd_idx_book[idx][0], seqs_cmpd_idx_book[idx][1], 1])
    # for idx in X_test_idx:
    #     seqs_cmpd_setlabel_list.append([seqs_cmpd_idx_book[idx][0], seqs_cmpd_idx_book[idx][1], 2])   
    # for idx in X_valid_idx:
    #     seqs_cmpd_setlabel_list.append([seqs_cmpd_idx_book[idx][0], seqs_cmpd_idx_book[idx][1], 3]) 

    #====================================================================================================#
    # Stupid Version of getting seqs_cmpd_setlabel_list.
    # seqs_cmpd_setlabel_list = []
    # book_slice = seqs_cmpd_idx_book[0:10000] if len(seqs_cmpd_idx_book)>10000 else seqs_cmpd_idx_book
    # for idx, seqs_cmpd_pairs in enumerate(book_slice):
    #     if y_data[idx] is not None:
    #         if idx in X_train_idx:
    #             seqs_cmpd_setlabel_list.append([seqs_cmpd_pairs[0], seqs_cmpd_pairs[1], 1]) # Train    Blue
    #         elif idx in X_test_idx:
    #             seqs_cmpd_setlabel_list.append([seqs_cmpd_pairs[0], seqs_cmpd_pairs[1], 2]) # Test     Red?
    #         elif idx in X_valid_idx:
    #             seqs_cmpd_setlabel_list.append([seqs_cmpd_pairs[0], seqs_cmpd_pairs[1], 3]) # Valid    Yellow?
    #         else:
    #             seqs_cmpd_setlabel_list.append([seqs_cmpd_pairs[0], seqs_cmpd_pairs[1], 4]) # Removed  Green/Grey CD-hit or split task I
    #     else:
    #         seqs_cmpd_setlabel_list.append([seqs_cmpd_pairs[0], seqs_cmpd_pairs[1], 5])     # None     White
    #====================================================================================================#
    # Stupid Version of getting seqs_cmpd_setlabel_list.
    # seqs_cmpd_setlabel_list = []
    # for idx, one_pair in tqdm.tqdm(seqs_cmpd_pairs_slice_idx):
        # if y_data_idx_dict[idx] is not None:
        #     if idx in X_train_idx:
        #         seqs_cmpd_setlabel_list.append([one_pair[0], one_pair[1], 1])  # Train    Blue
        #     elif idx in X_test_idx: 
        #         seqs_cmpd_setlabel_list.append([one_pair[0], one_pair[1], 2])  # Test     Red?
        #     elif idx in X_valid_idx: 
        #         seqs_cmpd_setlabel_list.append([one_pair[0], one_pair[1], 3])  # Valid    Yellow?
        #     else: 
        #         seqs_cmpd_setlabel_list.append([one_pair[0], one_pair[1], 4])  # Removed  Green/Grey CD-hit or split task I
        # else: 
        #     seqs_cmpd_setlabel_list.append([one_pair[0], one_pair[1], 5])      # None     White


    #====================================================================================================#
    '''
    from AP_funcs import cart_prod

    num_seqs_plot = num_cmpd_plot = 188

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Prepare index for plot including their seqs_id and cmpd_id.
    seqs_cmpd_pairs_slice_list = cart_prod([list(range(0,num_seqs_plot)), list(range(0,num_cmpd_plot))])
    seqs_cmpd_pairs_slice_set = set(seqs_cmpd_pairs_slice_list)

    seqs_cmpd_pairs_slice_set_screened = seqs_cmpd_pairs_slice_set.intersection(set([tuple(x) for x in seqs_cmpd_idx_book]))
    
    seqs_cmpd_pairs_slice_list_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_pairs_slice_list]
    seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
    seqs_cmpd_pairs_slice_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, seqs_cmpd_pairs_slice_list_encrypt))[0]))
    
    seqs_cmpd_pairs_slice_idx_extend = []
    seqs_cmpd_plot_dict = {}
    for idx in seqs_cmpd_pairs_slice_idx:
        one_pair = seqs_cmpd_idx_book[idx]
        seqs_cmpd_pairs_slice_idx_extend.append((idx, one_pair))
        seqs_cmpd_pairs_slice_idx_extend.append((idx, one_pair)) #(idx, (one_pair))
        seqs_cmpd_plot_dict[idx] = one_pair    

    seqs_cmpd_plot_idx_list = list(seqs_cmpd_plot_dict.keys())
    seqs_cmpd_plot_idx_set = set(seqs_cmpd_plot_idx_list)



    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Reduce y_data for quick search.
    y_data_idx_dict = {}
    for idx in seqs_cmpd_plot_idx_list:
        y_data_idx_dict[idx] = y_data[idx]



    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Now get labels.
    seqs_cmpd_setlabel_list = []
    for idx in set(X_train_idx).intersection(seqs_cmpd_plot_idx_set):
        if y_data_idx_dict[idx] is not None:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 1])                     # Train         Blue
        else:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 5])  

    for idx in set(X_test_idx).intersection(seqs_cmpd_plot_idx_set):
        if y_data_idx_dict[idx] is not None:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 2])                     # Test          Red
        else:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 5])  
    
    for idx in set(X_valid_idx).intersection(seqs_cmpd_plot_idx_set):
        if y_data_idx_dict[idx] is not None:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 3])                     # Valid         Yellow
        else:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 5])                     # None          White 

    for idx in seqs_cmpd_plot_idx_set.difference(set(X_train_idx)).difference(set(X_test_idx)).difference(set(X_valid_idx)):
        if y_data_idx_dict[idx] is not None:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 4])                     # Removed       Green/Grey CD-hit or split task I
        else:
            seqs_cmpd_setlabel_list.append([seqs_cmpd_plot_dict[idx][0], seqs_cmpd_plot_dict[idx][1], 6])                     # Removed None  lightgreen  

    # The rest are colored BLACK, meaning that those are screened datapoints due to zero values (or super small values)
    


    #====================================================================================================#
    vl     = 0
    vh     = 6
    categorical_heatmap_ZX(seqs_cmpd_setlabel_list, 
                           num_x         = min([num_seqs_plot, X_seqs_num]), 
                           num_y         = min([num_cmpd_plot, X_cmpd_num]), 
                           value_min     = vl,
                           value_max     = vh,
                           c_ticks       = [x + (vh-vl)/(vh+1)/2 for x in np.linspace(vl, vh, num = vh+2)[0:-1]],
                           c_ticklabels  = ["Unwanted\nZeroes", "Train", "Test", "Valid", "Removed", "No Values", "Removed\nNone"],
                           plot_title    = 'Categorical HeatMap Showing Data Splitting',
                           x_label       = "Sequence Index",
                           y_label       = "Compound Index",
                           cmap          = mpl.colors.ListedColormap(['black', 'royalblue', 'darkred', 'gold', 'darkgreen', 'white', (182/255,251/255,182/255)]),
                           font_size     = 15,
                           result_folder = results_sub_folder,
                           file_name     = "Data_Split" + ".png"
                           )    
                           '''







    ###################################################################################################################
   #               .g8""8q.   `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd              #
   #             .dP'    `YM.   MM       M  P'   MM   `7   MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y              #
   #             dM'      `MM   MM       M       MM        MM   ,M9   MM       M       MM      `MMb.                  #
   #             MM        MM   MM       M       MM        MMmmdM9    MM       M       MM        `YMMNq.              #
   #             MM.      ,MP   MM       M       MM        MM         MM       M       MM      .     `MM              #
   #             `Mb.    ,dP'   YM.     ,M       MM        MM         YM.     ,M       MM      Mb     dM              #
   #               `"bmmd"'      `bmmmmd"'     .JMML.    .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"               #
    ###################################################################################################################
    # Get splitted data of the combined dataset using the splitted index.
    #====================================================================================================#
    X_tr_seqs_emb, X_tr_seqs, X_tr_cmpd, X_tr_smiles, y_tr = Get_X_y_data_selected(X_train_idx, X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, log_value)
    X_ts_seqs_emb, X_ts_seqs, X_ts_cmpd, X_ts_smiles, y_ts = Get_X_y_data_selected(X_test_idx , X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, log_value)
    X_va_seqs_emb, X_va_seqs, X_va_cmpd, X_va_smiles, y_va = Get_X_y_data_selected(X_valid_idx, X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, log_value)
    y_scalar = None


    if log_value == False:
        y_tr, y_scalar = normalize_targets(y_tr)
        y_ts = y_scalar.transform(y_ts)
        y_va = y_scalar.transform(y_va)

        y_tr = np.array(y_tr, dtype = np.float32)
        y_ts = np.array(y_ts, dtype = np.float32)
        y_va = np.array(y_va, dtype = np.float32)


    #print("Done getting X_data and y_data!")
    print("-"*50)
    print("Final results of splitting: ")
    print("X_tr_seqs_emb_dimension: ", len(X_tr_seqs_emb), ", X_tr_cmpd_dimension: ", len(X_tr_cmpd), ", y_tr_dimension: ", y_tr.shape )
    print("X_ts_seqs_emb_dimension: ", len(X_ts_seqs_emb), ", X_ts_cmpd_dimension: ", len(X_ts_cmpd), ", y_ts_dimension: ", y_ts.shape )
    print("X_va_seqs_emb_dimension: ", len(X_va_seqs_emb), ", X_va_cmpd_dimension: ", len(X_va_cmpd), ", y_va_dimension: ", y_va.shape )
    print("="*80)




    return X_tr_seqs_emb, X_tr_seqs, X_tr_cmpd, X_tr_smiles, y_tr, \
           X_ts_seqs_emb, X_ts_seqs, X_ts_cmpd, X_ts_smiles, y_ts, \
           X_va_seqs_emb, X_va_seqs, X_va_cmpd, X_va_smiles, y_va, \
           X_seqs_all_hiddens_dim, X_cmpd_encodings_dim, seqs_max_len, NN_input_dim, y_scalar








