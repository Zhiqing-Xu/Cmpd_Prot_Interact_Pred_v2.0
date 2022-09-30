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
#########################################################################################################
#########################################################################################################
import sys
import time
import copy
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
from torch.nn import MSELoss
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
#------------------------------
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
#------------------------------
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import seaborn as sns
import matplotlib.pyplot as plt
#--------------------------------------------------#
from copy import deepcopy
from tpot import TPOTRegressor
from scipy import stats

from pathlib import Path
from ipywidgets import IntProgress
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
from AP_convert import Get_Unique_SMILES, MolFromSmiles_ZX
#------------------------------
from Z05_utils import *
from Z05_split_data import *
from Z05_run_train import *
#------------------------------
from ZX02_nn_utils import StandardScaler
from ZX02_nn_utils import build_optimizer, build_lr_scheduler
from ZX03_nn_args import TrainArgs
from ZX04_funcs import onek_encoding_unk
from ZX05_loss_functions import get_loss_func

#------------------------------
from Z05G_Cpd_Data import *
from Z05G_Conv_Mpnn import *

#--------------------------------------------------#








#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                    `7MMF'`7MN.   `7MF'`7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                                                #
#                      MM    MMN.    M    MM   `MM.  MM       M  P'   MM   `7 ,MI    "Y                                                                #
#   ,pP""Yq.           MM    M YMb   M    MM   ,M9   MM       M       MM      `MMb.                                                                    #
#  6W'    `Wb          MM    M  `MN. M    MMmmdM9    MM       M       MM        `YMMNq.                                                                #
#  8M      M8          MM    M   `MM.M    MM         MM       M       MM      .     `MM                                                                #
#  YA.    ,A9 ,,       MM    M     YMM    MM         YM.     ,M       MM      Mb     dM                                                                #
#   `Ybmmd9'  db     .JMML..JML.    YM  .JMML.        `bmmmmd"'     .JMML.    P"Ybmmd"                                                                 #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 


if __name__ == "__main__":
    # Everything starts here!
    ##############################################################################################################
    ##############################################################################################################
    ## All Input Arguments
    Step_code        = "X05G_"
    dataset_nme_list = ["phosphatase",        # 0
                        "kinase",             # 1
                        "esterase",           # 2
                        "halogenase",         # 3
                        "aminotransferase",   # 4
                        "kcat_c",             # 5
                        "kcat",               # 6
                        "kcat_mt",            # 7
                        "kcat_wt",            # 8
                        "Ki_all_org",         # 9
                        "Ki_small",           # 10
                        "Ki_select",          # 11
                        "KM_BRENDA",          # 12
                        ] 
    dataset_nme      = dataset_nme_list[12]
    data_folder      = Path("X_DataProcessing/")
    properties_file  = "X00_" + dataset_nme + "_compounds_properties_list.p"
    MSA_info_file    = "X03_" + dataset_nme + "_MSA_info.p"
    cstm_splt_file   = "X03_" + dataset_nme + "_cstm_splt.p"
    #--------------------------------------------------#
    encoding_file_list = ["X02A_" + dataset_nme + "_ECFP2_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_ECFP4_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_ECFP6_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_JTVAE_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_MorganFP_encodings_dict.p",]
    encoding_file = encoding_file_list[2]
    #--------------------------------------------------#
    seqs_fasta_file     =  "X00_" + dataset_nme + ".fasta"
    embedding_file_list = ["X03_" + dataset_nme + "_embedding_ESM_1B.p", 
                           "X03_" + dataset_nme + "_embedding_BERT.p", 
                           "X03_" + dataset_nme + "_embedding_TAPE.p", 
                           "X03_" + dataset_nme + "_embedding_ALBERT.p", 
                           "X03_" + dataset_nme + "_embedding_T5.p", 
                           "X03_" + dataset_nme + "_embedding_TAPE_FT.p", 
                           "X03_" + dataset_nme + "_embedding_Xlnet.p",]
    embedding_file = embedding_file_list[0]
    #--------------------------------------------------#
    # embedding_file is a dict, {"seqs_embeddings":seqs_embeddings, "seqs_ids":seqs_ids, "seqs_all_hiddens":seqs_all_hiddens}
    # properties_file is a list, [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
    #====================================================================================================#
    # Select properties (Y) of the model 
    screen_bool     = False
    clf_thrhld_type = 2 # 2: 1e-5, 3: 1e-2
    log_value       = bool(0) ##### !!!!! If value is True, screen_bool will be changed
    MSA_bool        = False
    #====================================================================================================#
    # Data Split Methods
    split_type   = 0  # 0, 1, 2, 3, 4, 5, 6, 7
    # split_type = 0, train/test split completely randomly selected
    # split_type = 1, train/test split contains different seq-subs pairs
    # split_type = 2, train/test split contains different seqs
    # split_type = 3, train/test split contains different subs
    # split_type = 4, train/test split contains different CD-hit seqs, train contains non-CD-hit seqs.
    # split_type = 5, train/test split contains different CD-hit seqs
    # split_type = 6, train/test split completely randomly selected, with non-CD-hit seqs data all contained in train.
    # split_type = 7, train/test split completely randomly selected, with non-CD-hit seqs data being left out.
    #====================================================================================================#
    # Prediction NN settings
    NN_type_list   = ["Reg", "Clf"]
    NN_type        = NN_type_list[0]
    epoch_num      = 300
    batch_size     = 192
    learning_rate  =  [0.01        , # 0
                       0.005       , # 1
                       0.002       , # 2
                       0.001       , # 3
                       0.0005      , # 4
                       0.0002      , # 5
                       0.0001      , # 6
                       0.00005     , # 7
                       0.00002     , # 8
                       0.00001     , # 8
                       0.000005    , # 10
                       0.000002    , # 11
                       0.000001    , # 12
                       ][2]          # 

    #====================================================================================================#
    hid_dim    = 384    # 256
    kernal_1   = 3      # 5
    out_dim    = 1      # 2
    kernal_2   = 3      # 3
    last_hid   = 1440   # 1024
    dropout    = 0.1     # 0
    #====================================================================================================#
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #====================================================================================================#
    hyperparameters_dict = dict([])
    for one_hyperpara in ["hid_dim", "kernal_1", "out_dim", "kernal_2", "last_hid", "dropout"]:
        hyperparameters_dict[one_hyperpara] = locals()[one_hyperpara]
    #====================================================================================================#
    # If log_value is True, screen_bool will be changed.
    if log_value==True:
        screen_bool = True
    #--------------------------------------------------#
    # If value is "Clf", log_value will be changed.
    if NN_type=="Clf":
        screen_bool = False # Actually Useless.
        log_value = False
    #--------------------------------------------------#
    if os.name == 'nt' or platform == 'win32':
        pass
    else:
        print("Running on Linux, change epoch number to 200")
        epoch_num = 200
    #====================================================================================================#
    # Results
    results_folder = Path("X_DataProcessing/" + Step_code +"intermediate_results/")
    output_file_0 = Step_code + "_all_X_y.p"
    output_file_header = Step_code + "_result_"

    print("\n" + "="*80)
    #====================================================================================================#
    # Initialize argument.
    arguments = ['--data_path'       ,  ''                       ,
                 '--dataset_type'    ,  'regression'             ,
                 '--hidden_size'     ,  '1000'                   ,
                 '--features_path'   ,  '--no_features_scaling'  ,
                 #'--features_only'  ,  ''                       ,
                 #'--atom_messages'  , 
                 #'--undirected'     ,

                ]

    # Use the "tap" package here, better than argparse
    args = TrainArgs().parse_args(arguments)
    print(args)





    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #               `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM  `7MM"""Mq.       db      `7MM"""Mq.        db      MMP""MM""YMM `7MMF'  .g8""8q.   `7MN.   `7MF'    #
    #   __,           MM   `MM.  MM   `MM.   MM    `7    MM   `MM.     ;MM:       MM   `MM.      ;MM:     P'   MM   `7   MM  .dP'    `YM.   MMN.    M      #
    #  `7MM           MM   ,M9   MM   ,M9    MM   d      MM   ,M9     ,V^MM.      MM   ,M9      ,V^MM.         MM        MM  dM'      `MM   M YMb   M      #
    #    MM           MMmmdM9    MMmmdM9     MMmmMM      MMmmdM9     ,M  `MM      MMmmdM9      ,M  `MM         MM        MM  MM        MM   M  `MN. M      #
    #    MM           MM         MM  YM.     MM   Y  ,   MM          AbmmmqMA     MM  YM.      AbmmmqMA        MM        MM  MM.      ,MP   M   `MM.M      #
    #    MM  ,,       MM         MM   `Mb.   MM     ,M   MM         A'     VML    MM   `Mb.   A'     VML       MM        MM  `Mb.    ,dP'   M     YMM      #
    #  .JMML.db     .JMML.     .JMML. .JMM..JMMmmmmMMM .JMML.     .AMA.   .AMMA..JMML. .JMM..AMA.   .AMMA.   .JMML.    .JMML.  `"bmmd"'   .JML.    YM      #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
    # Step 1. Create temp folder for results.
    results_sub_folder = Create_Temp_Folder(results_folder, 
                                               encoding_file, 
                                               embedding_file, 
                                               dataset_nme, 
                                               Step_code, 
                                               NN_type, 
                                               split_type, 
                                               screen_bool, 
                                               log_value, 
                                               clf_thrhld_type,
                                               MSA_bool)





    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #                .g8""8q.   `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq. `7MMF'   `7MF'MMP""MM""YMM    `7MM"""Mq. `7MM"""Mq.  `7MMF'`7MN.   `7MF'MMP""MM""YMM #
    #              .dP'    `YM.   MM       M  P'   MM   `7   MM   `MM.  MM       M  P'   MM   `7      MM   `MM.  MM   `MM.   MM    MMN.    M  P'   MM   `7 #
    #  pd*"*b.     dM'      `MM   MM       M       MM        MM   ,M9   MM       M       MM           MM   ,M9   MM   ,M9    MM    M YMb   M       MM      #
    # (O)   j8     MM        MM   MM       M       MM        MMmmdM9    MM       M       MM           MMmmdM9    MMmmdM9     MM    M  `MN. M       MM      #
    #     ,;j9     MM.      ,MP   MM       M       MM        MM         MM       M       MM           MM         MM  YM.     MM    M   `MM.M       MM      #
    #  ,-='    ,,  `Mb.    ,dP'   YM.     ,M       MM        MM         YM.     ,M       MM           MM         MM   `Mb.   MM    M     YMM       MM      #
    # Ammmmmmm db    `"bmmd"'      `bmmmmd"'     .JMML.    .JMML.        `bmmmmd"'     .JMML.       .JMML.     .JMML. .JMM..JMML..JML.    YM     .JMML.    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$# 
    # Step 2. Output print (to print details of the model including,
    # dimensions of dataset, # of seqs, # of subs and hyperparameters of the model).
    output_print(dataset_nme,
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
                 hyperparameters_dict)





    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #   pd""b.          `7MM"""Yb.      db  MMP""MM""YMM  db           .M"""bgd `7MM"""Mq.`7MMF'      `7MMF'MMP""MM""YMM                                   #
    #  (O)  `8b           MM    `Yb.   ;MM: P'   MM   `7 ;MM:         ,MI    "Y   MM   `MM. MM          MM  P'   MM   `7                                   #
    #       ,89           MM     `Mb  ,V^MM.     MM     ,V^MM.        `MMb.       MM   ,M9  MM          MM       MM                                        #
    #     ""Yb.           MM      MM ,M  `MM     MM    ,M  `MM          `YMMNq.   MMmmdM9   MM          MM       MM                                        #
    #        88           MM     ,MP AbmmmqMA    MM    AbmmmqMA       .     `MM   MM        MM      ,   MM       MM                                        #
    #  (O)  .M'   ,,      MM    ,dP'A'     VML   MM   A'     VML      Mb     dM   MM        MM     ,M   MM       MM                                        #
    #   bmmmd'    db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.    P"Ybmmd"  .JMML.    .JMMmmmmMMM .JMML.   .JMML.                                      #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#    
    # Step 3. Get all dataset for train/test/validate model (data split).
    # If random split, split_type = 0, go to Z05_split_data to adjust split ratio. (Find split_seqs_cmpd_idx_book())
    # If split_type = 1, 2 or 3, go to Z05_utils to adjust split ratio. (Find split_idx())
    X_tr_seqs_emb, X_tr_seqs, X_tr_cmpd, X_tr_smls, y_tr, \
    X_ts_seqs_emb, X_ts_seqs, X_ts_cmpd, X_ts_smls, y_ts, \
    X_va_seqs_emb, X_va_seqs, X_va_cmpd, X_va_smls, y_va, \
    X_seqs_all_hiddens_dim, X_cmpd_encodings_dim, seqs_max_len, NN_input_dim, y_scalar = \
        tr_ts_va_for_NN(dataset_nme,
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
                        seed)






    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #       ,AM         `7MM"""Yb.      db  MMP""MM""YMM  db       .M"""bgd `7MM"""YMM MMP""MM""YMM                                                        #
    #      AVMM           MM    `Yb.   ;MM: P'   MM   `7 ;MM:     ,MI    "Y   MM    `7 P'   MM   `7                                                        #
    #    ,W' MM           MM     `Mb  ,V^MM.     MM     ,V^MM.    `MMb.       MM   d        MM                                                             #
    #  ,W'   MM           MM      MM ,M  `MM     MM    ,M  `MM      `YMMNq.   MMmmMM        MM                                                             #
    #  AmmmmmMMmm         MM     ,MP AbmmmqMA    MM    AbmmmqMA   .     `MM   MM   Y  ,     MM                                                             #
    #        MM   ,,      MM    ,dP'A'     VML   MM   A'     VML  Mb     dM   MM     ,M     MM                                                             #
    #        MM   db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.P"Ybmmd"  .JMMmmmmMMM   .JMML.                                                           #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#    
    # Step 4. Prepare smiles for molecule processing MPNN.
    #print("len(X_tr_smiles): ", len(X_tr_smiles))
    #print("For testing smiles input only, features shall be input using functions in HG_rdkit.. ")

    X_tr_smiles_dataset, X_ts_smiles_dataset, X_va_smiles_dataset = \
        Z05G_Cpd_Data(X_tr_smls, 
                      X_ts_smls, 
                      X_va_smls, 
                      X_tr_cmpd  , 
                      X_ts_cmpd  , 
                      X_va_cmpd  , 
                      args        )






    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #   M******         `7MM"""Yb.      db  MMP""MM""YMM  db     `7MMF'        .g8""8q.      db     `7MM"""Yb. `7MM"""YMM  `7MM"""Mq.                      #
    #  .M                 MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM        .dP'    `YM.   ;MM:      MM    `Yb. MM    `7    MM   `MM.                     #
    #  |bMMAg.            MM     `Mb  ,V^MM.     MM     ,V^MM.     MM        dM'      `MM  ,V^MM.     MM     `Mb MM   d      MM   ,M9                      #
    #       `Mb           MM      MM ,M  `MM     MM    ,M  `MM     MM        MM        MM ,M  `MM     MM      MM MMmmMM      MMmmdM9                       #
    #        jM           MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM      , MM.      ,MP AbmmmqMA    MM     ,MP MM   Y  ,   MM  YM.                       #
    #  (O)  ,M9   ,,      MM    ,dP'A'     VML   MM   A'     VML   MM     ,M `Mb.    ,dP'A'     VML   MM    ,dP' MM     ,M   MM   `Mb.                     #
    #   6mmm9     db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMMmmmmMMM   `"bmmd"'.AMA.   .AMMA.JMMmmmdP' .JMMmmmmMMM .JMML. .JMM.                    #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    # Step 5. Call DataLoader.
    train_loader, valid_loader, test_loader = \
        get_SQemb_CPmpn_pairs(X_tr_seqs_emb          , 
                               X_tr_smiles_dataset   , 
                               y_tr                  , 
                               X_va_seqs_emb         , 
                               X_va_smiles_dataset   , 
                               y_va                  , 
                               X_ts_seqs_emb         , 
                               X_ts_smiles_dataset   , 
                               y_ts                  , 
                               seqs_max_len          , 
                               batch_size            ,
                               args)






    loss_func = get_loss_func(args)
    print("loss_func: ", loss_func)

    # optimizer = build_optimizer(model, args)
    # Learning rate schedulers
    # scheduler = build_lr_scheduler(optimizer, args)





    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #     .6*"            .g8"""bgd       db      `7MMF'      `7MMF'          `7MMM.     ,MMF'  .g8""8q.   `7MM"""Yb.   `7MM"""YMM  `7MMF'                 #
    #   ,M'             .dP'     `M      ;MM:       MM          MM              MMMb    dPMM  .dP'    `YM.   MM    `Yb.   MM    `7    MM                   #
    #  ,Mbmmm.          dM'       `     ,V^MM.      MM          MM              M YM   ,M MM  dM'      `MM   MM     `Mb   MM   d      MM                   #
    #  6M'  `Mb.        MM             ,M  `MM      MM          MM              M  Mb  M' MM  MM        MM   MM      MM   MMmmMM      MM                   #
    #  MI     M8        MM.            AbmmmqMA     MM      ,   MM      ,       M  YM.P'  MM  MM.      ,MP   MM     ,MP   MM   Y  ,   MM      ,            #
    #  WM.   ,M9 ,,     `Mb.     ,'   A'     VML    MM     ,M   MM     ,M       M  `YM'   MM  `Mb.    ,dP'   MM    ,dP'   MM     ,M   MM     ,M            #
    #   WMbmmd9  db       `"bmmmd'  .AMA.   .AMMA..JMMmmmmMMM .JMMmmmmMMM     .JML. `'  .JMML.  `"bmmd"'   .JMMmmmdP'   .JMMmmmmMMM .JMMmmmmMMM            #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    # Step 6. Create NN model and Initialize
    print("\n\n\n>>> Initializing the model... ")
    #====================================================================================================#
    # Get the model (CNN)
    model = SQembConv_CPmpn_Model(args     = args,
                                  in_dim   = NN_input_dim,
                                  hid_dim  = hid_dim,
                                  kernal_1 = kernal_1,
                                  out_dim  = out_dim, #2
                                  kernal_2 = kernal_2,
                                  max_len  = seqs_max_len,
                                  cmpd_dim = X_cmpd_encodings_dim,
                                  last_hid = last_hid, #256
                                  dropout  = dropout
                                  )

    print("Count Model Parameters: ", param_count_all(model))
    #initialize_weights(model)

    #--------------------------------------------------#
    model.double()
    model.cuda()
    #--------------------------------------------------#
    print("#"*80)
    print(model)
    #model.float()
    #print( summary( model,[(seqs_max_len, NN_input_dim), (X_cmpd_encodings_dim, )] )  )
    #model.double()
    print("#"*80)
    #--------------------------------------------------#
    # Model Hyperparaters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    max_r = \
        run_train(model                =  model         , 
                  optimizer            =  optimizer     , 
                  criterion            =  criterion     , 
                  epoch_num            =  epoch_num     , 
                  train_loader         =  train_loader  , 
                  valid_loader         =  valid_loader  , 
                  test_loader          =  test_loader   , 
                  y_scalar             =  y_scalar      ,
                  log_value            =  log_value     ,
                  screen_bool          =  screen_bool   , 
                  results_sub_folder   =  results_sub_folder , 
                  output_file_header   =  output_file_header , 
                  input_var_names_list =  ["seqs_embeddings" , "cmpd_dataset"] , 
                  target_name          =  "y_property"       , 
                  )



    #########################################################################################################
    #########################################################################################################
    print("="*80)
    print("max_r: ", max(max_r))










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




