# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:38:41 2022

@author: Rana
"""

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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
#--------------------------------------------------#
from copy import deepcopy
from tpot import TPOTRegressor
from pathlib import Path
from ipywidgets import IntProgress
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
from Z05_utils import *
from Z05_split_data import *
from Z05_run_train import run_train

from Z05A_Conv import *
from Z05L_LSTM import *

#--------------------------------------------------#
from ZX02_nn_utils import *


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
    Step_code        = "X05L_"
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
    dataset_nme      = dataset_nme_list[6]
    data_folder      = Path("X_DataProcessing/")
    properties_file  = "X00_" + dataset_nme + "_compounds_properties_list.p"
    MSA_info_file    = "X03_" + dataset_nme + "_MSA_info.p"
    cstm_splt_file   = "X03_" + dataset_nme + "_cstm_splt.p"
    #--------------------------------------------------#
    encoding_file_list = ["X02A_" + dataset_nme + "_ECFP2_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_ECFP4_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_ECFP6_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_JTVAE_encodings_dict.p", 
                          "X02A_" + dataset_nme + "_MorganFP_encodings_dict.p",
                          "X02B_" + dataset_nme + "_MgFP6_encodings_dict.p"]
    encoding_file = encoding_file_list[5]
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
    log_value       = bool(1) ##### !!!!! If value is True, screen_bool will be changed
    MSA_bool        = False
    #====================================================================================================#
    # Data Split Methods
    split_type   = 0  # 0, 1, 2, 3, 4, 5, 6, 7
    # split_type = 0, train/test split completely randomly selected
    # split_type = 1, train/test split contains different seqs-cmpd pairs
    # split_type = 2, train/test split contains different seqs
    # split_type = 3, train/test split contains different cmpd
    # split_type = 4, train/test split contains different CD-hit seqs, train contains non-CD-hit seqs.
    # split_type = 5, train/test split contains different CD-hit seqs
    # split_type = 6, train/test split completely randomly selected, with each non-CD-hit seqs data send to its representative sequence's split.
    # split_type = 7, train/test split completely randomly selected, with non-CD-hit seqs data being left out.
    #====================================================================================================#
    # Prediction NN settings
    NN_type_list   = ["Reg", "Clf"]
    NN_type        = NN_type_list[0]
    epoch_num      = 50
    batch_size     = 32
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
                       ][6]          # 

    #====================================================================================================#
    hid_dim    = 256    # 256
    latent_dim   = 244      # 244
    out_dim    = 1      # 2
    num_layers   = 1      # 3
    last_hid = 1024 # 1024
    dropout    = 0.1     # 0
    #====================================================================================================#
    seed = 42 # 42, 0, 1, 2, 3, 4
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #====================================================================================================#
    hyperparameters_dict = dict([])
    for one_hyperpara in ["hid_dim", "latent_dim", "out_dim", "num_layers", "last_hid", "dropout"]:
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
        print("Running on Linux, change epoch number to 100")
        epoch_num = 100
    #====================================================================================================#
    # Results
    results_folder = Path("X_DataProcessing/" + Step_code +"intermediate_results/")
    output_file_0 = Step_code + "_all_X_y.p"
    output_file_header = Step_code + "_result_"

    ##############################################################################################################
    ##############################################################################################################
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

    ##############################################################################################################
    ##############################################################################################################
    # Step 2. Output print (to print details of the model including,
    # dimensions of dataset, # of seqs, # of cmpd and hyperparameters of the model).
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


    ##############################################################################################################
    ##############################################################################################################
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


    ##############################################################################################################
    ##############################################################################################################
    # Step 4. Call DataLoader.
    train_loader, valid_loader, test_loader = \
        generate_LSTM_loader(X_tr_seqs_emb, 
                            X_tr_cmpd, 
                            y_tr, 
                            X_va_seqs_emb, 
                            X_va_cmpd, 
                            y_va, 
                            X_ts_seqs_emb, 
                            X_ts_cmpd, 
                            y_ts, 
                            seqs_max_len, 
                            batch_size)

    #########################################################################################################
    #########################################################################################################
    # Step 5. Create NN model and Initiate
    print("\n\n\n>>> Initializing the model... ")
    #====================================================================================================#
    # Get the model (LSTM)
    model = SQembLSTM_CPenc_Model(in_dim   = NN_input_dim,
                                  hid_dim  = hid_dim,
                                  latent_dim = latent_dim,
                                  out_dim  = out_dim, #2
                                  cmpd_dim = X_cmpd_encodings_dim,
                                  max_len = seqs_max_len,
                                  num_layers = num_layers,
                                  last_hid  = last_hid,
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
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss()


    ###################################################################################################################
    ###################################################################################################################
    # Step 6. Now, train the model
    print("\n\n\n>>>  Training... ")
    print("="*80)

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
                  input_var_names_list =  ["seqs_embeddings" , "seqs_lens", "cmpd_encodings"] , 
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










