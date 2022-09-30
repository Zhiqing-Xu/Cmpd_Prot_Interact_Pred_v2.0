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
import re
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
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
#--------------------------------------------------#
from Bio import SeqIO
from tqdm import tqdm
from copy import deepcopy
from tpot import TPOTRegressor
from pathlib import Path
from ipywidgets import IntProgress
from collections import defaultdict
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
from Z05H_DL_kcat import *
#------------------------------
from Z05_utils import *
from Z05_split_data import *
#------------------------------
from ZX02_nn_utils import *
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
    Step_code        = "X05H_"
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
    clf_thrhld_type = 2      # 2: 1e-5, 3: 1e-2
    log_value       = True   ##### !!!!! If value is True, screen_bool will be changed
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
    # Not Used for KNN.
    NN_type_list   = ["Reg", "Clf"]
    NN_type        = NN_type_list[0]
    epoch_num      =  0
    batch_size     =  0
    learning_rate  =  0

    #====================================================================================================#
    # Not Used for KNN.
    hid_dim    = 0   
    kernal_1   = 0   
    out_dim    = 0   
    kernal_2   = 0   
    last_hid   = 0   
    dropout    = 0
    #====================================================================================================#
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    # Not Used for KNN.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #====================================================================================================#
    hyperparameters_dict = dict([])
    for one_hyperpara in ["dropout"]:
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
        epoch_num = 0
    #====================================================================================================#
    # Results
    results_folder = Path("X_DataProcessing/" + Step_code +"intermediate_results/")
    output_file_0 = Step_code + "_all_X_y.p"
    output_file_header = Step_code + "_result_"





    
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
    #       ,AM         .M"""bgd      db `7MMF'   `7MF'`7MM"""YMM     MMP""MM""YMM `7MMF'  `7MMF'`7MM"""YMM     `7MM"""Yb.      db  MMP""MM""YMM  db       #
    #      AVMM        ,MI    "Y     ;MM:  `MA     ,V    MM    `7     P'   MM   `7   MM      MM    MM    `7       MM    `Yb.   ;MM: P'   MM   `7 ;MM:      #
    #    ,W' MM        `MMb.        ,V^MM.  VM:   ,V     MM   d            MM        MM      MM    MM   d         MM     `Mb  ,V^MM.     MM     ,V^MM.     #
    #  ,W'   MM          `YMMNq.   ,M  `MM   MM.  M'     MMmmMM            MM        MMmmmmmmMM    MMmmMM         MM      MM ,M  `MM     MM    ,M  `MM     #
    #  AmmmmmMMmm      .     `MM   AbmmmqMA  `MM A'      MM   Y  ,         MM        MM      MM    MM   Y  ,      MM     ,MP AbmmmqMA    MM    AbmmmqMA    #
    #        MM   ,,   Mb     dM  A'     VML  :MM;       MM     ,M         MM        MM      MM    MM     ,M      MM    ,dP'A'     VML   MM   A'     VML   #
    #        MM   db   P"Ybmmd" .AMA.   .AMMA. VF      .JMMmmmmMMM       .JMML.    .JMML.  .JMML..JMMmmmmMMM    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA. #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  

    # Step 4. Dump.
    all_data_split_dict = dict([])
    all_data_split_dict_file_output = Step_code + dataset_nme + "_SpltType_" + str(split_type) + "_data_splt.p"

    if log_value == False:
        y_tr = y_scalar.inverse_transform(y_tr)
        y_ts = y_scalar.inverse_transform(y_ts)
        y_va = y_scalar.inverse_transform(y_va)

    if log_value == True:
        y_tr = np.power(10, y_tr)
        y_ts = np.power(10, y_ts)
        y_va = np.power(10, y_va)

    all_data_split_dict["X_tr_seqs"] = X_tr_seqs           # sequences in training set.
    all_data_split_dict["X_ts_seqs"] = X_ts_seqs           # sequences in test set.
    all_data_split_dict["X_va_seqs"] = X_va_seqs           # sequences in validation set.

    #all_data_split_dict["X_tr_seqs_emb"] = X_tr_seqs_emb   # seqs embeddings in training set.
    #all_data_split_dict["X_ts_seqs_emb"] = X_ts_seqs_emb   # seqs embeddings in test set.
    #all_data_split_dict["X_va_seqs_emb"] = X_va_seqs_emb   # seqs embeddings in validation set.

    all_data_split_dict["X_tr_cmpd"] = X_tr_cmpd           # ECFP6 count encodings in training set.
    all_data_split_dict["X_ts_cmpd"] = X_ts_cmpd           # ECFP6 count encodings in test set.
    all_data_split_dict["X_va_cmpd"] = X_va_cmpd           # ECFP6 count encodings in validation set.

    all_data_split_dict["X_tr_smls"] = X_tr_smls           # SMILES in training set.
    all_data_split_dict["X_ts_smls"] = X_ts_smls           # SMILES in test set.
    all_data_split_dict["X_va_smls"] = X_va_smls           # SMILES in validation set.

    all_data_split_dict["y_tr"] = y_tr                     # target values in training set.
    all_data_split_dict["y_ts"] = y_ts                     # target values in test set.
    all_data_split_dict["y_va"] = y_va                     # target values in validation set.



    pickle.dump(all_data_split_dict, open(data_folder / all_data_split_dict_file_output, "wb"))
    
    





    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
    #   M******            `7MM"""Yb.   `7MMF'          `7MM                              mm                                                               #
    #  .M                    MM    `Yb.   MM              MM                              MM                                                               #
    #  |bMMAg.               MM     `Mb   MM              MM  ,MP'     ,p6"bo   ,6"Yb.  mmMMmm                                                             #
    #       `Mb              MM      MM   MM              MM ;Y       6M'  OO  8)   MM    MM                                                               #
    #        jM              MM     ,MP   MM      ,       MM;Mm       8M        ,pm9MM    MM                                                               #
    #  (O)  ,M9   ,,         MM    ,dP'   MM     ,M       MM `Mb.     YM.    , 8M   MM    MM                                                               #
    #   6mmm9     db       .JMMmmmdP'   .JMMmmmmMMM     .JMML. YA.     YMbmd'  `Moo9^Yo.  `Mbmo                                                            #
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#








    #====================================================================================================#
    #                        `7MMF'                              `7MM  
    #               __,        MM                                  MM  
    #   M******    `7MM        MM         ,pW"Wq.   ,6"Yb.    ,M""bMM  
    #  .M            MM        MM        6W'   `Wb 8)   MM  ,AP    MM  
    #  |bMMAg.       MM        MM      , 8M     M8  ,pm9MM  8MI    MM  
    #       `Mb ,,   MM        MM     ,M YA.   ,A9 8M   MM  `Mb    MM  
    #        jM db .JMML.    .JMMmmmmMMM  `Ybmd9'  `Moo9^Yo. `Wbmd"MML.
    #  (O)  ,M9                                                        
    #   6mmm9                                                          
    #====================================================================================================#

    all_data_split_dict_file_output = Step_code + dataset_nme + "_SpltType_" + str(split_type) + "_data_splt.p"
    all_data_split_dict = pickle.load(open(data_folder / all_data_split_dict_file_output, "rb"))
    
    X_tr_seqs  =  all_data_split_dict["X_tr_seqs"]
    X_ts_seqs  =  all_data_split_dict["X_ts_seqs"]
    X_va_seqs  =  all_data_split_dict["X_va_seqs"]
    X_tr_smls  =  all_data_split_dict["X_tr_smls"]
    X_ts_smls  =  all_data_split_dict["X_ts_smls"]
    X_va_smls  =  all_data_split_dict["X_va_smls"]
    y_tr = all_data_split_dict["y_tr"]
    y_ts = all_data_split_dict["y_ts"]
    y_va = all_data_split_dict["y_va"]

    all_seqs = X_tr_seqs+X_ts_seqs+X_va_seqs
    all_smls = X_tr_smls+X_ts_smls+X_va_smls
    all_y    = np.concatenate([y_tr, y_ts, y_va])

    print("len(all_seqs  ): ", len(all_seqs  ))
    print("len(all_smls  ): ", len(all_smls  ))
    print("len(all_y     ): ", len(all_y     ))    

    print("all_seqs [6]): ", all_seqs [6])
    print("all_smls [6]): ", all_smls [6])
    print("all_y    [6]): ", all_y    [6])   

    print("all_seqs [11118]): ", all_seqs [11118])
    print("all_smls [11118]): ", all_smls [11118])
    print("all_y    [11118]): ", all_y    [11118]) 

    print("all_seqs [119]): ", all_seqs [119])
    print("all_smls [119]): ", all_smls [119])
    print("all_y    [119]): ", all_y    [119]) 


    # Step 5. Create NN model and Initiate
    print("\n\n\n>>> Initializing the model... ")





    #====================================================================================================#
    #                            .M"""bgd            mm      mm      db                                
    #                           ,MI    "Y            MM      MM                                        
    #   M******     pd*"*b.     `MMb.      .gP"Ya  mmMMmm  mmMMmm  `7MM  `7MMpMMMb.   .P"Ybmmm ,pP"Ybd 
    #  .M          (O)   j8       `YMMNq. ,M'   Yb   MM      MM      MM    MM    MM  :MI  I8   8I   `" 
    #  |bMMAg.         ,;j9     .     `MM 8M""""""   MM      MM      MM    MM    MM   WmmmP"   `YMMMa. 
    #       `Mb ,,  ,-='        Mb     dM YM.    ,   MM      MM      MM    MM    MM  8M        L.   I8 
    #        jM db Ammmmmmm     P"Ybmmd"   `Mbmmd'   `Mbmo   `Mbmo .JMML..JMML  JMML. YMMMMMb  M9mmmP' 
    #  (O)  ,M9                                                                      6'     dP         
    #   6mmm9                                                                        Ybmmmd'           
    #====================================================================================================#

    radius         =  2
    ngram          =  3
    dim            =  20
    layer_gnn      =  3
    window         =  11
    layer_cnn      =  3
    layer_output   =  3
    lr             =  1e-3
    lr_decay       =  0.5
    decay_interval =  10
    weight_decay   =  1e-6
    iteration      =  50

    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration) = \
        map(int, [dim, layer_gnn, window, layer_cnn, layer_output, decay_interval, iteration])

    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')





    #====================================================================================================#
    #                           `7MM"""YMM                                      `7MM           
    #                             MM    `7                                        MM           
    #   M******     pd""b.        MM   d    `7MMpMMMb.   ,p6"bo   ,pW"Wq.    ,M""bMM   .gP"Ya  
    #  .M          (O)  `8b       MMmmMM      MM    MM  6M'  OO  6W'   `Wb ,AP    MM  ,M'   Yb 
    #  |bMMAg.          ,89       MM   Y  ,   MM    MM  8M       8M     M8 8MI    MM  8M"""""" 
    #       `Mb ,,    ""Yb.       MM     ,M   MM    MM  YM.    , YA.   ,A9 `Mb    MM  YM.    , 
    #        jM db       88     .JMMmmmmMMM .JMML  JMML. YMbmd'   `Ybmd9'   `Wbmd"MML. `Mbmmd' 
    #  (O)  ,M9    (O)  .M'                                                                    
    #   6mmm9       bmmmd'                                                                     
    #====================================================================================================#
    word_dict = defaultdict(lambda: len(word_dict))
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

    def dlkcat_preprocess(seqs_list, smls_list, y_list, word_dict, atom_dict, bond_dict, fingerprint_dict, edge_dict):


        proteins    = list()
        compounds   = list()
        adjacencies = list()
        regression  = list()

        for i, (one_seqs, one_smls, one_y) in enumerate(zip(seqs_list, smls_list, y_list)):
            smiles   = one_smls
            sequence = one_seqs
            Kcat     = one_y
            if "." not in smiles and float(Kcat) > 0:

                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)


                fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
                compounds.append(fingerprints)

                adjacency = create_adjacency(mol)
                adjacencies.append(adjacency)

                words = split_sequence(sequence, ngram, word_dict)
                proteins.append(words)

                #regression.append(np.array([math.log2(float(Kcat))]))
                regression.append(  np.array([math.log2(float(Kcat))]) if log_value else np.array([float(Kcat)]) )

        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)





        return [torch.tensor(i).long() .to(device) for i in compounds   ] ,\
               [torch.tensor(i).float().to(device) for i in adjacencies ] ,\
               [torch.tensor(i).long() .to(device) for i in proteins    ] ,\
               [torch.tensor(i).float().to(device) for i in regression  ] ,\
               n_fingerprint, n_word



    cmpd, adjc, prot, regn, n_fingerprint, n_word = dlkcat_preprocess(X_tr_seqs+X_ts_seqs+X_va_seqs, 
                                                                      X_tr_smls+X_ts_smls+X_va_smls, 
                                                                      np.concatenate([y_tr, y_ts, y_va]),
                                                                      word_dict,
                                                                      atom_dict,
                                                                      bond_dict,
                                                                      fingerprint_dict,
                                                                      edge_dict,                      
                                                                      )


    [print(cmpd[x]) for x in range(len(cmpd))[0:20] ]
    print("adjacencies:      ", len(adjc))
    print("proteins:         ", len(prot))
    print("interactions:     ", len(regn))
    print("fingerprint_dict: ", len(fingerprint_dict))
    print("word_dict:        ", len(word_dict))



    dataset = list(zip(cmpd, adjc, prot, regn))
    print("len(dataset): ", len(dataset))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Test on DL-kcat's savings to reproduce their results
    '''
    savings = {"dataset": dataset, "n_fingerprint": n_fingerprint, "n_word": n_word}
    pickle.dump(savings, open("./X05H_dataset_TS.p", "wb"))

    
    savings       =  pickle.load(open("./X05H_dataset_DL.p", "rb"))
    dataset       =  savings["dataset"]
    n_fingerprint =  savings["n_fingerprint"]
    n_word        =  savings["n_word"]



    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    '''

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # 
    cmpd_tr, adjc_tr, prot_tr, regn_tr, _ , _  = dlkcat_preprocess(X_tr_seqs, X_tr_smls, y_tr,
                                                                   word_dict,
                                                                   atom_dict,
                                                                   bond_dict,
                                                                   fingerprint_dict,
                                                                   edge_dict,                      
                                                                   )

    cmpd_ts, adjc_ts, prot_ts, regn_ts, _ , _  = dlkcat_preprocess(X_ts_seqs, X_ts_smls, y_ts,
                                                                   word_dict,
                                                                   atom_dict,
                                                                   bond_dict,
                                                                   fingerprint_dict,
                                                                   edge_dict,                      
                                                                   )

    cmpd_va, adjc_va, prot_va, regn_va, _ , _  = dlkcat_preprocess(X_va_seqs, X_va_smls, y_va,
                                                                   word_dict,
                                                                   atom_dict,
                                                                   bond_dict,
                                                                   fingerprint_dict,
                                                                   edge_dict,                      
                                                                   )


    dataset_train = list(zip(cmpd_tr, adjc_tr, prot_tr, regn_tr))
    dataset_dev   = list(zip(cmpd_va, adjc_va, prot_va, regn_va))
    dataset_test  = list(zip(cmpd_ts, adjc_ts, prot_ts, regn_ts))






    #====================================================================================================#
    #                              `7MMM.     ,MMF'               `7MM           `7MM  
    #                                MMMb    dPMM                   MM             MM  
    #    M******         ,AM         M YM   ,M MM   ,pW"Wq.    ,M""bMM   .gP"Ya    MM  
    #   .M              AVMM         M  Mb  M' MM  6W'   `Wb ,AP    MM  ,M'   Yb   MM  
    #   |bMMAg.       ,W' MM         M  YM.P'  MM  8M     M8 8MI    MM  8M""""""   MM  
    #        `Mb ,, ,W'   MM         M  `YM'   MM  YA.   ,A9 `Mb    MM  YM.    ,   MM  
    #         jM db AmmmmmMMmm     .JML. `'  .JMML. `Ybmd9'   `Wbmd"MML. `Mbmmd' .JMML.
    #   (O)  ,M9          MM                                                           
    #    6mmm9            MM                                                           
    #====================================================================================================#

    """Set a model."""
    torch.manual_seed(1234)

    # model = KcatPrediction().to(device)
    # trainer = Trainer(model)

    model = KcatPrediction(device        = device,
                           n_fingerprint = n_fingerprint,
                           n_word        = n_word,
                           dim           = dim,
                           layer_gnn     = layer_gnn,
                           window        = window,
                           layer_cnn     = layer_cnn,
                           layer_output  = layer_output, 
                           ).to(device)
    trainer = Trainer(model, lr = lr, weight_decay = weight_decay)

    tester = Tester(model)

    """Start training."""
    print('Training...')






    #====================================================================================================#
    #                            `7MM"""Mq.                          
    #                              MM   `MM.                         
    #   M******     M******        MM   ,M9  `7MM  `7MM  `7MMpMMMb.  
    #  .M          .M              MMmmdM9     MM    MM    MM    MM  
    #  |bMMAg.     |bMMAg.         MM  YM.     MM    MM    MM    MM  
    #       `Mb ,,      `Mb        MM   `Mb.   MM    MM    MM    MM  
    #        jM db       jM      .JMML. .JMM.  `Mbod"YML..JMML  JMML.
    #  (O)  ,M9    (O)  ,M9                                          
    #   6mmm9       6mmm9                                            
    #====================================================================================================#

    for epoch in range(1, iteration+1):
        begin_time = time.time()
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, rmse_train, r2_train = trainer.train(dataset_train)

        va_MAE, va_MSE, va_RMSE, va_R2, r_value_va, va_rho, valid_preds, valid_vals = tester.test(dataset_dev)
        ts_MAE, ts_MSE, ts_RMSE, ts_R2, r_value   , ts_rho, test_preds , test_vals  = tester.test(dataset_test)

        print("_" * 101)
        print("\nepoch: {} | time_elapsed: {:5.4f} | train_loss: {:5.4f} | vali_R_VALUE: {:5.4f} | test_R_VALUE: {:5.4f} ".format( 
             str((epoch+1)+1000).replace("1","",1), 
             np.round((time.time()-begin_time), 5),
             np.round(loss_train, 5), 
             np.round(r_value_va, 5), 
             np.round(r_value, 5),
             )
             )

        print("           | va_MAE: {:4.3f} | va_MSE: {:4.3f} | va_RMSE: {:4.3f} | va_R2: {:4.3f} | va_rho: {:4.3f} ".format( 
             va_MAE, 
             va_MSE,
             va_RMSE, 
             va_R2, 
             va_rho,
             )
             )

        print("           | ts_MAE: {:4.3f} | ts_MSE: {:4.3f} | ts_RMSE: {:4.3f} | ts_R2: {:4.3f} | ts_rho: {:4.3f} ".format( 
             ts_MAE, 
             ts_MSE,
             ts_RMSE, 
             ts_R2, 
             ts_rho,
             )
             )


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Plot:
        print("Conducting evaluation")

        y_real    = test_vals
        y_pred    = test_preds
        y_real_va = valid_vals
        y_pred_va = valid_preds

        print("len(y_pred): ", len(y_pred))
        print("len(y_pred_va): ", len(y_pred_va))

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


        _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)

        reg_scatter_distn_plot(y_pred,
                            y_real,
                            fig_size        =  (10,8),
                            marker_size     =  35,
                            fit_line_color  =  "brown",
                            distn_color_1   =  "gold",
                            distn_color_2   =  "lightpink",
                            title           =  "Predictions vs. Actual Values, R = " + \
                                                str(round(r_value,3)),
                            plot_title      =  "Predictions VS. Acutual Values",
                            x_label         =  "Actual Values",
                            y_label         =  "Predictions",
                            cmap            =  None,
                            font_size       =  18,
                            result_folder   =  results_sub_folder,
                            file_name       =  "TS" + "Epoch" + str(epoch) + output_file_header,
                            ) #For checking predictions fittings.


        _, _, r_value, _ , _ = scipy.stats.linregress(y_pred_va, y_real_va)

        reg_scatter_distn_plot(y_pred_va,
                            y_real_va,
                            fig_size        =  (10,8),
                            marker_size     =  35,
                            fit_line_color  =  "brown",
                            distn_color_1   =  "gold",
                            distn_color_2   =  "lightpink",
                            title           =  "Predictions vs. Actual Values, R = " + \
                                                str(round(r_value,3)),
                            plot_title      =  "Predictions VS. Acutual Values",
                            x_label         =  "Actual Values",
                            y_label         =  "Predictions",
                            cmap            =  None,
                            font_size       =  18,
                            result_folder   =  results_sub_folder,
                            file_name       =  "VA" + "Epoch" + str(epoch) + output_file_header,
                            ) #For checking predictions fittings.                            

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        if log_value == False and screen_bool==True: 
           

            y_real = np.delete(y_real, np.where(y_pred == 0.0))
            y_pred = np.delete(y_pred, np.where(y_pred == 0.0))
            y_real = np.log10(y_real)
            y_pred = np.log10(y_pred)
            
            reg_scatter_distn_plot(y_pred,
                                y_real,
                                fig_size        =  (10,8),
                                marker_size     =  35,
                                fit_line_color  =  "brown",
                                distn_color_1   =  "gold",
                                distn_color_2   =  "lightpink",
                                title           =  "Predictions vs. Actual Values, R = " + \
                                                    str(round(r_value,3)),
                                plot_title      =  "Predictions VS. Acutual Values",
                                x_label         =  "Actual Values",
                                y_label         =  "Predictions",
                                cmap            =  None,
                                font_size       =  18,
                                result_folder   =  results_sub_folder,
                                file_name       =  output_file_header + "_logplot"
                                ) #For checking predictions fittings.









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










