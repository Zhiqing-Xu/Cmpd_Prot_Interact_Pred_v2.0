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
# Imports
import random
#--------------------------------------------------#
import numpy as np
import pandas as pd
import seaborn as sns
#--------------------------------------------------#
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib import cm
#--------------------------------------------------#
import sklearn
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from AP_funcs import cart_prod, cart_dual_prod_re




###################################################################################################################
#          `7MMF'        .g8""8q.         db      `7MM"""Yb.             db      `7MMF'      `7MMF'               #
#            MM        .dP'    `YM.      ;MM:       MM    `Yb.          ;MM:       MM          MM                 #
#            MM        dM'      `MM     ,V^MM.      MM     `Mb         ,V^MM.      MM          MM                 #
#            MM        MM        MM    ,M  `MM      MM      MM        ,M  `MM      MM          MM                 #
#            MM      , MM.      ,MP    AbmmmqMA     MM     ,MP        AbmmmqMA     MM      ,   MM      ,          #
#            MM     ,M `Mb.    ,dP'   A'     VML    MM    ,dP'       A'     VML    MM     ,M   MM     ,M          #
#          .JMMmmmmMMM   `"bmmd"'   .AMA.   .AMMA..JMMmmmdP'       .AMA.   .AMMA..JMMmmmmMMM .JMMmmmmMMM          #
###################################################################################################################
def Get_represented_X_y_data(X_seqs_all_hiddens_list, X_all_seqs_list, cmpd_properties_list, X_cmpd_encodings_dict, screen_bool, classification_threshold_type):
    # new: X_seqs_all_hiddens_list, old: seqs_all_hiddens_list
    # cmpd_properties_list: [[one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    X_seqs_all_hiddens = []
    X_all_seqs = []
    X_cmpd_encodings=[]
    X_cmpd_smiles = []
    y_data = []
    seqs_cmpd_idx_book=[]
    print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
    for i in range(len(cmpd_properties_list)):
        for j in range(len(X_seqs_all_hiddens_list)):
            X_smiles_rep = cmpd_properties_list[i][-1] # list of SMILES
            X_one_all_hiddens = X_seqs_all_hiddens_list[j]
            X_one_seqs = X_all_seqs_list[j]
            if not (screen_bool==True and cmpd_properties_list[i][classification_threshold_type][j]==False):
                # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
                # cmpd_properties_list[i] ----> [one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
                # cmpd_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
                # cmpd_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
                # cmpd_properties_list[i][1][j] ----> y_prpty_reg[j]
                X_seqs_all_hiddens.append(X_one_all_hiddens)
                X_all_seqs.append(X_one_seqs)
                X_cmpd_encodings.append(X_cmpd_encodings_dict[X_smiles_rep])
                X_cmpd_smiles.append(X_smiles_rep)
                y_data.append(cmpd_properties_list[i][1][j])
                seqs_cmpd_idx_book.append([j, i]) # [seqs_idx, cmpd_idx]
    return X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, seqs_cmpd_idx_book

#====================================================================================================#
def Get_represented_X_y_data_clf(X_seqs_all_hiddens_list, X_all_seqs_list, cmpd_properties_list, X_cmpd_encodings_dict, screen_bool, classification_threshold_type):
    # new: X_seqs_all_hiddens_list, old: seqs_all_hiddens_list
    # cmpd_properties_list: [[one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    X_seqs_all_hiddens = []
    X_all_seqs = []
    X_cmpd_encodings=[]
    X_cmpd_smiles = []
    y_data = []
    seqs_cmpd_idx_book=[]
    print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
    for i in range(len(cmpd_properties_list)):
        for j in range(len(X_seqs_all_hiddens_list)):
            X_smiles_rep = cmpd_properties_list[i][-1] # list of SMILES
            X_one_all_hiddens = X_seqs_all_hiddens_list[j]
            X_one_seqs = X_all_seqs_list[j]
            # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
            # cmpd_properties_list[i] ----> [one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
            # cmpd_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
            # cmpd_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
            X_seqs_all_hiddens.append(X_one_all_hiddens)
            X_all_seqs.append(X_one_seqs)
            X_cmpd_encodings.append(X_cmpd_encodings_dict[X_smiles_rep])
            X_cmpd_smiles.append(X_smiles_rep)
            y_data.append(cmpd_properties_list[i][classification_threshold_type][j])
            seqs_cmpd_idx_book.append([j, i]) # [seqs_idx, cmpd_idx]
    return X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, seqs_cmpd_idx_book







###################################################################################################################
#           `7MM"""Mq. `7MM"""Mq.  `7MM"""YMM       .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM            #
#             MM   `MM.  MM   `MM.   MM    `7      ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7            #
#             MM   ,M9   MM   ,M9    MM   d        `MMb.       MM   ,M9   MM          MM       MM                 #
#             MMmmdM9    MMmmdM9     MMmmMM          `YMMNq.   MMmmdM9    MM          MM       MM                 #
#             MM         MM  YM.     MM   Y  ,     .     `MM   MM         MM      ,   MM       MM                 #
#             MM         MM   `Mb.   MM     ,M     Mb     dM   MM         MM     ,M   MM       MM                 #
#           .JMML.     .JMML. .JMM..JMMmmmmMMM     P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.               #
###################################################################################################################
def split_idx(X_num, train_split = 0.8, test_split = 0.1, random_state = 42): 
    # This function is used for split_type = 1, 2, 3, 4, 5. (NOT for random split.)
    # X_seqs_idx = y_seqs_idx = list(range(len(X_seqs_all_hiddens_list)))
    # X_cmpd_idx = y_cmpd_idx = list(range(len(cmpd_properties_list)))
    #--------------------------------------------------#
    print("train_split: ", train_split, ", test_split: ", test_split)
    X_idx = y_idx = list(range(X_num))
    X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_idx, y_idx, test_size = (1-train_split), random_state = random_state)
    X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
    return X_tr_idx, X_ts_idx, X_va_idx


#====================================================================================================#
def split_seqs_idx_custom(X_num, customized_idx_list, customized_idx_dict, train_split = 0.7, test_split = 0.2, random_state = 42, split_type = 4): 
    # FOR SEQUENCES ONLY !!!
    # customized_idx_list: idx of those SEQUENCES selected by CD-HIT.

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 4: 
        # non-customized idx -> the set that contains its representatives.
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        print("len(customized_idx_list): ", len(customized_idx_list))
        print("len(customized_idx_dict): ", len(customized_idx_dict))
        customized_idx_dict_keys = list(customized_idx_dict.keys())
        customized_idx_dict_values = [id2 for id in customized_idx_dict for id2 in customized_idx_dict[id]]
        #--------------------------------------------------#
        # Split the representative sequences.
        X_tr_idx = y_tr_idx = customized_idx_list
        X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_tr_idx, y_tr_idx, test_size = (1-train_split), random_state = random_state)
        X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
        #--------------------------------------------------#
        # Split the .

        X_idx = y_idx = list(range(X_num))
        X_tr_idx_extra = [non_rep_idx for rep_idx in X_tr_idx for non_rep_idx in customized_idx_dict[rep_idx]]
        X_ts_idx_extra = [non_rep_idx for rep_idx in X_ts_idx for non_rep_idx in customized_idx_dict[rep_idx]]
        X_va_idx_extra = [non_rep_idx for rep_idx in X_va_idx for non_rep_idx in customized_idx_dict[rep_idx]]

        X_tr_idx_extra_extra = [idx for idx in X_idx if idx not in customized_idx_dict_values and idx not in customized_idx_dict_keys]
        #--------------------------------------------------#
        X_tr_idx = X_tr_idx + X_tr_idx_extra + X_tr_idx_extra_extra
        X_ts_idx = X_ts_idx + X_ts_idx_extra
        X_va_idx = X_va_idx + X_va_idx_extra

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type == 5:
        # non-customized idx -> DUMPED
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        print("len(customized_idx_list): ", len(customized_idx_list))

        X_tr_idx_extra = []
        #--------------------------------------------------#
        X_tr_idx = y_tr_idx = customized_idx_list
        X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_tr_idx, y_tr_idx, test_size = (1-train_split), random_state = random_state)
        X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state = random_state)
        #--------------------------------------------------#
        X_tr_idx = X_tr_idx + X_tr_idx_extra

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    return X_tr_idx, X_ts_idx, X_va_idx







###################################################################################################################
#           .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM           db      `7MMF'      `7MMF'               #
#          ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7          ;MM:       MM          MM                 #
#          `MMb.       MM   ,M9   MM          MM       MM              ,V^MM.      MM          MM                 #
#            `YMMNq.   MMmmdM9    MM          MM       MM             ,M  `MM      MM          MM                 #
#          .     `MM   MM         MM      ,   MM       MM             AbmmmqMA     MM      ,   MM      ,          #
#          Mb     dM   MM         MM     ,M   MM       MM            A'     VML    MM     ,M   MM     ,M          #
#          P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.        .AMA.   .AMMA..JMMmmmmMMM .JMMmmmmMMM          #
###################################################################################################################
def split_seqs_cmpd_idx_book(tr_idx_seqs, 
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
                             random_state = 42, 
                             train_split = 0.7, 
                             test_split = 0.2,
                             ):
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # split_type = 0, train/test split completely randomly selected
    # split_type = 1, train/test split contains different seq-cmpd pairs
    # split_type = 2, train/test split contains different seqs
    # split_type = 3, train/test split contains different cmpd
    # split_type = 4, train/test split contains different CD-hit seqs, train contains non-CD-hit seqs.
    # split_type = 5, train/test split contains different CD-hit seqs
    # split_type = 6, train/test split completely randomly selected, with non-CD-hit seqs data all contained in train.
    # split_type = 7, train/test split completely randomly selected, with non-CD-hit seqs data being left out.
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    tr_idx, ts_idx, va_idx = [], [], []
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type==0: 
        # In this case, split_idx outputs are not used at all. split ratio is defined HERE in the func, not from Z05_utils
        dataset_size = len(seqs_cmpd_idx_book)
        X_data_idx = np.array(list(range(dataset_size)))
        tr_idx, ts_idx, y_train, y_test = train_test_split(X_data_idx, y_data, test_size = (1-train_split), random_state = random_state)             # tr : ts & va 
        va_idx, ts_idx, y_valid, y_test = train_test_split(ts_idx, y_test, test_size = (test_split/(1.0-train_split)), random_state = random_state)  # va : ts      
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type in [1,2,3,4,5]:
        all_idx_cmpd = list(range(X_cmpd_num))
        all_idx_seqs = list(range(X_seqs_num))
        #--------------------------------------------------#
        if split_type==1:
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([tr_idx_seqs, tr_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([ts_idx_seqs, ts_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([va_idx_seqs, va_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        if split_type==2: 
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([tr_idx_seqs, all_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([ts_idx_seqs, all_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([va_idx_seqs, all_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        if split_type==3:
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([all_idx_seqs, tr_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([all_idx_seqs, ts_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([all_idx_seqs, va_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        if split_type==4 or split_type==5: # Exactly same as TYPE 2.
            tr_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([tr_idx_seqs, all_idx_cmpd])]
            ts_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([ts_idx_seqs, all_idx_cmpd])]
            va_idx_pairs_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in cart_prod([va_idx_seqs, all_idx_cmpd])]
            seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        #--------------------------------------------------#
        tr_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, tr_idx_pairs_encrypt))[0]))
        ts_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, ts_idx_pairs_encrypt))[0]))
        va_idx = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, va_idx_pairs_encrypt))[0]))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type==6: # Meaningless
        # non-customized idx -> training set
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        all_idx_cmpd = list(range(X_cmpd_num))

        #--------------------------------------------------#
        n_CD_hit_seqs_idx_list      = [idx for idx in list(range(X_seqs_num)) if (idx not in customized_idx_list)]
        dataset_n_CD_hit            = cart_prod([n_CD_hit_seqs_idx_list, all_idx_cmpd])
        dataset_n_CD_hit_encrypt    = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in dataset_n_CD_hit]
        seqs_cmpd_idx_book_encrypt  = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        dataset_n_CD_hit_idx        = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, dataset_n_CD_hit_encrypt))[0]))
        #--------------------------------------------------#
        y_CD_hit_seqs_idx_list      = customized_idx_list
        dataset_y_CD_hit            = cart_prod([y_CD_hit_seqs_idx_list, all_idx_cmpd])
        dataset_y_CD_hit_encrypt    = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in dataset_y_CD_hit]
        seqs_cmpd_idx_book_encrypt = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        dataset_y_CD_hit_idx        = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, dataset_y_CD_hit_encrypt))[0]))
        #--------------------------------------------------#
        tr_idx, ts_idx, y_tr_idx, y_ts_idx = train_test_split(dataset_y_CD_hit_idx, dataset_y_CD_hit_idx, test_size = (1-train_split), random_state = random_state) # tr : ts & va 
        va_idx, ts_idx, y_va_idx, y_ts_idx = train_test_split(ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)), random_state = random_state)        
        #--------------------------------------------------#
        tr_idx = list(tr_idx) + list(dataset_n_CD_hit_idx)
        ts_idx = list(ts_idx)
        va_idx = list(va_idx)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    if split_type==7: # Try to predict seqs-cmpd pairs with no help from sequences similarities.
        # non-customized idx -> training set
        # customized idx -> 70% training set, 20% validation set, 10% test set.
        all_idx_cmpd = list(range(X_cmpd_num))
        #--------------------------------------------------#
        y_CD_hit_seqs_idx_list      = customized_idx_list
        dataset_y_CD_hit            = cart_prod([y_CD_hit_seqs_idx_list, all_idx_cmpd])
        dataset_y_CD_hit_encrypt    = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in dataset_y_CD_hit]
        seqs_cmpd_idx_book_encrypt  = [one_pair[0] + one_pair[1] * X_seqs_num for one_pair in seqs_cmpd_idx_book]
        dataset_y_CD_hit_idx        = list(set(np.where(np.isin(seqs_cmpd_idx_book_encrypt, dataset_y_CD_hit_encrypt))[0]))
        #--------------------------------------------------#
        tr_idx, ts_idx, y_tr_idx, y_ts_idx = train_test_split(dataset_y_CD_hit_idx, dataset_y_CD_hit_idx, test_size = (1-train_split), random_state = random_state) # tr : ts & va 
        va_idx, ts_idx, y_va_idx, y_ts_idx = train_test_split(ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)), random_state = random_state) 
        #--------------------------------------------------#
        tr_idx = list(tr_idx)
        ts_idx = list(ts_idx)
        va_idx = list(va_idx)


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    tr_seqs_idx_verify = list(set([seqs_cmpd_idx_book[idx][0] for idx in tr_idx]))
    ts_seqs_idx_verify = list(set([seqs_cmpd_idx_book[idx][0] for idx in ts_idx]))
    va_seqs_idx_verify = list(set([seqs_cmpd_idx_book[idx][0] for idx in va_idx]))
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    #print(tr_seqs_idx_verify)
    print("-"*50)
    print("Verify the number of sequences in each split: ")
    print("len(tr_seqs_idx_verify): ", len(tr_seqs_idx_verify))
    print("len(ts_seqs_idx_verify): ", len(ts_seqs_idx_verify))
    print("len(va_seqs_idx_verify): ", len(va_seqs_idx_verify))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    return tr_idx, ts_idx, va_idx





###################################################################################################################
#         `7MMF'`7MM"""Yb.`YMM'   `MP'       mm               `7MM"""Yb.      db  MMP""MM""YMM  db                #
#           MM    MM    `Yb.VMb.  ,P         MM                 MM    `Yb.   ;MM: P'   MM   `7 ;MM:               #
#           MM    MM     `Mb `MM.M'        mmMMmm ,pW"Wq.       MM     `Mb  ,V^MM.     MM     ,V^MM.              #
#           MM    MM      MM   MMb           MM  6W'   `Wb      MM      MM ,M  `MM     MM    ,M  `MM              #
#           MM    MM     ,MP ,M'`Mb.         MM  8M     M8      MM     ,MP AbmmmqMA    MM    AbmmmqMA             #
#           MM    MM    ,dP',P   `MM.        MM  YA.   ,A9      MM    ,dP'A'     VML   MM   A'     VML            #
#         .JMML..JMMmmmdP'.MM:.  .:MMa.      `Mbmo`Ybmd9'     .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.          #
###################################################################################################################
def Get_X_y_data_selected(X_idx, X_seqs_all_hiddens, X_all_seqs, X_cmpd_encodings, X_cmpd_smiles, y_data, log_value):
    X_seqs_emb_selected = []
    X_seqs_selected=[]
    X_cmpd_selected=[]
    X_smls_selected=[]
    y_data_selected=[]
    for idx in X_idx:
        #print(y_data[idx])
        if y_data[idx]==None:
            continue
        #print(y_data[idx])
        X_seqs_emb_selected.append(X_seqs_all_hiddens[idx])
        X_seqs_selected.append(X_all_seqs[idx])
        X_cmpd_selected.append(X_cmpd_encodings[idx])
        X_smls_selected.append(X_cmpd_smiles[idx])
        y_data_selected.append(y_data[idx])

    y_data_selected = np.array(y_data_selected)
    if log_value==True:
        y_data_selected=np.log10(y_data_selected)
    return X_seqs_emb_selected, X_seqs_selected, X_cmpd_selected, X_smls_selected, y_data_selected







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












