#!/usr/bin/env python
# coding: utf-8
###################################################################################################################
###################################################################################################################
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
        print('workbookDir: ' + workbookDir)
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
        print('workbookDir: ' + workbookDir)
#--------------------------------------------------#
import sys
import time
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd
#--------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA 
#--------------------------------------------------#
from HG_rdkit import *






#====================================================================================================#
def get_full_MorganFP(smiles_str, radius = 4):   
    # SMILES -> Morgan FP list.
    return smiles_list_to_all_Morgan_list(smiles_list = [smiles_str,], radius = radius, duplicates = True)
#====================================================================================================#
def generate_all_MorganFPs(list_smiles, radius = 4):
# return a list of MorganFPs of all depth for a list of compounds (duplicates removed !!!)
    all_MorganFPs_list = \
        smiles_list_to_all_Morgan_list(smiles_list = list_smiles, radius = radius)
    print("Total number of Morgan FPs: ", len(all_MorganFPs_list))
    return all_MorganFPs_list # No duplicates!!
#====================================================================================================#
def generate_all_smiles_MorganFPs_dict(list_smiles, radius = 4): # MorganFPs
    all_smiles_MorganFPs_dict=dict([])
    for smiles_a in list_smiles:
        all_smiles_MorganFPs_dict[smiles_a] = get_full_MorganFP(smiles_a, radius = radius)
    return all_smiles_MorganFPs_dict
#====================================================================================================#
def generate_all_smiles_MorganFPs_list_dict(list_smiles, radius = 4):
    all_MorganFPs = set([])
    all_smiles_MorganFPs_dict = dict([])
    count_x = 0
    for smiles_a in list_smiles:
        print(count_x," out of ", len(list_smiles) )
        count_x+=1
        discriptors = get_full_MorganFP(smiles_a, radius = radius)
        #print(smiles_a)
        all_smiles_MorganFPs_dict[smiles_a] = discriptors
        all_MorganFPs = all_MorganFPs.union(set(discriptors))
    print("Total number of Morgan FPs: ", len(all_MorganFPs))
    return list(all_MorganFPs), all_smiles_MorganFPs_dict

###################################################################################################################
###################################################################################################################
# Encode Compounds.
#====================================================================================================#
def list_smiles_to_MorganFP_through_dict(smiles_list, all_smiles_MorganFPs_dict):
    MorganFP_list=[]
    for one_smiles in smiles_list:
        MorganFP_list=MorganFP_list + all_smiles_MorganFPs_dict[one_smiles]
    return MorganFP_list
#====================================================================================================#
def smiles_to_MorganFP_vec( smiles_x, all_MorganFPs, all_smiles_MorganFPs_dict):
    dimension=len(all_MorganFPs)
    Xi = [0]*dimension
    Xi_MorganFP_list = list_smiles_to_MorganFP_through_dict( [smiles_x, ] ,all_smiles_MorganFPs_dict)
    for one_MorganFP in Xi_MorganFP_list:
        Xi[all_MorganFPs.index(one_MorganFP)]=Xi_MorganFP_list.count(one_MorganFP)
    return np.array(Xi)
#====================================================================================================#
def Get_MorganFPs_encoding(X_cmpd_representations, all_MorganFPs, all_smiles_MorganFPs_dict):
    X_cmpd_encodings=[]
    for one_smiles in X_cmpd_representations:
        one_cmpd_encoding = smiles_to_MorganFP_vec(one_smiles, all_MorganFPs, all_smiles_MorganFPs_dict) # compound_encoding
        X_cmpd_encodings.append(one_cmpd_encoding)
    return X_cmpd_encodings
#====================================================================================================#
def Get_MorganFPs_encoding_dict(X_cmpd_smiles_list, all_MorganFPs, all_smiles_MorganFPs_dict):
    X_cmpd_encodings_dict=dict([])
    max_encoding_count = 0
    for one_smiles in X_cmpd_smiles_list:
        one_cmpd_encoding = smiles_to_MorganFP_vec(one_smiles, all_MorganFPs, all_smiles_MorganFPs_dict) # compound_encoding
        X_cmpd_encodings_dict[one_smiles] = one_cmpd_encoding
        max_encoding_count = max(one_cmpd_encoding) if max(one_cmpd_encoding)>max_encoding_count else max_encoding_count
    print("max_encoding_count (used in Cross-Attention): ", max_encoding_count)
    return X_cmpd_encodings_dict, max_encoding_count

#====================================================================================================#
def Get_Morgan_encoding(X_cmpd_representations, cmpd_SMILES_Morgan1024_dict):
    X_cmpd_encodings=[]
    for one_smiles in X_cmpd_representations:
        one_cmpd_encoding = cmpd_SMILES_Morgan1024_dict[one_smiles] # compound_encoding
        X_cmpd_encodings.append(one_cmpd_encoding)
    return X_cmpd_encodings
#====================================================================================================#
def Get_Morgan_encoding(X_cmpd_representations, cmpd_SMILES_Morgan1024_dict):
    return cmpd_SMILES_Morgan1024_dict





###################################################################################################################
###################################################################################################################
def X02B_Get_Cmpd_Encodings(data_folder, 
                            properties_file,
                            cmpd_encodings,
                            data_folder_2,
                            cmpd_JTVAE_file,
                            cmpd_Morgan1024_file,
                            cmpd_Morgan2048_file,

                            i_o_put_file_1,
                            i_o_put_file_2,
                            output_folder,
                            output_file_1
                            ):



    
    #====================================================================================================#
    # Get cmpd_properties_list.
    with open( data_folder / properties_file, 'rb') as cmpd_properties:
        cmpd_properties_list = pickle.load(cmpd_properties) # [[one_compound, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    #====================================================================================================#
    # Get all SMILES.
    all_smiles_list=[]
    for one_list_prpt in cmpd_properties_list:
        all_smiles_list.append(one_list_prpt[-1])
    #print(all_smiles_list)
    #====================================================================================================#
    # Get all compounds MorganFP encoded.
    
    all_MorganFPs, all_smiles_MorganFPs_dict = generate_all_smiles_MorganFPs_list_dict(all_smiles_list, radius = int(cmpd_encodings[-1]))
    pickle.dump(all_MorganFPs, open(data_folder / i_o_put_file_1,"wb") )
    pickle.dump(all_smiles_MorganFPs_dict, open(data_folder / i_o_put_file_2,"wb"))
    
    #====================================================================================================#
    # Encode Compounds.
    # cmpd_encodings_list = []

    if cmpd_encodings in ["MgFP1", "MgFP2", "MgFP3", "MgFP4", "MgFP5", "MgFP6"]:
        with open( data_folder / i_o_put_file_1, 'rb') as all_MorganFPs:
            all_MorganFPs = pickle.load(all_MorganFPs)
        with open( data_folder / i_o_put_file_2, 'rb') as all_smiles_MorganFPs_dict:
            all_smiles_MorganFPs_dict = pickle.load(all_smiles_MorganFPs_dict)
        X_cmpd_encodings_dict, _ = Get_MorganFPs_encoding_dict(all_smiles_list, all_MorganFPs, all_smiles_MorganFPs_dict)

    #====================================================================================================#
    pickle.dump( X_cmpd_encodings_dict, open( output_folder / output_file_1, "wb" ) )
    #====================================================================================================#
    









    print(Step_code + " Done!")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'      db      `7MMF'`7MN.   `7MF'                 M             M             M                                                    #
#     MMMb    dPMM       ;MM:       MM    MMN.    M                   M             M             M                                                    #
#     M YM   ,M MM      ,V^MM.      MM    M YMb   M                   M             M             M                                                    #
#     M  Mb  M' MM     ,M  `MM      MM    M  `MN. M               `7M'M`MF'     `7M'M`MF'     `7M'M`MF'                                                #
#     M  YM.P'  MM     AbmmmqMA     MM    M   `MM.M                 VAMAV         VAMAV         VAMAV                                                  #
#     M  `YM'   MM    A'     VML    MM    M     YMM                  VVV           VVV           VVV                                                   #
#   .JML. `'  .JMML..AMA.   .AMMA..JMML..JML.    YM                   V             V             V                                                    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":
    ###################################################################################################################
    ###################################################################################################################
    # Args
    #--------------------------------------------------#
    # Inputs
    Step_code = "X02B_"
    dataset_nme_list = ["phosphatase",        # 0
                        "kinase",             # 1
                        "esterase",           # 2
                        "halogenase",         # 3
                        "aminotransferase",   # 4
                        "kcat_c",             # 5
                        "kcat",               # 6
                        "kcat_mt",            # 7
                        "kcat_wt",            # 8
                        "Ki_all_org",          # 9
                        "Ki_small",            # 10
                        "Ki_select",          # 11
                        "KM_BRENDA",          # 12
                        ]            
    dataset_nme = dataset_nme_list[6]
    data_folder = Path("X_DataProcessing/")
    properties_file = "X00_" + dataset_nme + "_compounds_properties_list.p"
    #--------------------------------------------------#
    # Select compound encodings
    cmpd_encodings_list = ["MgFP1", "MgFP2", "MgFP3", "MgFP4", "MgFP5", "MgFP6",]
    cmpd_encodings = cmpd_encodings_list[5]
    #---------- MgFP
    MgFP_type = cmpd_encodings[-1] if cmpd_encodings in ["MgFP1", "MgFP2", "MgFP3", "MgFP4", "MgFP5", "MgFP6",] else 5 # 1,2,3,4,5
    #---------- JTVAE
    data_folder_2 = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
    cmpd_JTVAE_file = "X00_" + dataset_nme + "_JTVAE_features.p"
    #---------- Morgan
    data_folder = Path("X_DataProcessing/")
    cmpd_Morgan1024_file = "X00_" + dataset_nme + "_Morgan1024_features.p"
    cmpd_Morgan2048_file = "X00_" + dataset_nme + "_Morgan2048_features.p"
    #--------------------------------------------------#
    # In/Outputs
    i_o_put_file_1 = Path( Path("temp") / Path(Step_code + dataset_nme + "_all_MgFPs" + str(MgFP_type) + ".p"))
    i_o_put_file_2 = Path( Path("temp") / Path(Step_code + dataset_nme + "_all_cmpds_MgFPs" + str(MgFP_type) + "_dict.p"))
    #--------------------------------------------------#
    # Outputs
    output_folder = Path("X_DataProcessing/")
    output_file_1 = Step_code + dataset_nme + "_" + cmpd_encodings  + "_" + "encodings_dict.p"

    ###################################################################################################################
    ###################################################################################################################

    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_folder",          type=Path, default = data_folder, help=\
        "Path to the directory containing your datasets.")
    parser.add_argument("--properties_file",      type=str,  default = properties_file, help=\
        "properties_file.")

    parser.add_argument("--cmpd_encodings",       type=str,  default = cmpd_encodings, help=\
        "cmpd_encodings.")


    parser.add_argument("--data_folder_2",        type=str,  default = data_folder_2, help=\
        "data_folder_2.")
    parser.add_argument("--cmpd_JTVAE_file",      type=str,  default = cmpd_JTVAE_file, help=\
        "cmpd_JTVAE_file.")
    parser.add_argument("--cmpd_Morgan1024_file", type=str,  default = cmpd_Morgan1024_file, help=\
        "cmpd_Morgan1024_file.")
    parser.add_argument("--cmpd_Morgan2048_file", type=str,  default = cmpd_Morgan2048_file, help=\
        "cmpd_Morgan2048_file.")
    parser.add_argument("--i_o_put_file_1",       type=str,  default = i_o_put_file_1, help=\
        "i_o_put_file_1.")
    parser.add_argument("--i_o_put_file_2",       type=str,  default = i_o_put_file_2, help=\
        "i_o_put_file_2.")

    parser.add_argument("--output_folder",        type=str,  default = output_folder, help=\
        "output_folder.")
    parser.add_argument("--output_file_1",        type=str,  default = output_file_1, help=\
        "output_file_1.")

    args = parser.parse_args()

    #====================================================================================================#
    # Main
    X02B_Get_Cmpd_Encodings(**vars(args))

    print("*" * 50)
    print(Step_code + " Done!") 




