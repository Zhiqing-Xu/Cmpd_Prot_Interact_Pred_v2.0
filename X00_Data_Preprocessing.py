#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
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
#--------------------------------------------------#
import pickle
import argparse
import numpy as np
import pandas as pd

#--------------------------------------------------#
from AP_convert import *
from AP_convert import Get_Unique_SMILES

GetUnqSmi = Get_Unique_SMILES(isomericSmiles = False, SMARTS_bool = False)

GetUnqSmi = Get_Unique_SMILES(isomericSmiles = True, canonical = False, SMARTS_bool = False)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#              `7MM"""Yb.      db  MMP""MM""YMM  db     `7MM"""YMM `7MM"""Mq.       db     `7MMM.     ,MMF'`7MM"""YMM                           M      #
#   __,          MM    `Yb.   ;MM: P'   MM   `7 ;MM:      MM    `7   MM   `MM.     ;MM:      MMMb    dPMM    MM    `7                           M      #
#  `7MM          MM     `Mb  ,V^MM.     MM     ,V^MM.     MM   d     MM   ,M9     ,V^MM.     M YM   ,M MM    MM   d                             M      #
#    MM          MM      MM ,M  `MM     MM    ,M  `MM     MM""MM     MMmmdM9     ,M  `MM     M  Mb  M' MM    MMmmMM                         `7M'M`MF'  #
#    MM          MM     ,MP AbmmmqMA    MM    AbmmmqMA    MM   Y     MM  YM.     AbmmmqMA    M  YM.P'  MM    MM   Y  ,                        VAM,V    #
#    MM  ,,      MM    ,dP'A'     VML   MM   A'     VML   MM         MM   `Mb.  A'     VML   M  `YM'   MM    MM     ,M                         VVV     #
#  .JMML.db    .JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA.JMML.     .JMML. .JMM.AMA.   .AMMA.JML. `'  .JMML..JMMmmmmMMM                          V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#  
# Import data and obtain a DataFrame.
def Get_cmpd_df(data_folder, data_file, data_file_binary, binary_class_bool, y_prpty_cls_threshold, target_nme):
    #====================================================================================================#
    cmpd_df = pd.read_csv(data_folder / data_file, index_col=0, header=0)
    # Remove sequences that are too long (remove those longer than max_seqs_len)
    cmpd_df["seqs_length"] = cmpd_df.SEQ.str.len()
    cmpd_df = cmpd_df[cmpd_df.seqs_length <= max_seqs_len]
    cmpd_df.reset_index(drop = True, inplace = True)
    # Remove compounds that are NOT identified as SMILES.
    cmpd_df["smiles_validity"] = cmpd_df["CMPD_SMILES"].apply(GetUnqSmi.ValidSMI)
    cmpd_df = cmpd_df[cmpd_df.smiles_validity == True]
    cmpd_df.reset_index(drop = True, inplace = True)


    #====================================================================================================#
    if binary_class_bool and ((data_folder / data_file_binary).exists()):
        cmpd_df_bi = pd.read_csv(data_folder / data_file_binary, index_col=0, header=0)
    else:
        print("binary classification file does not exist.")
        cmpd_df_bi = pd.read_csv(data_folder / data_file, index_col=0, header=0)
        #print(list(cmpd_df_bi[target_nme]))
        cmpd_df_bi[target_nme] = [1 if one_cvsn>y_prpty_cls_threshold else 0 
                                    for one_cvsn in list(cmpd_df_bi[target_nme])]
    #--------------------------------------------------#
    # Remove sequences that are too long (remove those longer than max_seqs_len)
    cmpd_df_bi["seqs_length"] = cmpd_df_bi.SEQ.str.len()
    cmpd_df_bi = cmpd_df_bi[cmpd_df_bi.seqs_length <= max_seqs_len]
    cmpd_df_bi.reset_index(drop = True, inplace = True)
    # Remove compounds that are NOT identified as SMILES.
    cmpd_df_bi["smiles_validity"] = cmpd_df_bi["CMPD_SMILES"].apply(GetUnqSmi.ValidSMI)
    cmpd_df_bi = cmpd_df_bi[cmpd_df_bi.smiles_validity == True]
    cmpd_df_bi.reset_index(drop = True, inplace = True)
    
    return cmpd_df, cmpd_df_bi

###################################################################################################################
###################################################################################################################
# Print the DataFrame obtained.
def beautiful_print(df):
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows', 20, 
                           'display.min_rows', 20, 
                           'display.max_columns', 6, 
                           #"display.max_colwidth", None,
                           "display.width", None,
                           "expand_frame_repr", True,
                           "max_seq_items", None,):  # more options can be specified
        # Once the display.max_rows is exceeded, 
        # the display.min_rows options determines 
        # how many rows are shown in the truncated repr.
        print(df)
    return 









#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                  `7MMF'     A     `7MF'`7MM"""Mq.  `7MMF'MMP""MM""YMM `7MM"""YMM      `7MM"""YMM    db       .M"""bgd MMP""MM""YMM   db      
#                    `MA     ,MA     ,V    MM   `MM.   MM  P'   MM   `7   MM    `7        MM    `7   ;MM:     ,MI    "Y P'   MM   `7  ;MM:     
#   pd*"*b.           VM:   ,VVM:   ,V     MM   ,M9    MM       MM        MM   d          MM   d    ,V^MM.    `MMb.          MM      ,V^MM.    
#  (O)   j8            MM.  M' MM.  M'     MMmmdM9     MM       MM        MMmmMM          MM""MM   ,M  `MM      `YMMNq.      MM     ,M  `MM    
#      ,;j9            `MM A'  `MM A'      MM  YM.     MM       MM        MM   Y  ,       MM   Y   AbmmmqMA   .     `MM      MM     AbmmmqMA   
#   ,-='    ,,          :MM;    :MM;       MM   `Mb.   MM       MM        MM     ,M       MM      A'     VML  Mb     dM      MM    A'     VML  
#  Ammmmmmm db           VF      VF      .JMML. .JMM..JMML.   .JMML.    .JMMmmmmMMM     .JMML..  AMA.   .AMMA.P"Ybmmd"     .JMML..AMA.   .AMMA.
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Output #1: Write a fasta file including all sequences
def output_fasta(cmpd_df, output_folder, output_file_0_0):
    # Also get a seqs_list including all sequences
    cmpd_df_row_num = cmpd_df.shape[0]
    with open(output_folder / output_file_0_0 , 'w') as f:
        count_x=0
        seqs_list=[]
        max_len = 0
        print("cmpd_df_row_num: ", cmpd_df_row_num)
        for i in range(cmpd_df_row_num):
            one_seq = (cmpd_df.loc[i,"SEQ"]).replace("-", "")
            max_len = len(one_seq) if len(one_seq)>max_len else max_len
            if one_seq not in seqs_list and len(one_seq)<=max_seqs_len:
                seqs_list.append(one_seq)
                count_x+=1
                if len(one_seq) <= 1024-2:
                    f.write(">seq"+str(count_x)+"\n")
                    f.write(one_seq.upper()+"\n")
                else:
                    f.write(">seq"+str(count_x)+"\n")
                    f.write(one_seq.upper()[0 : 1024-2]+"\n")
    print("number of seqs: ", len(seqs_list))
    print("number of seqs duplicates removed: ", len(set(seqs_list)))
    return seqs_list









#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   pd""b.      `7MMF'     A     `7MF'`7MM"""Mq.  `7MMF'MMP""MM""YMM `7MM"""YMM       .M"""bgd `7MMM.     ,MMF'`7MMF'`7MMF'      `7MM"""YMM   .M"""bgd #
#  (O)  `8b       `MA     ,MA     ,V    MM   `MM.   MM  P'   MM   `7   MM    `7      ,MI    "Y   MMMb    dPMM    MM    MM          MM    `7  ,MI    "Y #
#       ,89        VM:   ,VVM:   ,V     MM   ,M9    MM       MM        MM   d        `MMb.       M YM   ,M MM    MM    MM          MM   d    `MMb.     #
#     ""Yb.         MM.  M' MM.  M'     MMmmdM9     MM       MM        MMmmMM          `YMMNq.   M  Mb  M' MM    MM    MM          MMmmMM      `YMMNq. #
#        88         `MM A'  `MM A'      MM  YM.     MM       MM        MM   Y  ,     .     `MM   M  YM.P'  MM    MM    MM      ,   MM   Y  , .     `MM #
#  (O)  .M' ,,       :MM;    :MM;       MM   `Mb.   MM       MM        MM     ,M     Mb     dM   M  `YM'   MM    MM    MM     ,M   MM     ,M Mb     dM #
#   bmmmd'  db        VF      VF      .JMML. .JMM..JMML.   .JMML.    .JMMmmmmMMM     P"Ybmmd"  .JML. `'  .JMML..JMML..JMMmmmmMMM .JMMmmmmMMM P"Ybmmd"  #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Output #3: Write a text file including all smiles
def output_smiles(cmpd_smiles_list, output_folder, output_file_0_1):
    # Also get a cmpd_smiles_list including all smiles
    with open(output_folder / output_file_0_1 , 'w') as f:
        count_x=0
        for one_smiles in cmpd_smiles_list:
            f.write(one_smiles + "\n")
    return cmpd_smiles_list









#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#       ,AM            .g8""8q. `7MMF'   `7MF'MMP""MM""YMM `7MM"""Mq.`7MMF'   `7MF'MMP""MM""YMM  .M"""bgd                                       M      #
#      AVMM          .dP'    `YM. MM       M  P'   MM   `7   MM   `MM. MM       M  P'   MM   `7 ,MI    "Y                                       M      #
#    ,W' MM          dM'      `MM MM       M       MM        MM   ,M9  MM       M       MM      `MMb.                                           M      #
#  ,W'   MM          MM        MM MM       M       MM        MMmmdM9   MM       M       MM        `YMMNq.                                   `7M'M`MF'  #
#  AmmmmmMMmm        MM.      ,MP MM       M       MM        MM        MM       M       MM      .     `MM                                     VAM,V    #
#        MM   ,,     `Mb.    ,dP' YM.     ,M       MM        MM        YM.     ,M       MM      Mb     dM                                      VVV     #
#        MM   db       `"bmmd"'    `bmmmmd"'     .JMML.    .JMML.       `bmmmmd"'     .JMML.    P"Ybmmd"                                        V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Output #2: Write compounds_properties_list
# Obtain a list of compounds_properties for each sequence (This is for baseline model)
# [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
# Y_Property_Class_#1: threshold 1e-5 (y_prpty_cls_threshold)
# Y_Property_Class_#2: threshold 1e-2 (provided by LEARNING PROTEIN SEQUENCE EMBEDDINGS USING INFORMATION FROM STRUCTURE)
# Get a compound_list including all compounds (compounds here use SMILES representation)
def output_formated_dataset(cmpd_df, 
                            cmpd_df_bi, 
                            seqs_list, 
                            y_prpty_cls_threshold, 
                            target_nme, 
                            data_folder, 
                            smiles_file, 
                            output_folder, 
                            output_file_1):
    # Get all compound SMILES.
    cmpd_list=[] # Non-unique SMILES
    cmpd_smiles_list=[] # Unique SMILES
    cmpd_df_row_num = cmpd_df.shape[0]
    for i in range(cmpd_df_row_num):
        if cmpd_df.loc[i,"CMPD_SMILES"] not in cmpd_list:
            one_compound_smiles = GetUnqSmi.UNQSMI(cmpd_df.loc[i,"CMPD_SMILES"])

            cmpd_list.append(cmpd_df.loc[i,"CMPD_SMILES"])
            cmpd_smiles_list.append(one_compound_smiles)
    print("number of cmpd: ", len(cmpd_smiles_list)) # actually SMILES
    print("number of cmpd duplicates removed: ", len(set(cmpd_smiles_list))) # actually SMILES
    cmpd_smiles_list = list(set(cmpd_smiles_list))

    #cmpd_smiles_list = cmpd_list # !!!!!
    #====================================================================================================#
    # Use "phosphatase_smiles.dat" to obtain a dict for compounds names and SMILES (if exists)
    smiles_compound_dict=dict([])
    smiles_list=[]
    if ((data_folder / smiles_file).exists()):
        with open(data_folder / smiles_file) as f:
            lines = f.readlines()
            for one_line in lines:
                one_line = one_line.replace("\n","")
                one_pair_list = one_line.split("\t") # Get [compound, SMILES]
                smiles_compound_dict[GetUnqSmi.UNQSMI(one_pair_list[1])]=one_pair_list[0]
                smiles_list.append(GetUnqSmi.UNQSMI(one_pair_list[1]))
        print("number of smiles identified: ", len(smiles_compound_dict))
    #print("smiles_compound_dict: ", smiles_compound_dict)
    #====================================================================================================#
    # 
    y_prpty_reg_list = [[None for i in range(len(seqs_list))] for j in range(len(cmpd_smiles_list))]
    y_prpty_cls_2_list = [[None for i in range(len(seqs_list))] for j in range(len(cmpd_smiles_list))]

    cmpd_df = cmpd_df.reset_index()  # make sure indexes pair with number of rows
    count_records = 0 
    count_multiple_records = 0
    unique_seqs_cmpd_pairs_list = []
    for index, row in cmpd_df.iterrows():
        cmpd = GetUnqSmi.UNQSMI(row['CMPD_SMILES']) 
        #cmpd = row['CMPD_SMILES'] # !!!!!
        seqs = row['SEQ'].replace("-", "")
        vals = row[target_nme]
        if seqs not in seqs_list:
            print("seqs not in seqs_list: ", seqs)
            continue
        
        if vals != None: 
            '''
            if (cmpd, seqs) in unique_seqs_cmpd_pairs_list:
                #print("\nprevious record: ", vals )
                #print(  "new value found: ", y_prpty_reg_list[cmpd_smiles_list.index(cmpd)][seqs_list.index(seqs)])
                #count_multiple_records += 1
                '''
            unique_seqs_cmpd_pairs_list.append((cmpd, seqs))
            count_records+=1

        y_prpty_reg_list[cmpd_smiles_list.index(cmpd)][seqs_list.index(seqs)] = vals

    #print("Number of Seqs and Cmpd pairs: ", count_records)
    #print("Number of Seqs and Cmpd pairs w/ multi records: ", count_multiple_records)
    print("len(unique_seqs_cmpd_pairs_list): ", len(unique_seqs_cmpd_pairs_list))
    print("len(set(unique_seqs_cmpd_pairs_list)): ", len(set(unique_seqs_cmpd_pairs_list)))

    #====================================================================================================#
    cmpd_df_bi = cmpd_df_bi.reset_index()  # make sure indexes pair with number of rows
    for index, row in cmpd_df_bi.iterrows():
        cmpd = GetUnqSmi.UNQSMI(row['CMPD_SMILES'])
        #cmpd = row['CMPD_SMILES'] # !!!!!
        seqs = row['SEQ'].replace("-", "")
        vals = row[target_nme]
        if seqs not in seqs_list:
            continue
        y_prpty_cls_2_list[cmpd_smiles_list.index(cmpd)][seqs_list.index(seqs)] = vals

    #====================================================================================================#
    # Get compounds_properties_list
    compounds_properties_list = []
    count_unknown = 0

    for one_cmpd_smiles in cmpd_smiles_list:
        #--------------------------------------------------#
        #y_prpty_reg = np.array(cmpd_df.loc[cmpd_df['CMPD_SMILES'] == cmpd_list[cmpd_smiles_list.index(one_cmpd_smiles)]][target_nme])
        #y_prpty_cls_2 = np.array(cmpd_df_bi.loc[cmpd_df_bi['CMPD_SMILES'] == cmpd_list[cmpd_smiles_list.index(one_cmpd_smiles)]][target_nme])
        #--------------------------------------------------#
        y_prpty_reg = np.array(y_prpty_reg_list[cmpd_smiles_list.index(one_cmpd_smiles)])
        y_prpty_cls_2 = np.array(y_prpty_cls_2_list[cmpd_smiles_list.index(one_cmpd_smiles)])
        y_list_0_1=[]
        for y_value in y_prpty_reg:
            y_list_0_1.append(None if y_value==None else 1 if y_value>=y_prpty_cls_threshold else 0)
        y_prpty_cls=np.array(y_list_0_1)
        #--------------------------------------------------#
        if one_cmpd_smiles in smiles_compound_dict.keys():
            compounds_properties=[smiles_compound_dict[one_cmpd_smiles],y_prpty_reg,y_prpty_cls,y_prpty_cls_2,one_cmpd_smiles]
        else:
            count_unknown+=1
            compounds_properties=["compound_"+str(count_unknown),y_prpty_reg,y_prpty_cls,y_prpty_cls_2,one_cmpd_smiles]
        #print(compounds_properties)
        compounds_properties_list.append(compounds_properties)

    count_y = 0
    for one_cmpd_properties in compounds_properties_list:
        for one_y_data in one_cmpd_properties[1]:
            if one_y_data != None:
                count_y+=1
    print("Number of Data Points: ", count_y)
    #====================================================================================================#
    # compounds_properties
    # [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
    # Y_Property_Class_#1: threshold 1e-5 (y_prpty_cls_threshold)
    # Y_Property_Class_#2: threshold 1e-2 (provided by LEARNING PROTEIN SEQUENCE EMBEDDINGS USING INFORMATION FROM STRUCTURE)
    pickle.dump( compounds_properties_list, open( output_folder / output_file_1, "wb" ) )
    return cmpd_smiles_list, compounds_properties_list











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MMM.     ,MMF' .g8""8q. `7MM"""Mq.   .g8"""bgd       db     `7MN.   `7MF'     `7MM"""Yp, `7MMF'MMP""MM""YMM                                M      #
#    MMMb    dPMM .dP'    `YM. MM   `MM..dP'     `M      ;MM:      MMN.    M         MM    Yb   MM  P'   MM   `7                                M      #
#    M YM   ,M MM dM'      `MM MM   ,M9 dM'       `     ,V^MM.     M YMb   M         MM    dP   MM       MM                                     M      #
#    M  Mb  M' MM MM        MM MMmmdM9  MM             ,M  `MM     M  `MN. M         MM"""bg.   MM       MM                                 `7M'M`MF'  #
#    M  YM.P'  MM MM.      ,MP MM  YM.  MM.    `7MMF'  AbmmmqMA    M   `MM.M         MM    `Y   MM       MM                                   VAMAV    #
#    M  `YM'   MM `Mb.    ,dP' MM   `Mb.`Mb.     MM   A'     VML   M     YMM         MM    ,9   MM       MM                                    VVV     #
#  .JML. `'  .JMML. `"bmmd"' .JMML. .JMM. `"bmmmdPY .AMA.   .AMMA.JML.    YM       .JMMmmmd9  .JMML.   .JMML.                                   V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Get Morgan FP
#====================================================================================================#
# Morgan Function #1
def Get_Morgan_FP_1024(cmpd_smiles_list, output_folder, output_file_2):
    cmpd_SMILES_MorganFP1024_dict=dict([])
    for one_smiles in cmpd_smiles_list:
        rd_mol = Chem.MolFromSmiles(one_smiles)
        MorganFP = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=1024)
        MorganFP_features = np.array(MorganFP)
        cmpd_SMILES_MorganFP1024_dict[one_smiles]=MorganFP_features
    pickle.dump(cmpd_SMILES_MorganFP1024_dict, open(output_folder / output_file_2,"wb"))
    return cmpd_SMILES_MorganFP1024_dict
#====================================================================================================#
# Morgan Function #2
def Get_Morgan_FP_2048(cmpd_smiles_list, output_folder, output_file_3):
    cmpd_SMILES_MorganFP2048_dict=dict([])
    for one_smiles in cmpd_smiles_list:
        rd_mol = Chem.MolFromSmiles(one_smiles)
        MorganFP = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=2048)
        MorganFP_features = np.array(MorganFP)
        cmpd_SMILES_MorganFP2048_dict[one_smiles]=MorganFP_features
    pickle.dump(cmpd_SMILES_MorganFP2048_dict, open(output_folder / output_file_3,"wb"))
    return cmpd_SMILES_MorganFP2048_dict
#====================================================================================================#
# Morgan Function #3
from rdkit.Chem import Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
def ECFP_from_SMILES(smiles, radius=2, bit_len=1024, scaffold=0, index=None): # Not useful here !
    fps = np.zeros((len(smiles), bit_len))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        try:
            if scaffold == 1:
                mol = MurckoScaffold.GetScaffoldForMol(mol)
            elif scaffold == 2:
                mol = MurckoScaffold.MakeScaffoldGeneric(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:
            print(smile)
            fps[i, :] = [0] * bit_len
    return pd.DataFrame(fps, index=(smiles if index is None else index)) 
#====================================================================================================#
# Morgan Function #4
def morgan_fingerprint(smiles: str, radius: int = 2, num_bits: int = 1024, use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.
    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)
    return fp 











#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#`7MM"""Yb.      db  MMP""MM""YMM  db       `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd 
#  MM    `Yb.   ;MM: P'   MM   `7 ;MM:        MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y   MM    MMN.    M .dP'     `M 
#  MM     `Mb  ,V^MM.     MM     ,V^MM.       MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.       MM    M YMb   M dM'       ` 
#  MM      MM ,M  `MM     MM    ,M  `MM       MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq.   MM    M  `MN. M MM          
#  MM     ,MP AbmmmqMA    MM    AbmmmqMA      MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM   MM    M   `MM.M MM.    `7MMF
#  MM    ,dP'A'     VML   MM   A'     VML     MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM   MM    M     YMM `Mb.     MM 
#.JMMmmmdP'.AMA.   .AMMA.JMML.AMA.   .AMMA. .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def X00_Data_Processing(binary_class_bool,
                        data_folder,
                        data_file,
                        data_file_binary,
                        smiles_file,
                        max_seqs_len,
                        y_prpty_cls_threshold,
                        output_folder,
                        output_file_0_0,
                        output_file_0_1,
                        output_file_1,
                        output_file_2,
                        output_file_3):

    # Get_cmpd_df
    cmpd_df, cmpd_df_bi = Get_cmpd_df(data_folder, 
                                      data_file, 
                                      data_file_binary, 
                                      binary_class_bool, 
                                      y_prpty_cls_threshold, 
                                      target_nme)
    beautiful_print(cmpd_df)
    #beautiful_print(cmpd_df_bi)
    #--------------------------------------------------#
    # Output #1: Write a fasta file including all sequences
    seqs_list = output_fasta(cmpd_df, output_folder, output_file_0_0)
    #--------------------------------------------------#
    # Output #2: Write compounds_properties_list
    cmpd_smiles_list, compounds_properties_list = output_formated_dataset(cmpd_df, 
                                                                          cmpd_df_bi, 
                                                                          seqs_list, 
                                                                          y_prpty_cls_threshold, 
                                                                          target_nme, 
                                                                          data_folder, 
                                                                          smiles_file, 
                                                                          output_folder, 
                                                                          output_file_1)
    #--------------------------------------------------#
    '''
    for one_list in compounds_properties_list:
        print(one_list)
        '''
    #--------------------------------------------------#
    # Output #3: Write cmpd_smiles_list
    cmpd_smiles_list = output_smiles(cmpd_smiles_list, output_folder, output_file_0_1)
    #--------------------------------------------------#
    # Morgan Function #1
    cmpd_SMILES_MorganFP1024_dict = Get_Morgan_FP_1024(cmpd_smiles_list, output_folder, output_file_2)
    cmpd_SMILES_MorganFP2048_dict = Get_Morgan_FP_2048(cmpd_smiles_list, output_folder, output_file_3)

    return











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
    Step_code = "X00_"
    #--------------------------------------------------#
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
                        "KM_RANA_1",          # 13
                        ] 
    dataset_nme      = dataset_nme_list[6]
    #--------------------------------------------------#
    #                    dataset_nme           value_col                dataset_path
    data_info_dict   = {"phosphatase"      : ["Conversion"      , "phosphatase_chiral.csv" ,  350], 
                        "kinase"           : ["-Log10Kd"        , "davis.csv"              , 1000],
                        "esterase"         : ["activity"        , "esterase.csv"           , 1200],
                        "halogenase"       : ["Conversion_NaBr" , "halogenase_NaBr.csv"    , 1000],
                        "aminotransferase" : ["LogSpActivity"   , "aminotransferase.csv"   , 1000],
                        "kcat_c"           : ["k_cat"           , "kcat_c.csv"             , 1000],
                        "kcat"             : ["k_cat"           , "kcat.csv"               , 1000],
                        "kcat_mt"          : ["k_cat"           , "kcat_mt.csv"            , 1000],
                        "kcat_wt"          : ["k_cat"           , "kcat_wt.csv"            , 1000],
                        "Ki_all_org"       : ["Ki_Value"        , "Ki_all_org.csv"         , 1000],
                        "Ki_small"         : ["Ki_Value"        , "Ki_small.csv"           , 1000], 
                        "Ki_select"        : ["Ki_Value"        , "Ki_select.csv"          , 1000],
                        "KM_BRENDA"        : ["Km"    , "KM_BRENDA.csv"          , 2000],
                        "KM_RANA_1"        : ["Km"              , "KM_rana_1.csv"          , 2000],   
                        ""                 : [""                , ""                       , 2000],   
                       }

    #--------------------------------------------------#
    target_nme  = data_info_dict[dataset_nme][0]
    data_file   = data_info_dict[dataset_nme][1]
    max_seqs_len = data_info_dict[dataset_nme][2]
    #--------------------------------------------------#
    binary_class_bool = True
    #--------------------------------------------------#
    # Inputs
    data_folder = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
    data_file_binary = data_file.replace(".csv", "_binary.csv") # y_prpty_cls_threshold = around 1e-2
    smiles_file = dataset_nme + "" + "_smiles.dat"

    #--------------------------------------------------#
    # Settings
    
    y_prpty_cls_threshold = 1e-5 # Used for type II screening
    #--------------------------------------------------#
    # Outputs
    output_folder = Path("X_DataProcessing/")
    output_file_0_0 = Step_code + dataset_nme + ".fasta"
    output_file_0_1 = Step_code + dataset_nme + ".smiles"
    output_file_1 = Step_code + dataset_nme + "_compounds_properties_list.p"
    output_file_2 = Step_code + dataset_nme + "_Morgan1024_features.p"
    output_file_3 = Step_code + dataset_nme + "_Morgan2048_features.p"
    ###################################################################################################################
    ###################################################################################################################
    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--binary_class_bool",     type=bool, default = binary_class_bool, help=\
        "If there is a file for binary classification.")

    parser.add_argument("--data_folder",           type=Path, default = data_folder, help=\
        "Path to the directory containing your datasets.")
    parser.add_argument("--data_file",             type=str,  default = data_file, help=\
        "Filename to be read.")
    parser.add_argument("--data_file_binary",      type=str,  default = data_file_binary, help=\
        "Filename (binary classification) to be read.")
    parser.add_argument("--smiles_file",           type=str,  default = smiles_file, help=\
        "Filename of your SMILES file.")
    parser.add_argument("--max_seqs_len",           type=int,  default = max_seqs_len, help=\
        "Maximum Sequence Length.")
    parser.add_argument("--y_prpty_cls_threshold", type=int,  default = y_prpty_cls_threshold, help=\
        "y_prpty_cls_threshold.")

    parser.add_argument("--output_folder",         type=Path, default = output_folder, help=\
        "Path to the directory containing output.")
    parser.add_argument("--output_file_0_0",         type=str,  default = output_file_0_0, help=\
        "Filename of output_file_0_0.")
    parser.add_argument("--output_file_0_1",         type=str,  default = output_file_0_1, help=\
        "Filename of output_file_0_1.")
    parser.add_argument("--output_file_1",         type=str,  default = output_file_1, help=\
        "Filename of output_file_1.")
    parser.add_argument("--output_file_2",         type=str,  default = output_file_2, help=\
        "Filename of output_file_3.")
    parser.add_argument("--output_file_3",         type=str,  default = output_file_3, help=\
        "Filename of output_file_3.")
    args = parser.parse_args()
    #====================================================================================================#
    # Main
    #--------------------------------------------------#
    # Run Main
    X00_Data_Processing(**vars(args))
    print("*" * 50)
    print(Step_code + " Done!")
    #====================================================================================================#



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




