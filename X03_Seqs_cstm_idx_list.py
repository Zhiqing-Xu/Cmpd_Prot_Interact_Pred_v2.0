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
'''
import simpleaudio as sa
#--------------------------------------------------#
def alert_sound(frequency, seconds):
    frequency     # Our played note will be 440 Hz
    seconds       # Note duration of 3 seconds
    fs = 44100    # 44100 samples per second
    t = np.linspace(0, seconds, seconds * fs, False)   # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    note = np.sin(frequency * t * 2 * np.pi)           # Generate a 440 Hz sine wave
    audio = note * (2**15 - 1) / np.max(np.abs(note))  # Ensure that highest value is in 16-bit range
    audio = audio.astype(np.int16)                     # Convert to 16-bit data
    play_obj = sa.play_buffer(audio, 1, 2, fs)         # Start playback
    play_obj.wait_done()                               # Wait for playback to finish before exiting
    '''
###################################################################################################################
###################################################################################################################
import sys
import time
import torch
import numpy as ny
import pandas as pd
import pickle
import argparse
import requests
import subprocess
#--------------------------------------------------#
from torch import nn
from torch.utils import data as data
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
from tape import ProteinBertForMaskedLM
#--------------------------------------------------#
from Z01_ModifiedModels import *
from pathlib import Path
#--------------------------------------------------#
from Bio import SeqIO
from tqdm.auto import tqdm
#====================================================================================================#
# Imports for LM
from transformers import BertModel, BertTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraForMaskedLM, ElectraModel
from transformers import T5EncoderModel, T5Tokenizer
from transformers import XLNetModel, XLNetTokenizer
#--------------------------------------------------#
import esm
#====================================================================================================#
# Imports for MSA
from glob import glob
from Bio.Align.Applications import MafftCommandline



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class LoaderClass(data.Dataset):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
#====================================================================================================#
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x,target = None):
        return (x,)






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#     `7MMM.     ,MMF' .M"""bgd       db                                                                                                               #
#       MMMb    dPMM  ,MI    "Y      ;MM:                                                                                                              #
#       M YM   ,M MM  `MMb.         ,V^MM.                                                                                                             #
#       M  Mb  M' MM    `YMMNq.    ,M  `MM                                                                                                             #
#       M  YM.P'  MM  .     `MM    AbmmmqMA                                                                                                            #
#       M  `YM'   MM  Mb     dM   A'     VML                                                                                                           #
#     .JML. `'  .JMML.P"Ybmmd"  .AMA.   .AMMA.                                                                                                         #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def X03C_MSA(dataset_nme,
            model_select, 
            data_folder ,
            input_seqs_fasta_file, 
            output_file_name_header, 
            pretraining_name=None, 
            batch_size=100, 
            xlnet_mem_len=512):
    #====================================================================================================#
    # Name output fasta file with MSA info.
    output_MSA_file = input_seqs_fasta_file
    output_MSA_file = str(output_MSA_file).replace(".fasta", "_MSA.fasta")
    output_MSA_file = str(output_MSA_file).replace("00", "03")
    #--------------------------------------------------#
    # Prepare query for obtaining aligned sequences
    query_str0 = "cd " + str(data_folder)
    query_str2 = "wsl.exe mafft --anysymbol" + " \"" + str(input_seqs_fasta_file) + "\"" + " > \"" + output_MSA_file + "\""
    print("Run in Terminal: ")
    print(query_str0)
    print(query_str2)
    #--------------------------------------------------#
    # Run Query
    query_result = subprocess.check_output(query_str0 + "&&" +  query_str2, shell=True)
    print(query_result)
    #====================================================================================================#
    # Read through all sequences in the input fasta.
    # Open original fasta file here and save length of each sequence in a dictionary.
    with open(data_folder / input_seqs_fasta_file) as f:
        lines = f.readlines()
    #--------------------------------------------------#
    one_sequence = ""
    seqs_list = []
    seqs_nme = ""
    seqs_len_dict = dict([])
    for one_line in lines:
        # 
        if ">seq" in one_line:
            if one_sequence != "":
                seqs_list.append(one_sequence)
                seqs_len_dict[seqs_nme] = len(one_sequence)

            # new sequence start from here
            one_sequence = ""
            seqs_nme = one_line.replace("\n", "")

        if ">seq" not in one_line:
            one_sequence = one_sequence + one_line.replace("\n", "")

        if lines.index(one_line) == len(lines) - 1:
            seqs_list.append(one_sequence)
            seqs_len_dict[seqs_nme] = len(one_sequence)
    #====================================================================================================#
    # Obtain MSA info. Get gaps locations in each sequence.
    # Open aligned sequence file here.
    with open(data_folder / output_MSA_file) as f:
        lines = f.readlines()
    #--------------------------------------------------#
    # 
    MSA_info_dict = dict([])
    #--------------------------------------------------#
    itr = 0
    for one_line in lines:
        
        if ">seq" in one_line:
            itr = 1
            seqs_nme = one_line.replace("\n", "")
            MSA_info_dict[seqs_nme] = {}

            MSA_info_dict[seqs_nme]['seqs_length'] = seqs_len_dict[seqs_nme]
            MSA_info_dict[seqs_nme]['gaps'] = []
        
        if ">seq" not in one_line:
            itr -= 1
            for char in one_line:
                if "-" in char:
                    MSA_info_dict[seqs_nme]['gaps'].append(itr) # gaps are numbered as per 0-indexing
                itr+=1
    #====================================================================================================#
    aligned_seqs_len = (MSA_info_dict['>seq1']['seqs_length'] + len(MSA_info_dict[">seq1"]['gaps']))
    #====================================================================================================#
    # Outputs
    output_MSA_info_file = output_file_name_header.replace("embedding_", "MSA_info") + ".p"
    pickle.dump(MSA_info_dict, open(data_folder / output_MSA_info_file,"wb") )
    #====================================================================================================#
    return 




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#        .g8"""bgd `7MMF'   `7MF' .M"""bgd MMP""MM""YMM   .g8""8q.   `7MMM.     ,MMF'     .M"""bgd `7MM"""Mq. `7MMF'      `7MMF'MMP""MM""YMM           #
#      .dP'     `M   MM       M  ,MI    "Y P'   MM   `7 .dP'    `YM.   MMMb    dPMM      ,MI    "Y   MM   `MM.  MM          MM  P'   MM   `7           #
#      dM'       `   MM       M  `MMb.          MM      dM'      `MM   M YM   ,M MM      `MMb.       MM   ,M9   MM          MM       MM                #
#      MM            MM       M    `YMMNq.      MM      MM        MM   M  Mb  M' MM        `YMMNq.   MMmmdM9    MM          MM       MM                #
#      MM.           MM       M  .     `MM      MM      MM.      ,MP   M  YM.P'  MM      .     `MM   MM         MM      ,   MM       MM                #
#      `Mb.     ,'   YM.     ,M  Mb     dM      MM      `Mb.    ,dP'   M  `YM'   MM      Mb     dM   MM         MM     ,M   MM       MM                #
#        `"bmmmd'     `bmmmmd"'  P"Ybmmd"     .JMML.      `"bmmd"'   .JML. `'  .JMML.    P"Ybmmd"  .JMML.     .JMMmmmmMMM .JMML.   .JMML.              #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def X03B_cstm_idx_list(dataset_nme,
                      model_select, 
                      data_folder ,
                      input_seqs_fasta_file, 
                      output_file_name_header, 
                      pretraining_name=None, 
                      batch_size=100, 
                      xlnet_mem_len=512):

    #--------------------------------------------------#
    # Get CD-hit sequence fasta file.
    seqs_all_ref_file = input_seqs_fasta_file
    seqs_all_ref_file = str(seqs_all_ref_file).replace(".fasta", "_ref.fasta")
    seqs_CD_hit_file = input_seqs_fasta_file # SEQ file with reduced similarities.
    seqs_CD_hit_file = str(seqs_CD_hit_file).replace(dataset_nme, dataset_nme + "_CDhit")
    seqs_CD_hit_clstr_file = str(seqs_CD_hit_file).replace(".fasta", "_clstr.sorted")
    #--------------------------------------------------#
    #====================================================================================================#
    # All sequences.
    with open(data_folder / input_seqs_fasta_file) as f:
        lines = f.readlines()
    one_sequence = ""
    seqs_list = []
    seqs_dict = {}
    seqs_nme = ""
    for line_idx, one_line in enumerate(lines):
        if ">seq" in one_line:
            if one_sequence != "":
                seqs_list.append(one_sequence)
                seqs_dict[seqs_nme] = one_sequence
            # new sequence start from here
            one_sequence = ""
            seqs_nme = one_line.replace("\n", "")
        if ">seq" not in one_line:
            one_sequence = one_sequence + one_line.replace("\n", "")
        if line_idx == len(lines) - 1:
            seqs_list.append(one_sequence)
            seqs_dict[seqs_nme] = one_sequence
    #print("seqs_dict: ", seqs_dict)
    print("number of sequences: ", len(seqs_list))
    #====================================================================================================#
    # All sequences.
    with open(data_folder / seqs_all_ref_file) as f:
        lines = f.readlines()
    one_sequence = ""
    seqs_ref_list = []
    seqs_ref_dict = {}
    seqs_ref_nme = ""
    for line_idx, one_line in enumerate(lines):
        if ">seq" in one_line:
            if one_sequence != "":
                seqs_ref_list.append(one_sequence)
                seqs_ref_dict[seqs_ref_nme] = one_sequence
            # new sequence start from here
            one_sequence = ""
            seqs_ref_nme = one_line.replace("\n", "")
        if ">seq" not in one_line:
            one_sequence = one_sequence + one_line.replace("\n", "")
        if line_idx == len(lines) - 1:
            seqs_ref_list.append(one_sequence)
            seqs_ref_dict[seqs_ref_nme] = one_sequence
    #print("seqs_ref_dict: ", seqs_ref_dict)
    print("number of sequences: ", len(seqs_ref_list))

    #====================================================================================================#
    # CD-hit sequences.
    with open(data_folder / seqs_CD_hit_file) as f:
        lines = f.readlines()
    one_sequence = ""
    seqs_list_CD = []
    seqs_nme = ""
    for line_idx, one_line in enumerate(lines):
        if ">seq" in one_line:
            if one_sequence != "":
                seqs_list_CD.append(one_sequence)
            # new sequence start from here
            one_sequence = ""
            seqs_nme = one_line.replace("\n", "")
        if ">seq" not in one_line:
            one_sequence = one_sequence + one_line.replace("\n", "")
        if line_idx == len(lines) - 1:
            seqs_list_CD.append(one_sequence)
    print("number of sequences after CD-hit screening: ", len(seqs_list_CD))


    #====================================================================================================#
    customized_idx_list = [ seqs_list.index(one_sequence) for one_sequence in seqs_list_CD]
    #print(customized_idx_list)

    #====================================================================================================#
    # CD-hit sequence clusters.
    with open(data_folder / seqs_CD_hit_clstr_file) as f:
        lines = f.readlines()

    cluster_center = ""
    new_cluster = []
    cluster_dict = {}
    for line_idx, one_line in enumerate(lines):
        if one_line.find(">Cluster") != -1:
            if cluster_center != "" :
                cluster_dict[cluster_center] = new_cluster
            
            # new cluster start here,
            cluster_center = ""
            new_cluster = []

        if one_line.find(">") != -1 and one_line.find("...") != -1:
            seqs_nme = one_line[(one_line.index(">")+1) : one_line.index("...")]
            #print("seqs_nme: ", seqs_nme)
            if one_line.find("*") != -1:
                cluster_center = seqs_nme
            else:
                new_cluster.append(seqs_nme)
        if line_idx == len(lines) - 1:
            cluster_dict[cluster_center] = new_cluster

    #print(cluster_dict)
    customized_idx_dict = {}
    for one_cluster_center in list(cluster_dict.keys()):
        one_center_idx = seqs_list.index(seqs_ref_dict[">"+one_cluster_center])
        one_cluster_indices = []
        for one_seqs_nme in cluster_dict[one_cluster_center]:
            one_cluster_indices.append(seqs_list.index(seqs_ref_dict[">"+one_seqs_nme]))
        customized_idx_dict[one_center_idx] = one_cluster_indices
    #====================================================================================================#
    print("len(customized_idx_list): ", len(customized_idx_list))
    print("len(customized_idx_dict): ", len(customized_idx_dict))
    print("len of all values in clusters dict: ", len( [idx2 for idx in customized_idx_dict for idx2 in customized_idx_dict[idx]]    ))

    #print(customized_idx_list)
    #print(customized_idx_dict)

    print("should be empty set: ", set(customized_idx_list)- set(customized_idx_dict.keys()))


    # Outputs
    cstm_splt_file = output_file_name_header.replace("embedding_", "cstm_splt") + ".p"
    pickle.dump(customized_idx_list, open(data_folder / cstm_splt_file,"wb") )

    cstm_splt_clstr_file = output_file_name_header.replace("embedding_", "cstm_splt_clstr") + ".p"
    pickle.dump(customized_idx_dict, open(data_folder / cstm_splt_clstr_file,"wb") )
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
    #====================================================================================================#
    # Args
    Step_code = "X03_"
    dataset_nme_list     = ["phosphatase",        # 0
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
    dataset_nme          = dataset_nme_list[12]
    data_folder          = Path("X_DataProcessing/")
    input_seqs_fasta_file = "X00_" + dataset_nme + ".fasta"
    #====================================================================================================#
    # List Index:          [0]       [1]      [2]       [3]      [4]    [5]      [6]       [7]
    models_list      = ["TAPE_FT", "BERT", "ALBERT", "Electra", "T5", "Xlnet", "ESM_1B", "TAPE"]
    model_select     = models_list[6] ##### !!!!! models_list[3] Electra deprecated !
    pretraining_name = "X01_" + dataset_nme + "_FT_inter_epoch5_trial_training.pt"
    #====================================================================================================#
    output_file_name_header = Step_code + dataset_nme + "_embedding_"
    #====================================================================================================#
    batch_size    = 40
    xlnet_mem_len = 512
    
    #====================================================================================================#
    # ArgParser
    parser = argparse.ArgumentParser(
        description="Preprocesses the sequence datafile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_nme",              type=str,  default = dataset_nme,
                        help = "dataset_nme." )

    parser.add_argument("--model_select",             type=str,  default = model_select,
                        help = "model_select.")

    parser.add_argument("--data_folder",              type=Path, default = data_folder,
                        help = "Path to the directory containing your datasets.")

    parser.add_argument("--input_seqs_fasta_file",     type=str,  default = input_seqs_fasta_file,
                        help = "input_seqs_fasta_file.")

    parser.add_argument("--output_file_name_header",  type=str,  default = output_file_name_header,
                        help = "output_file_name_header.")

    parser.add_argument("--pretraining_name",         type=str,  default = pretraining_name,
                        help = "pretraining_name.")

    parser.add_argument("--batch_size",               type=int,  default = batch_size,
                        help = "Batch size.")

    parser.add_argument("--xlnet_mem_len",            type=int,  default = 512,
                        help = "xlnet_mem_len=512.")

    args = parser.parse_args()

    #====================================================================================================#
    # If dataset_nme is specified, use default pretrained model name and output name.
    vars_dict = vars(args)
    vars_dict["input_seqs_fasta_file"]    = "X00_"    + vars_dict["dataset_nme"] + ".fasta"
    vars_dict["pretraining_name"]        = "X01_"    + vars_dict["dataset_nme"] + "_FT_inter_epoch5_trial_training.pt"
    vars_dict["output_file_name_header"] = Step_code + vars_dict["dataset_nme"] + "_embedding_"

    print(vars_dict)

    #====================================================================================================#
    # Main

    X03B_cstm_idx_list(**vars_dict)

    #X03C_MSA(**vars_dict)


    print("*"*50)
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


