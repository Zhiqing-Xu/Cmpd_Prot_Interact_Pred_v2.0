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

#====================================================================================================#





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#      `7MM"""YMM  `7MMM.     ,MMF'`7MM"""Yp,     `7MM"""YMM `7MMF'   `7MF'`7MN.   `7MF'  .g8"""bgd                                                    #
#        MM    `7    MMMb    dPMM    MM    Yb       MM    `7   MM       M    MMN.    M  .dP'     `M                                                    #
#        MM   d      M YM   ,M MM    MM    dP       MM   d     MM       M    M YMb   M  dM'       `                                                    #
#        MMmmMM      M  Mb  M' MM    MM"""bg.       MM""MM     MM       M    M  `MN. M  MM                                                             #
#        MM   Y  ,   M  YM.P'  MM    MM    `Y       MM   Y     MM       M    M   `MM.M  MM.                                                            #
#        MM     ,M   M  `YM'   MM    MM    ,9       MM         YM.     ,M    M     YMM  `Mb.     ,'                                                    #
#      .JMMmmmmMMM .JML. `'  .JMML..JMMmmmd9      .JMML.        `bmmmmd"'  .JML.    YM    `"bmmmd'                                                     #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def X03_embedding_LM(dataset_nme,
                     model_select, 
                     data_folder ,
                     input_seqs_fasta_file, 
                     output_file_name_header, 
                     pretraining_name=None, 
                     batch_size=100, 
                     xlnet_mem_len=512):

    #====================================================================================================#
    assert model_select in models_list, "query model is not found, currently support ESM-1b, TAPE, BERT, AlBERT, Electra, T5 and Xlnet !!"
    #====================================================================================================#
    input_file = data_folder / input_seqs_fasta_file #data path (fasta)
    output_file = data_folder / (output_file_name_header + model_select + ".p") #output path (pickle)




    ###################################################################################################################
    #                               MMP""MM""YMM       db      `7MM"""Mq. `7MM"""YMM                                  #
    #                               P'   MM   `7      ;MM:       MM   `MM.  MM    `7                                  #
    #                                    MM          ,V^MM.      MM   ,M9   MM   d                                    #
    #                                    MM         ,M  `MM      MMmmdM9    MMmmMM                                    #
    #                                    MM         AbmmmqMA     MM         MM   Y  ,                                 #
    #                                    MM        A'     VML    MM         MM     ,M                                 #
    #                                  .JMML.    .AMA.   .AMMA..JMML.     .JMMmmmmMMM                                 #
    ###################################################################################################################
    if model_select == "TAPE_FT" or model_select == "TAPE":
        model = ProteinBertForMaskedLM.from_pretrained('bert-base')
        #--------------------------------------------------#
        if model_select == "TAPE":
            pretraining_name == None
        #--------------------------------------------------#
        if pretraining_name is not None:
            checkpoint = torch.load( data_folder / pretraining_name )
            model.load_state_dict(checkpoint['model_state_dict'])
        #--------------------------------------------------#
        model.mlm = Identity()
        model.eval()
        embed = datasets.EmbedDataset(data_file=input_file,tokenizer="iupac")
        loader = data.DataLoader(embed,batch_size,False,collate_fn = embed.collate_fn)
        #--------------------------------------------------#
        count_x = 0
        model.cuda()
        #--------------------------------------------------#
        seq_encodings = []
        seq_all_hiddens = []
        seq_ids = []
        for seq_batch in loader:
            count_x+=1
            ids, input_ids, input_mask = seq_batch["ids"],seq_batch["input_ids"],seq_batch["input_mask"]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                #print(seq_emd)
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print("features.shape: ", features.shape)
            seq_encodings.append(features)
            seq_ids += ids
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")




    ###################################################################################################################
    #                               `7MM"""Yp, `7MM"""YMM  `7MM"""Mq.  MMP""MM""YMM                                   #
    #                                 MM    Yb   MM    `7    MM   `MM. P'   MM   `7                                   #
    #                                 MM    dP   MM   d      MM   ,M9       MM                                        #
    #                                 MM"""bg.   MMmmMM      MMmmdM9        MM                                        #
    #                                 MM    `Y   MM   Y  ,   MM  YM.        MM                                        #
    #                                 MM    ,9   MM     ,M   MM   `Mb.      MM                                        #
    #                               .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.   .JMML.                                      #
    ###################################################################################################################
    if model_select == "BERT":
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        #--------------------------------------------------#
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        model.cuda()
        model.eval()
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        #--------------------------------------------------#
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = []
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")







    ###################################################################################################################
    #                         db      `7MMF'      `7MM"""Yp, `7MM"""YMM  `7MM"""Mq.  MMP""MM""YMM                     #
    #                        ;MM:       MM          MM    Yb   MM    `7    MM   `MM. P'   MM   `7                     #
    #                       ,V^MM.      MM          MM    dP   MM   d      MM   ,M9       MM                          #
    #                      ,M  `MM      MM          MM"""bg.   MMmmMM      MMmmdM9        MM                          #
    #                      AbmmmqMA     MM      ,   MM    `Y   MM   Y  ,   MM  YM.        MM                          #
    #                     A'     VML    MM     ,M   MM    ,9   MM     ,M   MM   `Mb.      MM                          #
    #                   .AMA.   .AMMA..JMMmmmmMMM .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.   .JMML.                        #
    ###################################################################################################################
    if model_select == "ALBERT":
        tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
        model = AlbertModel.from_pretrained("Rostlab/prot_albert")
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        #--------------------------------------------------#
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        model.cuda()
        model.eval()
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        #--------------------------------------------------#
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")



    ###################################################################################################################
    #         `7MM"""YMM  `7MM  `7MM"""YMM    .g8"""bgd MMP""MM""YMM `7MM"""Mq.        db                .g8""8q.     #
    #           MM    `7    MM    MM    `7  .dP'     `M P'   MM   `7   MM   `MM.      ;MM:             .dP      YM.   #
    #           MM   d      MM    MM   d    dM'       `      MM        MM   ,M9      ,V^MM.            dM'    ,V`MM   #
    #           MMmmMM      MM    MMmmMM    MM               MM        MMmmdM9      ,M  `MM            MM   ,M'  MM   #
    #           MM   Y  ,   MM    MM   Y  , MM.              MM        MM  YM.      AbmmmqMA           MM.,V'   ,MP   #
    #           MM     ,M   MM    MM     ,M `Mb.     ,'      MM        MM   `Mb.   A'     VML          `Mb     ,dP'   #
    #         .JMMmmmmMMM .JMML..JMMmmmmMMM   `"bmmmd'     .JMML.    .JMML. .JMM..AMA.   .AMMA.          `"bmmd"'     #
    ###################################################################################################################
    if model_select == "Electra": ##### !!!!! Deprecated!
        generatorModelUrl = 'https://www.dropbox.com/s/5x5et5q84y3r01m/pytorch_model.bin?dl=1'
        discriminatorModelUrl = 'https://www.dropbox.com/s/9ptrgtc8ranf0pa/pytorch_model.bin?dl=1'
        generatorConfigUrl = 'https://www.dropbox.com/s/9059fvix18i6why/config.json?dl=1'
        discriminatorConfigUrl = 'https://www.dropbox.com/s/jq568evzexyla0p/config.json?dl=1'
        vocabUrl = 'https://www.dropbox.com/s/wck3w1q15bc53s0/vocab.txt?dl=1'
        downloadFolderPath = 'models/electra/'
        discriminatorFolderPath = os.path.join(downloadFolderPath, 'discriminator')
        generatorFolderPath = os.path.join(downloadFolderPath, 'generator')
        discriminatorModelFilePath = os.path.join(discriminatorFolderPath, 'pytorch_model.bin')
        generatorModelFilePath = os.path.join(generatorFolderPath, 'pytorch_model.bin')
        discriminatorConfigFilePath = os.path.join(discriminatorFolderPath, 'config.json')
        generatorConfigFilePath = os.path.join(generatorFolderPath, 'config.json')
        vocabFilePath = os.path.join(downloadFolderPath, 'vocab.txt')

        if not os.path.exists(discriminatorFolderPath):
            os.makedirs(discriminatorFolderPath)
        if not os.path.exists(generatorFolderPath):
            os.makedirs(generatorFolderPath)
        def download_file(url, filename):
            response = requests.get(url, stream=True)
            with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                                total=int(response.headers.get('content-length', 0)),
                                desc=filename) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)
        if not os.path.exists(generatorModelFilePath):
            download_file(generatorModelUrl, generatorModelFilePath)
        if not os.path.exists(discriminatorModelFilePath):
            download_file(discriminatorModelUrl, discriminatorModelFilePath)
        if not os.path.exists(generatorConfigFilePath):
            download_file(generatorConfigUrl, generatorConfigFilePath)
        if not os.path.exists(discriminatorConfigFilePath):
            download_file(discriminatorConfigUrl, discriminatorConfigFilePath)
        if not os.path.exists(vocabFilePath):
            download_file(vocabUrl, vocabFilePath)

        tokenizer = ElectraTokenizer(vocabFilePath, do_lower_case=False )
        model = ElectraModel.from_pretrained(discriminatorFolderPath)
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")



    ###################################################################################################################
    #                                             MMP""MM""YMM   M******                                              #
    #                                             P'   MM   `7  .M                                                    #
    #                                                  MM       |bMMAg.                                               #
    #                                                  MM            `Mb                                              #
    #                                                  MM             jM                                              #
    #                                                  MM       (O)  ,M9                                              #
    #                                                .JMML.      6mmm9                                                #
    ###################################################################################################################
    if model_select == "T5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")



    ###################################################################################################################
    #                       `YMM'   `MP' `7MMF'      `7MN.   `7MF'`7MM"""YMM  MMP""MM""YMM                            #
    #                         VMb.  ,P     MM          MMN.    M    MM    `7  P'   MM   `7                            #
    #                          `MM.M'      MM          M YMb   M    MM   d         MM                                 #
    #                            MMb       MM          M  `MN. M    MMmmMM         MM                                 #
    #                          ,M'`Mb.     MM      ,   M   `MM.M    MM   Y  ,      MM                                 #
    #                         ,P   `MM.    MM     ,M   M     YMM    MM     ,M      MM                                 #
    #                       .MM:.  .:MMa..JMMmmmmMMM .JML.    YM  .JMMmmmmMMM    .JMML.                               #
    ###################################################################################################################
    if model_select == "Xlnet":
        tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
        model = XLNetModel.from_pretrained("Rostlab/prot_xlnet",mem_len=xlnet_mem_len)
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                padded_seq_len = len(input_mask[seq_num])
                seq_emd = output[seq_num][padded_seq_len-seq_len:padded_seq_len-2]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")





    ###################################################################################################################
    #                   `7MM"""YMM   .M"""bgd `7MMM.     ,MMF'                      *MM                               #
    #                     MM    `7  ,MI    "Y   MMMb    dPMM                 __,     MM                               #
    #                     MM   d    `MMb.       M YM   ,M MM                `7MM     MM,dMMb.                         #
    #                     MMmmMM      `YMMNq.   M  Mb  M' MM                  MM     MM    `Mb                        #
    #                     MM   Y  , .     `MM   M  YM.P'  MM      mmmmm       MM     MM     M8                        #
    #                     MM     ,M Mb     dM   M  `YM'   MM                  MM     MM.   ,M9                        #
    #                   .JMMmmmmMMM P"Ybmmd"  .JML. `'  .JMML.              .JMML.   P^YbmdP'                         #
    ###################################################################################################################
    if model_select == "ESM_1B":
        #model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id),str(seq_record.seq)))        
        model.eval()
        model.cuda()
        seq_encodings = []
        seq_all_hiddens = []
        seq_ids = []
        for i in range(0,len(data_set),batch_size):
            print()
            if i+batch_size<=len(data_set):
                batch = data_set[i:i+batch_size]
            else:
                batch = data_set[i:]
            #print(batch)
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            seq_ids += batch_labels
            print(batch_tokens.size())
            batch_tokens = batch_tokens.cuda()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            results = results["representations"][33].cpu().detach()
            print(results.size())
            sequence_representations = []
            for i, ( _ , seq ) in enumerate(batch):
                seq_all_hiddens.append(results[i, 1 : len(seq) + 1].numpy())
                sequence_representations.append(results[i, 1 : len(seq) + 1].mean(0))
            sequence_representations = np.stack(sequence_representations)
            seq_encodings.append(sequence_representations)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")

    ###################################################################################################################
    ###################################################################################################################
    return seq_embedding_output




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
    vars_dict["pretraining_name"]         = "X01_"    + vars_dict["dataset_nme"] + "_FT_inter_epoch5_trial_training.pt"
    vars_dict["output_file_name_header"]  = Step_code + vars_dict["dataset_nme"] + "_embedding_"

    print(vars_dict)

    #====================================================================================================#
    # Main
    X03_embedding_LM(**vars_dict)



    #alert_sound(440,0.5)
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


