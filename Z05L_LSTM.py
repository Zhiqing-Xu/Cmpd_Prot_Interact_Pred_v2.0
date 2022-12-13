# -*- coding: utf-8 -*-

"""

Created on Tue Oct  4 13:47:07 2022



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


###################################################################################################################
###################################################################################################################
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
#--------------------------------------------------#
import torch 
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#--------------------------------------------------#
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
#--------------------------------------------------#
import pickle
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
###################################################################################################################
###################################################################################################################
class LSTM_dataset(data.Dataset):
    def __init__(self, embedding, compound, target, max_len):
        super().__init__()
        self.embedding = embedding
        self.compound = compound
        self.target = target
        self.max_len = max_len
    def __len__(self):
        return len(self.embedding)
    def __getitem__(self, idx):
        return self.embedding[idx], self.compound[idx], self.target[idx]
    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, compound, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size, self.max_len, emb_dim], 0.0)
        lengths = []
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
            lengths.append(seq.shape[0])

        return {'seqs_embeddings': torch.from_numpy(arra), 'seqs_lens': torch.tensor(np.array(lengths)), 'cmpd_encodings': torch.tensor(np.array(list(compound))), 'y_property': torch.tensor(np.array(list(target)))}


###################################################################################################################
###################################################################################################################
def generate_LSTM_loader(X_tr_seqs, X_tr_cmpd, y_tr,
                        X_va_seqs, X_va_cmpd, y_va,
                        X_ts_seqs, X_ts_cmpd, y_ts,
                        seqs_max_len, batch_size):
    X_y_tr = LSTM_dataset(list(X_tr_seqs), list(X_tr_cmpd), y_tr, seqs_max_len)
    X_y_va = LSTM_dataset(list(X_va_seqs), list(X_va_cmpd), y_va, seqs_max_len)
    X_y_ts = LSTM_dataset(list(X_ts_seqs), list(X_ts_cmpd), y_ts, seqs_max_len)
    train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn=X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn=X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn=X_y_ts.collate_fn)
    return train_loader, valid_loader, test_loader



###################################################################################################################
###################################################################################################################
class LoaderClass(data.Dataset):
    def __init__(self, seqs_embeddings, cmpd_encodings, y_property):
        super(LoaderClass, self).__init__()
        self.seqs_embeddings = seqs_embeddings
        self.cmpd_encodings = cmpd_encodings
        self.y_property = y_property
    def __len__(self):
        return self.seqs_embeddings.shape[0]
    def __getitem__(self, idx):
        return self.seqs_embeddings[idx], self.cmpd_encodings[idx], self.y_property[idx]
    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        seqs_embeddings, cmpd_encodings, y_property = zip(*batch)
        batch_size = len(seqs_embeddings)
        seqs_embeddings_dim = seqs_embeddings[0].shape[1]    
        return {'seqs_embeddings': torch.tensor(seqs_embeddings), 'cmpd_encodings': torch.tensor(list(cmpd_encodings)), 'y_property': torch.tensor(list(y_property))}



###################################################################################################################
###################################################################################################################
class SQembLSTM_CPenc_Model(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 latent_dim: int,
                 out_dim: int,
                 cmpd_dim: int,
                 max_len: int,
                 num_layers: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.encoder_LSTM = nn.LSTM(in_dim, hid_dim, batch_first=True, num_layers=num_layers)
        self.mean = nn.Linear(in_features=hid_dim*num_layers, out_features = latent_dim)
        self.log_variance = nn.Linear(in_features=hid_dim*num_layers, out_features = latent_dim)
        #--------------------------------------------------#
        self.fc_early = nn.Linear(max_len*latent_dim+cmpd_dim,1)
        #--------------------------------------------------#
        self.fc_1 = nn.Linear(int(latent_dim+cmpd_dim),last_hid)
        self.fc_2 = nn.Linear(last_hid,last_hid)
        self.fc_3 = nn.Linear(last_hid,1)
        
        #--------------------------------------------------#
        self.num_layers = num_layers
        self.hid_dim = hid_dim; self.latent_dim = latent_dim
    
    def initial_hidden_vars(self, batch_size):
        hidden_cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).double().cuda()
        state_cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).double().cuda()
        return (hidden_cell, state_cell)
    
        
    def encoder(self, x_inputs, padding_length, hidden_encoder):
        # Pad the packed input (already done when input into the NN)
        
        packed_output_encoder, hidden_encoder = self.encoder_LSTM(x_inputs, hidden_encoder)
        output_encoder, _ = nn.utils.rnn.pad_packed_sequence(packed_output_encoder, batch_first=True, total_length=padding_length)
        
        # Estimate the mean and the variance
        mean = self.mean(hidden_encoder[0]).cuda(); output_encoder.cuda()
        log_var = self.log_variance(hidden_encoder[0])
        std = torch.exp(0.5*log_var).cuda()
        
        # Generating unit Gaussian noise
        output_encoder = output_encoder.contiguous().cuda(); batch_size = output_encoder.shape[0]
        seq_len = output_encoder.shape[1]
        noise = torch.randn(batch_size, self.latent_dim).cuda()

    
        z = noise*std + mean
        return z, mean, log_var, hidden_encoder


    def forward(self, x, lengths, compound, hidden_encoder):
        max_length = x.shape[1]
        #hidden_cell = torch.zeros(self.num_layers, x.shape[0], self.hid_dim)
        #state_cell = torch.zeros(self.num_layers, x.shape[0], self.hid_dim)
        lengths = lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True, enforce_sorted=False)
        z, mean, log_var, hidden_encoder = self.encoder(x, max_length, hidden_encoder); z=z.flatten(0,1)
        
        #--------------------------------------------------#
        output = torch.cat((z.cuda(), compound.cuda()) ,1)
        
        #--------------------------------------------------#
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)


        return output


###################################################################################################################
###################################################################################################################

class SQembConv_CPenc_Model_clf(nn.Module):
    def __init__(self,
                 in_dim   : int,
                 hid_dim  : int,
                 kernal_1 : int,
                 out_dim  : int,
                 kernal_2 : int,
                 max_len  : int,
                 cmpd_dim  : int,
                 last_hid : int,
                 dropout  : float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout1 = nn.Dropout(dropout, inplace = True)
        #--------------------------------------------------#
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace = True)
        #--------------------------------------------------#
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace = True)
        #--------------------------------------------------#
        self.fc_early = nn.Linear(max_len*hid_dim+cmpd_dim,1)
        #--------------------------------------------------#
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout3 = nn.Dropout(dropout, inplace = True)
        #self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
        #--------------------------------------------------#
        self.fc_1 = nn.Linear(int(2*max_len*out_dim+cmpd_dim),last_hid)
        self.fc_2 = nn.Linear(last_hid,last_hid)
        self.fc_3 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()

    def forward(self, enc_inputs, compound)
        #--------------------------------------------------#
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout1(output)
        #--------------------------------------------------#
        output_1 = nn.functional.relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        #--------------------------------------------------#
        output_2 = nn.functional.relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)
        #--------------------------------------------------#
        single_conv = torch.cat( (torch.flatten(output_2,1),compound) ,1)
        single_conv = self.cls(self.fc_early(single_conv))
        #--------------------------------------------------#
        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        #--------------------------------------------------#
        output = torch.cat((output_1,output_2),1)
        #--------------------------------------------------#
        #output = self.pooling(output)
        #--------------------------------------------------#
        output = torch.cat( (torch.flatten(output,1), compound) ,1)
        #--------------------------------------------------#
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = torch.sigmoid(self.fc_3(output))
        return output





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

if __name__ == "__main__":
    print()

