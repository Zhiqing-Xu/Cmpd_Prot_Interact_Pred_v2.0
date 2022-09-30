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
class CNN_dataset(data.Dataset):
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
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq

        return {'seqs_embeddings': torch.from_numpy(arra), 'cmpd_encodings': torch.tensor(np.array(list(compound))), 'y_property': torch.tensor(np.array(list(target)))}


###################################################################################################################
###################################################################################################################
def generate_CNN_loader(X_tr_seqs, X_tr_cmpd, y_tr,
                        X_va_seqs, X_va_cmpd, y_va,
                        X_ts_seqs, X_ts_cmpd, y_ts,
                        seqs_max_len, batch_size):
    X_y_tr = CNN_dataset(list(X_tr_seqs), list(X_tr_cmpd), y_tr, seqs_max_len)
    X_y_va = CNN_dataset(list(X_va_seqs), list(X_va_cmpd), y_va, seqs_max_len)
    X_y_ts = CNN_dataset(list(X_ts_seqs), list(X_ts_cmpd), y_ts, seqs_max_len)
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
class SQembConv_CPenc_Model(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 kernal_2: int,
                 max_len: int,
                 cmpd_dim: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        #--------------------------------------------------#
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace=False)
        #--------------------------------------------------#
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace=False)
        #--------------------------------------------------#
        self.fc_early = nn.Linear(max_len*hid_dim+cmpd_dim,1)
        #--------------------------------------------------#
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        #self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
        #--------------------------------------------------#
        self.fc_1 = nn.Linear(int(2*max_len*out_dim+cmpd_dim),last_hid)
        self.fc_2 = nn.Linear(last_hid,last_hid)
        self.fc_3 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()


    def forward(self, enc_inputs, compound):
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
        output = self.fc_3(output)
        
        return output, single_conv

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


    def forward(self, enc_inputs, compound):
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
        return output, single_conv




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == "__main__":
    print()
