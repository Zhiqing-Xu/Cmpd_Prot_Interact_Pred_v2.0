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
import math
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

###################################################################################################################
###################################################################################################################
def padding(all_hiddens, max_len, embedding_dim):
    seq_input = []
    seq_mask = []
    for seq in all_hiddens:
        padding_len = max_len - len(seq)
        seq_mask.append(np.concatenate((np.ones(len(seq)),np.zeros(padding_len))).reshape(-1,max_len))
        seq_input.append(np.concatenate(seq,np.zeros((padding_len,embedding_dim))).reshape(-1,max_len,embedding_dim))
    seq_input = np.concatenate(seq_input, axis=0)
    seq_mask = np.concatenate(seq_mask, axis=0)
    return seq_input, seq_mask

#====================================================================================================#
class ATT_dataset(data.Dataset):
    def __init__(self, embedding, compound, label):
        super().__init__()
        self.embedding = embedding
        self.compound = compound
        self.label = label

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.embedding[idx], self.compound[idx], self.label[idx]

    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, compound, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        max_len = np.max([s.shape[0] for s in embedding],0)
        arra = np.full([batch_size,max_len,emb_dim], 0.0)
        seq_mask = []
        for arr, seq in zip(arra, embedding):
            padding_len = max_len - len(seq)
            seq_mask.append(np.concatenate((np.ones(len(seq)),np.zeros(padding_len))).reshape(-1,max_len))
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
        seq_mask = np.concatenate(seq_mask, axis=0)
        return {'embedding': torch.from_numpy(arra), 'mask': torch.from_numpy(seq_mask), 'compound': torch.tensor(list(compound)), 'target': torch.tensor(list(target))}

#====================================================================================================#
def ATT_loader(dataset_class,training_embedding,training_compound,training_target,batch_size,validation_embedding,validation_compound,validation_target,test_embedding,test_compound,test_target):
    
    emb_train = dataset_class(list(training_embedding),list(training_compound),training_target)
    emb_validation = dataset_class(list(validation_embedding),list(validation_compound),validation_target)
    emb_test = dataset_class(list(test_embedding),list(test_compound),test_target)
    trainloader = data.DataLoader(emb_train,batch_size,True,collate_fn=emb_train.collate_fn)
    validation_loader = data.DataLoader(emb_validation,batch_size,False,collate_fn=emb_validation.collate_fn)
    test_loader = data.DataLoader(emb_test,batch_size,False,collate_fn=emb_test.collate_fn)
    return trainloader, validation_loader, test_loader


#====================================================================================================#
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2))
        #print(scores.size())
        scores.masked_fill_(attn_mask,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

#====================================================================================================#
class MultiHeadAttentionwithonekey(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = out_dim
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, out_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        Q = self.W_Q(input_Q).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(input_V.size(0),-1, self.n_heads, self.d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        #print(Q.size(), K.size())
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(input_Q.size(0), -1, self.n_heads * self.d_v)
        output = self.fc(context) # [batch_size, len_q, out_dim]
        return output, attn

#====================================================================================================#
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,cmpd_dim,d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cmpd_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)
            )

    def forward(self, inputs):
        '''
        inputs: [batch_size, src_len, out_dim]
        '''
        inputs = torch.flatten(inputs, start_dim=1)
        output = self.fc(inputs)
        return output

#====================================================================================================#
class EncoderLayer(nn.Module):
    def __init__(self, sub_vab, d_model,d_k,n_heads,d_v,out_dim,cmpd_dim,d_ff): #out_dim = 1, n_head = 4, d_k = 256
        super().__init__()
        self.sub_embedding = nn.Embedding(sub_vab,d_model)
        self.enc_self_attn = MultiHeadAttentionwithonekey(d_model,d_k,n_heads,d_v,out_dim)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,cmpd_dim,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask, input_mask, compound):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, 1], attn: [batch_size, n_heads, src_len, src_len]
        compound = compound.unsqueeze(-1).long()
        compound = self.sub_embedding(compound)
        enc_outputs, attn = self.enc_self_attn(compound, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        #pos_weights = nn.Softmax(dim=1)(enc_outputs.masked_fill_(input_mask.unsqueeze(2).data.eq(0), -1e9)).permute(0,2,1) # [ batch_size, 1, src_len]
        #enc_outputs = torch.matmul(pos_weights,enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, d_model]
        return enc_outputs, attn

#====================================================================================================#
class SQembCAtt_CPenc_Model(nn.Module):
    def __init__(self,sub_vab,d_model,d_k,n_heads,d_v,out_dim,cmpd_dim,d_ff):
        super().__init__()
        self.layers = EncoderLayer(sub_vab,d_model,d_k,n_heads,d_v,out_dim,cmpd_dim,d_ff)

    def get_attn_pad_mask(self, decoder_input, seq_mask):
        batch_size, len_q = decoder_input.size()
        _, len_k = seq_mask.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_mask.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)        

    def forward(self, enc_inputs, input_mask, compound):
        '''
        enc_inputs: [batch_size, src_len, embedding_dim]
        input_mask: [batch_size, src_len]
        '''

        enc_self_attn_mask = self.get_attn_pad_mask(compound,input_mask) # [batch_size, src_len, src_len]
        # enc_outputs: [batch_size, src_len, out_dim], enc_self_attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attn = self.layers(enc_inputs, enc_self_attn_mask, input_mask, compound)
        return enc_outputs, enc_self_attn

#====================================================================================================#
class SQembWtLr_CPenc_Model(nn.Module):
    def __init__(self, d_model,d_k,cmpd_dim, d_ff, dropout):
        super().__init__()
        self.weight_h = weight_norm(nn.Linear(d_model, d_k),dim=None)
        self.weight = weight_norm(nn.Linear(d_k, 1),dim=None)
        self.hidden = weight_norm(nn.Linear(d_model+cmpd_dim, d_ff),dim=None)
        self.dropout = nn.Dropout(p=dropout)
        self.feedforward = weight_norm(nn.Linear(d_ff, 1),dim=None)

    def forward(self, enc_inputs, input_mask, compound):
        '''
        enc_inputs: [batch_size, src_len, embedding_dim]
        input_mask: [batch_size, src_len]
        '''
        position_weight = nn.functional.relu(self.weight_h(enc_inputs))
        position_weight = nn.Softmax(dim=1)(self.weight(position_weight).masked_fill_(input_mask.unsqueeze(2).data.eq(0), -1e9)).permute(0,2,1) #[batch_size, 1, src_len]
        enc_outputs = torch.matmul(position_weight,enc_inputs) #[batch_size, 1, d_model]
        enc_outputs = torch.cat((torch.flatten(enc_outputs, start_dim=1),compound),1) #[batch_size, d_model+cmpd_dim]
        enc_outputs = nn.functional.relu(self.hidden(enc_outputs))
        #print(enc_outputs.size())
        enc_outputs = self.dropout(enc_outputs)
        enc_outputs = self.feedforward(enc_outputs)
        return enc_outputs, position_weight


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == "__main__":
    print()