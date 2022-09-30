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

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA 


#====================================================================================================#

a=[1,2,3,4,5]
print(a[0:3])











def func_1(a,b,c):
    return a+b+c


x_args = (4,5,6)
a_args = {"a": 1, "b" : 2, "c" : 3}

print(func_1(**a_args))

a = [[1,2], [1,2], [1,2]]
a[0] = [2,3]

b = a[0]




del(a)
print(b)




aa= bb


import chemprop

# training
arguments = [
    '--data_path', './tests/data/regression.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'test_checkpoints_reg',
    '--epochs', '5',
    '--save_smiles_splits'
]


args = chemprop.args.TrainArgs().parse_args(arguments)
print(args)


mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)










###################################################################################################################
###################################################################################################################

def generate_loader(dataset_class,training_embedding,training_compound,training_target,max_len,batch_size,validation_embedding,validation_compound,validation_target,test_embedding,test_compound,test_target):
    
    emb_train = dataset_class(list(training_embedding),list(training_compound),training_target,max_len)
    emb_validation = dataset_class(list(validation_embedding),list(validation_compound),validation_target,max_len)
    emb_test = dataset_class(list(test_embedding),list(test_compound),test_target,max_len)
    trainloader = data.DataLoader(emb_train,batch_size,True,collate_fn=emb_train.collate_fn)
    validation_loader = data.DataLoader(emb_validation,batch_size,False,collate_fn=emb_validation.collate_fn)
    test_loader = data.DataLoader(emb_test,batch_size,False,collate_fn=emb_test.collate_fn)
    return trainloader, validation_loader, test_loader

#====================================================================================================#
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
        arra = np.full([batch_size,self.max_len,emb_dim], 0.0)
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq        
        return {'embedding': torch.from_numpy(arra), 'compound': torch.tensor(list(compound)), 'target': torch.tensor(list(target))}

###################################################################################################################
###################################################################################################################
class CNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 kernal_2: int,
                 max_len: int,
                 cmpd_dim: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc_early = nn.Linear(max_len*hid_dim+cmpd_dim,1)
        self.conv2 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc = nn.Linear(max_len*out_dim+cmpd_dim,1)
        self.cls = nn.Sigmoid()

    def forward(self,enc_inputs, compound):
        """
        input:[batch_size,seq_len,embed_dim]
        """
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout(output)
        single_conv = torch.cat((torch.flatten(output, 1),compound),1)
        single_conv = self.fc_early(single_conv)
        output = nn.functional.relu(self.conv2(output))
        output = self.dropout2(output)
        output = torch.cat((torch.flatten(output, 1),compound),1)
        output = self.fc(output)
        return output, single_conv

###################################################################################################################
###################################################################################################################
class CNN_old1(nn.Module):
    def __init__(self,
                 in_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 max_len: int,
                 cmpd_dim: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernal_1, padding=int((kernal_1-1)/2)) 
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc_1 = nn.Linear(max_len*out_dim+cmpd_dim,last_hid)
        self.fc_2 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()

    def forward(self,enc_inputs, compound):
        #input:[batch_size,seq_len,embed_dim]
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout(output)
        single_conv = torch.cat(  (torch.flatten(output,1),  compound) ,1)
        output = torch.cat(  (torch.flatten(output,1),  compound) ,1)
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        return output, single_conv

###################################################################################################################
###################################################################################################################
class CNN_old2(nn.Module):
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
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc_early = nn.Linear(max_len*hid_dim+cmpd_dim,1)
        self.conv2 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc_1 = nn.Linear(max_len*out_dim+cmpd_dim,last_hid)
        self.fc_2 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()
    def forward(self, enc_inputs, compound):
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout(output)
        single_conv = torch.cat(  (torch.flatten(output,1),  compound) ,1)
        single_conv = self.cls(self.fc_early(single_conv))
        output = nn.functional.relu(self.conv2(output))
        output = self.dropout2(output)
        output = torch.cat(  (torch.flatten(output,1),  compound) ,1)
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        return output, single_conv

###################################################################################################################
###################################################################################################################
'''
model = CNN_old1(
            in_dim = NN_input_dim,
            kernal_1 = 3,
            out_dim = 2, #2
            max_len = seqs_max_len,
            cmpd_dim = X_cmpd_encodings_dim,
            last_hid = 256, #256
            dropout = 0.
            )
'''

###################################################################################################################
###################################################################################################################
def example_training(model,lr,opti,epoch_num,trainloader,validation_loader,save_model,test_loader):
    model.double()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    if opti == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    critation = nn.BCELoss()

    for epoch in range(epoch_num):#1500
        model.train()
        for seq in trainloader:
            enc_inputs, input_mask, target = seq
            enc_inputs, input_mask, target = enc_inputs.double().cuda(),input_mask.double().cuda(), target.double().cuda()
            output, _ = model(enc_inputs,input_mask)
            loss = critation(output,target.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        pre = []
        gt = []
        for seq in validation_loader:
            enc_inputs, input_mask, target = seq
            enc_inputs, input_mask = enc_inputs.double().cuda(), input_mask.double().cuda()
            output, _ = model(enc_inputs,input_mask)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            pre.append(output)
            gt.append(target)
        pre = np.concatenate(pre)
        gt = np.concatenate(gt)
        validation_auc = roc_auc_score(gt,pre)  
        print("epoch: {} | loss: {} | valiloss: {}".format(epoch,loss,validation_auc))
    if save_model:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, str(epoch)+"_epoch_trial_training.pt")
    test_pre = []
    test_gt = []
    for seq in test_loader:
            enc_inputs, input_mask, target = seq
            input, input_mask = input.double().cuda(), input_mask.double().cuda()
            output, _ = model(input, input_mask)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            test_pre.append(output)
            test_gt.append(target)
    test_pre = np.concatenate(test_pre)
    test_gt = np.concatenate(test_gt)
    fpr,tpr,_ = roc_curve(test_gt,test_pre)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()     

#====================================================================================================#
print("Test Done~")




















