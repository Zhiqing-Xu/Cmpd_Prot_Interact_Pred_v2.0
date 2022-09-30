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
import sys
import time
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import scipy
import random
import subprocess
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
#--------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#--------------------------------------------------#
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from tpot import TPOTRegressor
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
from datetime import datetime
#--------------------------------------------------#
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#########################################################################################################
#########################################################################################################
## Args
Step_code="Y04C_"
data_folder = Path("Y_DataProcessing/")
embedding_file = "Y02_Phosphatase_TAPE_embedding.p"
properties_file= "Y00_compounds_properties_list.p"
scoring_metric_list=[ 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2', 'explained_variance', 'no_grid_search']
scoring_metric=scoring_metric_list[4]
#--------------------------------------------------#
results_folder = Path("Y_DataProcessing/Y04C_intermediate_results/")
i_o_put_file_1 = "Y04B_all_ecfps.p"
i_o_put_file_2 = "Y04B_all_cmpds_ecfps_dict.p"
output_file_3 = "Y04C_all_X_y.p"
output_file = "Y04C_result_"
#--------------------------------------------------#
log_value=False
screen_bool=False
classification_threshold_type=3 # 2: 1e-5, 3: 1e-2
#########################################################################################################
#########################################################################################################
print(">>>>> Creating temporary subfolder and clear past empty folders! <<<<<")
now = datetime.now()
d_t_string = now.strftime("%Y%m%d_%H%M%S")
#====================================================================================================#
results_folder_contents = os.listdir(results_folder)
for item in results_folder_contents:
    if os.path.isdir(results_folder / item):
        try:
            os.rmdir(results_folder / item)
            print("Remove empty folder " + item + "!")
        except:
            print("Found Non-empty folder " + item + "!")
results_sub_folder=Path("Y_DataProcessing/Y04C_intermediate_results/Y04C_" + d_t_string+"/")
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)
print(">>>>> Temporary subfolder created! <<<<<")
#====================================================================================================#
orig_stdout = sys.stdout
f = open(results_sub_folder / 'print_out.txt', 'w')
sys.stdout = f
#########################################################################################################
#########################################################################################################
# Get Input files
# Get Sequence Embeddings from Y02 pickle.
with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
    seqs_embeddings_pkl = pickle.load(seqs_embeddings)
X_seq_embeddings = seqs_embeddings_pkl["embedding"] 
#====================================================================================================#
# Get seqs_properties_list.
with open( data_folder / properties_file, 'rb') as seqs_properties:
    seqs_properties_list = pickle.load(seqs_properties) # [ one_compound ,y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES ]

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# ECFP from CDK java file
def CDK_ECFP(smiles_str,ecfp_type,iteration_number):    
    # Use java file CDKImpl class to get ECFP from cmd line
    query_str1='java -cp .;cdk-2.2.jar CDKImpl ' + smiles_str + ' ' + ecfp_type + ' ' + str(iteration_number)
    query_result = subprocess.check_output(query_str1, shell=True)
    query_result = query_result.decode("gb2312")
    query_result=query_result.replace('[','')
    query_result=query_result.replace(']','')
    query_result=query_result.replace(' ','')
    query_result=query_result.replace('\n','')
    query_result=query_result.replace('\r','')
    if query_result!="":
        if query_result[-1]==',':
            query_result=query_result[0:-1]
        list_of_ecfp=query_result.split(",")
    else:
        list_of_ecfp=[]
    return list_of_ecfp 
#====================================================================================================#
def get_full_ecfp(smiles_str,ecfp_type,iteration_number):   
    # ECFP4 + itr2 or ECFP2 + itr1
    full_ecfp_list=[]
    for i in range(iteration_number+1):
        full_ecfp_list=full_ecfp_list+CDK_ECFP(smiles_str,ecfp_type,i)
    return full_ecfp_list
#====================================================================================================#
def generate_all_ECFPs(list_smiles,ecfp_type="ECFP2",iteration_number=1):
# return a list of ECFPs of all depth for a list of compounds (UNIQUE!!!)
    all_ecfps=set([])
    for smiles_a in list_smiles:
        discriptors = get_full_ecfp(smiles_a,ecfp_type,iteration_number)
        #print(smiles_a)
        all_ecfps=all_ecfps.union(set(discriptors))
    return all_ecfps
#====================================================================================================#
def generate_all_smiles_ecfps_dict(list_smiles,ecfp_type="ECFP2",iteration_number=1):
    all_smiles_ecfps_dict=dict([])
    for smiles_a in list_smiles:
        #print(smiles_a)
        all_smiles_ecfps_dict[smiles_a]=get_full_ecfp(smiles_a,ecfp_type,iteration_number)
    return all_smiles_ecfps_dict
#====================================================================================================#
def generate_all_smiles_ecfps_list_dict(list_smiles,ecfp_type="ECFP2",iteration_number=1):
    all_ecfps=set([])
    all_smiles_ecfps_dict=dict([])
    for smiles_a in list_smiles:
        discriptors = get_full_ecfp(smiles_a,ecfp_type,iteration_number)
        #print(smiles_a)
        all_smiles_ecfps_dict[smiles_a]=discriptors
        all_ecfps=all_ecfps.union(set(discriptors))
    return list(all_ecfps),all_smiles_ecfps_dict
#====================================================================================================#
# Get all compounds ECFP encoded.
all_smiles_list=[]
for one_list_prpt in seqs_properties_list:
    all_smiles_list.append(one_list_prpt[-1])
#print(all_smiles_list)
#--------------------------------------------------#
#all_ecfps,all_smiles_ecfps_dict=generate_all_smiles_ecfps_list_dict(all_smiles_list,ecfp_type="ECFP2",iteration_number=1)
#pickle.dump(all_ecfps, open(data_folder / i_o_put_file_1,"wb") )
#pickle.dump(all_smiles_ecfps_dict, open(data_folder / i_o_put_file_2,"wb"))
#====================================================================================================#
with open( data_folder / i_o_put_file_1, 'rb') as all_ecfps:
    all_ecfps = pickle.load(all_ecfps)
with open( data_folder / i_o_put_file_2, 'rb') as all_smiles_ecfps_dict:
    all_smiles_ecfps_dict = pickle.load(all_smiles_ecfps_dict)

#########################################################################################################
#########################################################################################################
def list_smiles_to_ecfp_through_dict(smiles_list, all_smiles_ecfps_dict):
    ecfp_list=[]
    for one_smiles in smiles_list:
        ecfp_list=ecfp_list + all_smiles_ecfps_dict[one_smiles]
    return ecfp_list
#====================================================================================================#
def smiles_to_ECFP_vec( smiles_x, all_ecfps, all_smiles_ecfps_dict):
    dimension=len(all_ecfps)
    Xi=[0]*dimension
    Xi_ecfp_list=list_smiles_to_ecfp_through_dict( [smiles_x, ] ,all_smiles_ecfps_dict)
    for one_ecfp in Xi_ecfp_list:
        Xi[all_ecfps.index(one_ecfp)]=Xi_ecfp_list.count(one_ecfp)
    return np.array(Xi)
#====================================================================================================#
def Get_X_y_data(X_seq_embeddings,seqs_properties_list,all_ecfps,all_smiles_ecfps_dict,screen_bool,classification_threshold_type):
    X_data = []
    y_data = []
    #print(X_seq_embeddings.shape)
    for i in range(len(seqs_properties_list)):
        for j in range(len(X_seq_embeddings)):
            Xi=smiles_to_ECFP_vec(seqs_properties_list[i][-1], all_ecfps, all_smiles_ecfps_dict)
            Xi_extended=list(np.concatenate((X_seq_embeddings[j,:],Xi)))
            if screen_bool:
                if seqs_properties_list[i][classification_threshold_type][j]==1:
                    X_data.append(Xi_extended)
                    y_data.append(seqs_properties_list[i][1][j])
            else:
                X_data.append(Xi_extended)
                y_data.append(seqs_properties_list[i][1][j])
    X_data=np.array(X_data)
    y_data=np.array(y_data)
    return X_data, y_data
#########################################################################################################
#########################################################################################################
if log_value==True:
    screen_bool=True
    classification_threshold_type=2 # 2: 1e-5, 3: 1e-2
#########################################################################################################
#########################################################################################################
# Get preprocessed dataset of X, y and get ready for machine learning models
X_data, y_data = Get_X_y_data(X_seq_embeddings,seqs_properties_list,all_ecfps,all_smiles_ecfps_dict,screen_bool,classification_threshold_type)
pickle.dump( (X_data, y_data, ) , open( results_folder / output_file_3, "wb" ) )
print("Done getting X_data and y_data!")
print("X_data_dimension: ", X_data.shape, "y_data_dimension: ", y_data.shape )
#########################################################################################################
#########################################################################################################
if log_value==True:
    y_data=np.log10(y_data)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Y04B Learning AND Prediction
# Data Split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.333, random_state=42)
#########################################################################################################
#########################################################################################################
tuned_parameters = [{'kernel': [ 'linear', 'poly', 'rbf', 'sigmoid'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100],'epsilon':[0.1,0.2,0.3,0.5]},]
'''
if scoring_metric!='no_grid_search':
    svr=svm.SVR()
    #print(svr.get_params().keys())
    reg_cv = GridSearchCV(svm.SVR(), tuned_parameters, cv=5, verbose=10)
    reg_cv.fit(X_train,y_train)
    print(reg_cv.best_params_) 
    # GridSearchCV results: {'C': 100, 'epsilon': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}
else:
    reg_cv = svm.SVR(C=100, epsilon=0.1, gamma=0.001, kernel="rbf", )
    reg_cv.fit(X_train, y_train)
'''


reg_cv = DecisionTreeRegressor()
reg_cv.fit(X_train, y_train)
#########################################################################################################
#########################################################################################################
y_real, y_pred = y_test, reg_cv.predict(X_test)
if log_value == False:
    y_pred[y_pred<0]=0

pred_vs_actual_df = pd.DataFrame(np.ones(len(y_pred)))
pred_vs_actual_df["actual"] = y_real
pred_vs_actual_df["predicted"] = y_pred
pred_vs_actual_df.drop(columns=0, inplace=True)
pred_vs_actual_df.head()
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pred_vs_actual_df["actual"].values,pred_vs_actual_df["predicted"].values)
#print("R_value", round(r_value,3))

sns.set_theme(style="darkgrid")

y_interval=max(np.concatenate((y_pred, y_real),axis=0))-min(np.concatenate((y_pred, y_real),axis=0))
x_y_range=(min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)

g = sns.jointplot(x="actual", y="predicted", data=pred_vs_actual_df,
                    kind="reg", truncate=False,
                    xlim=x_y_range, ylim=x_y_range,
                    color="blue",height=7)

g.fig.suptitle("Predictions vs. Actual Values, R = " + str(round(r_value,3)), fontsize=18, fontweight='bold')
g.fig.tight_layout()
g.fig.subplots_adjust(top=0.95)
g.ax_joint.text(0.4,0.6,"", fontsize=12)
g.ax_marg_x.set_axis_off()
g.ax_marg_y.set_axis_off()
g.ax_joint.set_xlabel('Actual Values',fontsize=18 ,fontweight='bold')
g.ax_joint.set_ylabel('Predictions',fontsize=18 ,fontweight='bold')
g.savefig(results_sub_folder / (output_file+".png") )
#########################################################################################################
#########################################################################################################
print(Step_code, " Done!")
sys.stdout = orig_stdout
f.close()
print(Step_code, " Done!")


