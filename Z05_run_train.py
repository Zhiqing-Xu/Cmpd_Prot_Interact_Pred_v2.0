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
import re
import sys
import copy
import time
import math
import scipy
import torch
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
#--------------------------------------------------#
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from Bio import SeqIO
from tqdm import tqdm
from tpot import TPOTRegressor
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
from Z05_split_data import *
from ZX01_PLOT import *


from Z05A_Conv import SQembConv_CPenc_Model # A
from Z05B_SAtt import SQembSAtt_CPenc_Model # B
from Z05C_CAtt import SQembCAtt_CPenc_Model # C
from Z05B_SAtt import SQembWtLr_CPenc_Model # D


from Z05G_Conv_Mpnn import SQembConv_CPmpn_Model # G





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def run_train(model, 
              optimizer, 
              criterion, 
              epoch_num, 
              train_loader, 
              valid_loader, 
              test_loader, 
              y_scalar, 
              log_value, 
              screen_bool, 
              results_sub_folder, 
              output_file_header, 
              input_var_names_list, 
              target_name = "y_property", 
              ):


    max_r = []

    for epoch in range(epoch_num): 
        begin_time = time.time()
        #====================================================================================================#
        # Train
        model.train()
        count_x = 0
        for one_seqs_cmpd_y_group in train_loader:
            len_train_loader = len(train_loader)
            count_x += 1

            if count_x == 20 :
                print(" " * 12, end = " ") 
            if ((count_x) % 160) == 0:
                print( str(count_x) + "/" + str(len_train_loader) + "->" + "\n" + " " * 12, end=" ")
            elif ((count_x) % 20) == 0:
                print( str(count_x) + "/" + str(len_train_loader) + "->", end=" ")
            #--------------------------------------------------#

            input_vars = [one_seqs_cmpd_y_group[one_var] for one_var in input_var_names_list]
            if   type(model) == SQembConv_CPenc_Model: # A
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda()]

            elif type(model) == SQembSAtt_CPenc_Model: # B
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembCAtt_CPenc_Model: # C
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembWtLr_CPenc_Model: # D
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembConv_CPmpn_Model: # G
                input_vars = [input_vars[0].double().cuda(), input_vars[1]]
            

            target = one_seqs_cmpd_y_group[target_name]
            target = target.double().cuda()

            output, _ = model(*input_vars)

            loss = criterion(output, target.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #====================================================================================================#
        # Validata
        model.eval()
        y_pred_valid = []
        y_real_valid = []
        #--------------------------------------------------#
        for one_seqs_cmpd_y_group in valid_loader:

            input_vars = [one_seqs_cmpd_y_group[one_var] for one_var in input_var_names_list]
            if   type(model) == SQembConv_CPenc_Model: # A
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda()]

            elif type(model) == SQembSAtt_CPenc_Model: # B
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembCAtt_CPenc_Model: # C
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembWtLr_CPenc_Model:         # D
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembConv_CPmpn_Model: # G
                input_vars = [input_vars[0].double().cuda(), input_vars[1]]


            output, _ = model(*input_vars)
            output = output.cpu().detach().numpy().reshape(-1)

            target = one_seqs_cmpd_y_group[target_name]
            target = target.numpy()

            y_pred_valid.append(output)
            y_real_valid.append(target)
        y_pred_valid = np.concatenate(y_pred_valid)
        y_real_valid = np.concatenate(y_real_valid)
        slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)
        #====================================================================================================#
        y_pred = []
        y_real = []
        #--------------------------------------------------#
        for one_seqs_cmpd_y_group in test_loader:

            input_vars = [one_seqs_cmpd_y_group[one_var] for one_var in input_var_names_list]
            if   type(model) == SQembConv_CPenc_Model: # A
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda()]

            elif type(model) == SQembSAtt_CPenc_Model: # B
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembCAtt_CPenc_Model: # C
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembWtLr_CPenc_Model: # D
                input_vars = [input_vars[0].double().cuda(), input_vars[1].double().cuda(), input_vars[2].double().cuda()]

            elif type(model) == SQembConv_CPmpn_Model: # G
                input_vars = [input_vars[0].double().cuda(), input_vars[1]]


            output, _ = model(*input_vars)
            output = output.cpu().detach().numpy().reshape(-1)
            
            target = one_seqs_cmpd_y_group[target_name]
            target = target.numpy()
            
            y_pred.append(output)
            y_real.append(target)
        y_pred = np.concatenate(y_pred)
        y_real = np.concatenate(y_real)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
        max_r.append(r_value)
        #--------------------------------------------------#
        #if log_value == False:
        #    y_pred[y_pred<0]=0
        #--------------------------------------------------#


        loss_copy = copy.copy(loss)
        print("\n" + "_" * 101, end = " ")
        print("\nepoch: {} | time_elapsed: {:5.4f} | train_loss: {:5.4f} | vali_R_VALUE: {:5.4f} | test_R_VALUE: {:5.4f} ".format( 
             str((epoch+1)+1000).replace("1","",1), 

             np.round((time.time()-begin_time), 5),
             np.round(loss_copy.cpu().detach().numpy(), 5), 
             np.round(r_value_va, 5), 
             np.round(r_value, 5),
             )
             )

        r_value, r_value_va = r_value, r_value_va 

        va_MAE  = np.round(mean_absolute_error(y_pred_valid, y_real_valid), 4)
        va_MSE  = np.round(mean_squared_error (y_pred_valid, y_real_valid), 4)
        va_RMSE = np.round(math.sqrt(va_MSE), 4)
        va_R2   = np.round(r2_score(y_real_valid, y_pred_valid), 4)
        va_rho  = np.round(scipy.stats.spearmanr(y_pred_valid, y_real_valid)[0], 4)
        
        
        ts_MAE  = np.round(mean_absolute_error(y_pred, y_real), 4)
        ts_MSE  = np.round(mean_squared_error (y_pred, y_real), 4)
        ts_RMSE = np.round(math.sqrt(ts_MSE), 4) 
        ts_R2   = np.round(r2_score(y_real, y_pred), 4)
        ts_rho  = np.round(scipy.stats.spearmanr(y_pred, y_real)[0], 4)



        print("           | va_MAE: {:4.3f} | va_MSE: {:4.3f} | va_RMSE: {:4.3f} | va_R2: {:4.3f} | va_rho: {:4.3f} ".format( 
             va_MAE, 
             va_MSE,
             va_RMSE, 
             va_R2, 
             va_rho,
             )
             )

        print("           | ts_MAE: {:4.3f} | ts_MSE: {:4.3f} | ts_RMSE: {:4.3f} | ts_R2: {:4.3f} | ts_rho: {:4.3f} ".format( 
             ts_MAE, 
             ts_MSE,
             ts_RMSE, 
             ts_R2, 
             ts_rho,
             )
             )

        y_pred_all = np.concatenate([y_pred, y_pred_valid], axis = None)
        y_real_all = np.concatenate([y_real, y_real_valid], axis = None)

        all_rval = np.round(scipy.stats.pearsonr(y_pred_all, y_real_all), 5)
        all_MAE  = np.round(mean_absolute_error(y_pred_all, y_real_all), 4)
        all_MSE  = np.round(mean_squared_error (y_pred_all, y_real_all), 4)
        all_RMSE = np.round(math.sqrt(ts_MSE), 4) 
        all_R2   = np.round(r2_score(y_real_all, y_pred_all), 4)
        all_rho  = np.round(scipy.stats.spearmanr(y_pred_all, y_real_all)[0], 4)

        print("           | tv_MAE: {:4.3f} | tv_MSE: {:4.3f} | tv_RMSE: {:4.3f} | tv_R2: {:4.3f} | tv_rho: {:4.3f} ".format( 
             all_MAE ,
             all_MSE ,
             all_RMSE,
             all_R2  ,
             all_rho ,
             )
             )
        print("           | tv_R_VALUE:", all_rval)


        print("_" * 101)


        #====================================================================================================#
        if ((epoch+1) % 1) == 0:
            if log_value == False:
                y_pred = y_scalar.inverse_transform(y_pred)
                y_real = y_scalar.inverse_transform(y_real)

            _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)

            reg_scatter_distn_plot(y_pred,
                                   y_real,
                                   fig_size        =  (10,8),
                                   marker_size     =  35,
                                   fit_line_color  =  "brown",
                                   distn_color_1   =  "gold",
                                   distn_color_2   =  "lightpink",
                                   title           =  "Predictions vs. Actual Values\n R = " + \
                                                         str(round(r_value,3)) + \
                                                         ", Epoch: " + str(epoch+1) ,
                                   plot_title      =  "Predictions VS. Acutual Values",
                                   x_label         =  "Actual Values",
                                   y_label         =  "Predictions",
                                   cmap            =  None,
                                   cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                                   font_size       =  18,
                                   result_folder   =  results_sub_folder,
                                   file_name       =  output_file_header + "_TS_" + "epoch_" + str(epoch+1),
                                   ) #For checking predictions fittings.




            _, _, r_value, _ , _ = scipy.stats.linregress(y_pred_valid, y_real_valid)                       
            reg_scatter_distn_plot(y_pred_valid,
                                   y_real_valid,
                                   fig_size        =  (10,8),
                                   marker_size     =  35,
                                   fit_line_color  =  "brown",
                                   distn_color_1   =  "gold",
                                   distn_color_2   =  "lightpink",
                                   title           =  "Predictions vs. Actual Values\n R = " + \
                                                         str(round(r_value,3)) + \
                                                         ", Epoch: " + str(epoch+1) ,
                                   plot_title      =  "Predictions VS. Acutual Values",
                                   x_label         =  "Actual Values",
                                   y_label         =  "Predictions",
                                   cmap            =  None,
                                   cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                                   font_size       =  18,
                                   result_folder   =  results_sub_folder,
                                   file_name       =  output_file_header + "_VA_" + "epoch_" + str(epoch+1),
                                   ) #For checking predictions fittings.


        #====================================================================================================#
            if log_value == False and screen_bool==True:
                y_real = np.delete(y_real, np.where(y_pred == 0.0))
                y_pred = np.delete(y_pred, np.where(y_pred == 0.0))
                y_real = np.log10(y_real)
                y_pred = np.log10(y_pred)
                
                reg_scatter_distn_plot(y_pred,
                                    y_real,
                                    fig_size       = (10,8),
                                    marker_size    = 20,
                                    fit_line_color = "brown",
                                    distn_color_1  = "gold",
                                    distn_color_2  = "lightpink",
                                    title          = "Predictions vs. Actual Values, R = " + \
                                                        str(round(r_value,3)) + \
                                                        ", Epoch: " + str(epoch+1) ,
                                    plot_title     = "Predictions VS. Acutual Values",
                                    x_label        = "Actual Values",
                                    y_label        = "Predictions",
                                    cmap           = None,
                                    font_size      = 18,
                                    result_folder  = results_sub_folder,
                                    file_name      = output_file_header + "_logplot" + "epoch_" + str(epoch+1),
                                    ) #For checking predictions fittings.
    #########################################################################################################
    #########################################################################################################

    return max_r