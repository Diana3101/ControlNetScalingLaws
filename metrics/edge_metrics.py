"""
---- took from https://github.com/hideki-kaneko/pytorch-hed/blob/master/evaluate.py
"""

import numpy as np
import pandas as pd
import os
import csv
import argparse
from tqdm import tqdm
from PIL import Image    
    
    
def get_ods_ap(img_pred, img_true, step=0.01):
    '''
        Calculate the ODS(optimal dataset scale) and AP(average precision).
    '''
    img_pred = img_pred / 255.0
    img_true = img_true / 255.0

    thres = 0.0
    list_pred = []
    list_and = []
    num_true = [np.sum(img_true)]
    while thres < 1.0:
        img_thred = (img_pred > thres).astype(np.int32)
        num_pred = np.sum(img_thred) 
        ### np.logical_and(0, 0) -> False 
        ### np.logical_and(0, 1) -> False 
        ### np.logical_and(1, 1) -> True 
        ### np.logical_and(1, -2) -> True
        # num_and - count of pixels where pred image and true image has edge (1)
        num_and = np.sum(np.logical_and(img_thred, img_true))
        list_pred.append(num_pred)
        list_and.append(num_and)
        thres += step

    df_pred = pd.DataFrame(list_pred).T
    df_true = pd.DataFrame([num_true])
    df_and = pd.DataFrame(list_and).T
    
    if not len(df_pred) == len(df_true) == len(df_and):
        print("Different rows")
        return 
    n_cols = len(df_pred.columns)
    f_best = 0.0
    sum_precision = 0.0
    for i in range(n_cols):
        try:
            ######### 788 - count of same 1.0 in pred and true / 2574 - count of 1.0 in pred image = 0.3
            # precision - count of correctly predicted from all predicted
            precision = int(np.sum(df_and.iloc[:,i])) / int(np.sum(df_pred.iloc[:,i]))
            recall = np.sum(df_and.iloc[:,i]) / np.sum(df_true.iloc[:])

            f = float(2*precision*recall / (recall + precision))

            sum_precision += precision
        except ZeroDivisionError:
            f = 0.0
        if f > f_best:
            f_best = f
        
    ois = f_best
    ap = sum_precision / n_cols
    # from source code
    ods = ois
    return ods, ap
