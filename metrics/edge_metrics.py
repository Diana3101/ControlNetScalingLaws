"""
---- took from https://github.com/hideki-kaneko/pytorch-hed/blob/master/evaluate.py
"""

import numpy as np
import pandas as pd   
    
    
def get_ods_ap(img_pred, img_true, is_uint8, step=0.01):
    '''
        Calculate the ODS(optimal dataset scale) and AP(average precision).
    '''
    if is_uint8:
        img_pred = img_pred / 255.0
        img_true = img_true / 255.0

    thres = 0.0
    list_pred = []
    list_and = []
    num_true = np.sum(img_true)

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

    
    n_cols = len(list_pred)
    f_best = 0.0
    sum_precision = 0.0

    for i in range(n_cols):
        if list_pred[i] != 0:
            ######### 788 - count of same 1.0 in pred and true / 2574 - count of 1.0 in pred image = 0.3
            # precision - count of correctly predicted from all predicted
            precision = list_and[i] / list_pred[i]
            recall = list_and[i] / num_true

            if precision == 0 and recall == 0:
                f = 0.0
            else:
                f = float(2*precision*recall / (recall + precision))
            
            sum_precision += precision
        else:
            sum_precision += 0.0
            f = 0.0
        if f > f_best:
            f_best = f
        
    ap = sum_precision / n_cols
    ods = f_best

    return ods, ap
