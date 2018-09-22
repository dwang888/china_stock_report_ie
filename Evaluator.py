# -*- coding: utf-8 -*-
'''
Created on 6/12c/2018
label doc from Boson NLP
@author: WD
'''
#%%
# import time
import re
import string
import sys
from dateutil.parser import parse
# from multiprocessing import Pool
import uuid
from datetime import *
import os
import pandas as pd
import numpy as np
import html2text
import codecs
import copy
import editdistance
from utils import HTMLParser
from utils import TextUtils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def is_nan(x):
    if x is None or x in ('', 'None', [], np.nan, 'NaN'):
        return True
    else:
        return False

def calculate_precision():
    pass


def get_precision_recall_f1(y_test, y_pred):
    u = set.intersection(set(y_test), set(y_pred))
    precision = len(u) / len(y_pred)
    recall = len(u) / len(y_test)
    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)

def get_precision_recall_f1_list(y_test, y_pred):
    set_test = set(y_test)
    overlap = [item for item in y_pred if item in set_test]
    precision = len(overlap) / len(y_pred)
    recall = len(set(overlap)) / len(set(y_test))
    print(len(overlap), len(y_test), len(y_pred))
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(f1, 4)


if __name__ == '__main__':

    if os.path.isdir("C:\\projects\\fddc\\"):
        path_proj_folder = "C:\\projects\\fddc\\"
    else:
        path_proj_folder = "F:\\fddc\\"

    path_label_dz_train = os.path.join(path_proj_folder, "data_official\\round1_train_20180518\\dingzeng\\dingzeng.train")
    my_cols = ["doc_id", "target", "issue_mode", "share", "money", "lockup", "buy_mode"]
    df_label_dz = pd.read_csv(path_label_dz_train,
                              names=my_cols,
                              sep='\t',
                              )
    df_label_dz['lockup'] = df_label_dz['lockup'].fillna(999).astype(int).astype(str)
    df_label_dz = df_label_dz.sort_values('doc_id')
    print(df_label_dz.head(10))

    path_output_label_pred = path_proj_folder + "lalel_pred_dingzeng.csv"
    df_label_dz_pred = pd.read_csv(path_output_label_pred,
                              )
    df_label_dz_pred = df_label_dz_pred.sort_values('doc_id')
    df_label_dz_pred['lockup'] = df_label_dz_pred['lockup'].replace(np.nan, 999)
    df_label_dz_pred['lockup'] = df_label_dz_pred['lockup'].apply(lambda x:str(int(x)))
    df_label_dz_pred['lockup'] = df_label_dz_pred['lockup'].replace('999', '')
    print(df_label_dz_pred.head(10))


    for col in ["target", "share", "money", "lockup", "buy_mode"][3:4]:
        labels_test = (df_label_dz['doc_id'].astype(str) + '_' + df_label_dz[col].astype(str)).tolist()
        labels_pred = (df_label_dz_pred['doc_id'].astype(str)+'_'+df_label_dz_pred[col].astype(str)).tolist()
        print(labels_test[:100])
        print(labels_pred[100:])
        precision, recall, f1 = get_precision_recall_f1_list(labels_test, labels_pred)
        print('precision, recall, f1 - ', col)
        print(precision, recall, f1)
