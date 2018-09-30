# -*- coding: utf-8 -*-
'''
Created on 09/01/2018
match back labels to html text
@author: WD
'''

import re
import string
import sys
from dateutil.parser import parse
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
import Text_Cleaner as tc



CLUSTER_RADIU = 200

html_parser = HTMLParser.HTMLParser()

def is_nan(x):
    if x is None or x in ('', 'None', [], np.nan, 'NaN') or pd.isna(x):
        return True
    else:
        return False



def get_keyword_idx(kws, txt):
    if not isinstance(kws, list):
        kws = [kws]
    idx_all = []
    for kw in kws:
        if not is_nan(kw):
            kw = str(kw)
            idxs = [m.start() for m in re.finditer(kw, txt)]
            idx_all.extend(idxs)
    return idx_all

def clean_dingzeng_labels(df):
    df['lockup'] = df['lockup'].fillna(0).astype(int)
    return df


#get text tile which contains all labels
def get_label_clusters(dict_kw_idx):
    base_label = 'target' # use this lable as centroi
    cluster_idx_all = []
    for chr_idx in dict_kw_idx[base_label]:
        scope = (chr_idx-CLUSTER_RADIU, chr_idx+CLUSTER_RADIU)
        count_label = 0
        other_label_idxs = [] # to store tamp idx for other found labels, for purpose of visualization
        for label in dict_kw_idx.keys():
            if label == base_label:
                continue
            for idx_label in dict_kw_idx[label]:#check each idx and see if it's in this cluster scope
                if idx_label>scope[0] and idx_label<scope[1]:
                    count_label += 1
                    other_label_idxs.append(idx_label)
                    # print(label)
                    break
        if count_label >= len(dict_kw_idx)-3:# found all labels in this text tile
            # cluster_idx_all.append([chr_idx,other_label_idxs]) # if wish to return seperate centor label and other labels
            cluster_idx_all.append([chr_idx] + other_label_idxs) # if wish to return all lables together

    # print(cluster_idx_all)
    cluster_idx_all = sorted(cluster_idx_all, key=lambda x:len(x), reverse=True)
    cluster_idx_all = cluster_idx_all[:2]
    if len(cluster_idx_all) > 0:
        print(len(cluster_idx_all[0]), ' labels found')
    return cluster_idx_all
#%%

def vis_mark_label_found(txt, idxs):
    idxs_sorted = copy.deepcopy(idxs)
    idxs_sorted = sorted(idxs_sorted, key=lambda x:x, reverse=True)
    # print(idxs_sorted)
    for idx in idxs_sorted:
        # print(txt[idx:idx+20])
        txt = txt[:idx] + "<LABEL>" + txt[idx:]
    return txt

def get_relative_editdistance(t1, t2):
    if t1 is None or t2 is None or t1=='' or t2=='':
        return 1
    else:
        return editdistance.eval(t1, t2)*1.0/max(len(t1), len(t2))



def is_dingzeng_table(col_nm):
    kws_table_cols = ['认购', '禁售', '限售', '发行对象', '发行数量', '申购', '获配', '锁定']
    for k in kws_table_cols:
        if k in col_nm:
            return True
    return False

def is_dingzeng_target(col_nm):
    if col_nm is None or col_nm == '':
        return False
    kws_share_amt = ['姓名','名称','股东名称','发行对象','申购对象','申购对象名称','机构名称','简称','股东姓名']
    min_distance = 1
    # print(col_nm)
    col_nm = re.split(r'\(|（', col_nm)[0].strip()
    for kw in kws_share_amt:
        dist = get_relative_editdistance(col_nm, kw)
        # print(dist, col_nm, kw)
        if dist < min_distance:
            min_distance = dist
    if min_distance < 0.5:
        return True
    else:
        return False


def is_share_amt_col(col_nm):
    if col_nm is None or col_nm == '':
        return False
    kws_share_amt = ['认购数量', '本次认购股数', '申购数量', '累计申购数量', '配售股数', '认购股数', '配售股数',
                     '获配数量', '获配股数', '发行数量', '本次发行', '认购股份数量', '发行增加的股数', '每档数量']
    min_distance = 1
    # print(col_nm)
    col_nm = re.split(r'\(|（', col_nm)[0].strip()
    for kw in kws_share_amt:
        dist = get_relative_editdistance(col_nm, kw)
        # print(dist, col_nm, kw)
        if dist < min_distance:
            min_distance = dist
    if min_distance < 0.5:
        return True
    else:
        return False

def is_money_amt_col(col_nm):
    if col_nm is None or col_nm == '':
        return False
    kws_share_amt = ['认购金额', '实际认购金额', '获配金额', '申购金额', '配售金额', '有效申购金额', '发行金额']
    min_distance = 1
    # print(col_nm)
    col_nm = re.split(r'\(|（', col_nm)[0].strip()
    for kw in kws_share_amt:
        dist = get_relative_editdistance(col_nm, kw)
        # print(dist, col_nm, kw)
        if dist < min_distance:
            min_distance = dist
    if min_distance < 0.5:
        return True
    else:
        return False

def is_price_col(col_nm):
    if col_nm is None or col_nm == '':
        return False
    kws_share_amt = ['报价', '申购价格', '认购价格', '申报价格', '发行价格', '价位', '每档报价', '价格']
    min_distance = 1
    # print(col_nm)
    col_nm = re.split(r'\(|（', col_nm)[0].strip()
    for kw in kws_share_amt:
        dist = get_relative_editdistance(col_nm, kw)
        # print(dist, col_nm, kw)
        if dist < min_distance:
            min_distance = dist
    if min_distance < 0.5:
        return True
    else:
        return False

def is_lockup(col_nm):
    if col_nm is None or col_nm == '':
        return False
    kws_share_amt = ['限售期', '禁售期', '锁定期', ]
    min_distance = 1
    # print(col_nm)
    col_nm = re.split(r'\(|（', col_nm)[0].strip()
    for kw in kws_share_amt:
        dist = get_relative_editdistance(col_nm, kw)
        # print(dist, col_nm, kw)
        if dist < min_distance:
            min_distance = dist
    if min_distance < 0.334:
        return True
    else:
        return False





def extract_dingzeng(df_table: pd.DataFrame):
    # print(df_table)
    df_pred = pd.DataFrame()

    for col in df_table.columns.tolist():
        if is_dingzeng_target(col):
            df_pred['target'] = df_table[col]

        if is_share_amt_col(col) == True:
            df_pred['share'] = df_table[col]
            # print(df_table[col])

        if is_money_amt_col(col):
            df_pred['money'] = df_table[col]
            # print(df_table[col])

        if is_lockup(col):
            df_pred['lockup'] = df_table[col]
            # print(df_table[col])

        if is_price_col(col):
            df_pred['lockup'] = df_table[col]
            # print(df_table[col])
    # print(df_pred)
    return df_pred

# convert pandas numeric value to string;
# determine if it's a float ot int format
def convert_pd_numeric_2_str(num):
    if is_nan(num):
        return 'NONE'
    if isinstance(num, str):
        num_float = float(num)
        if num_float == int(num_float):#sometimes, 123 - > 123.0
            return str(int(num_float))
        else:
            return num
    else:
        if int(num) == num:
            return str(int(num))
        else:
            return str(num)



def label_html_article(df_labels_cur, txt_html):
    for idx, labels in df_labels_cur.iterrows():
        target = labels['target']
        issue_mode = labels['issue_mode']
        if is_nan(issue_mode):
            issue_mode = 'NONE'
        share = convert_pd_numeric_2_str(labels['share'])
        money = convert_pd_numeric_2_str(labels['money'])
        lockup = convert_pd_numeric_2_str(labels['lockup'])
        buy_mode = labels['buy_mode']
        if is_nan(buy_mode):
            buy_mode = 'NONE'
        # print(target, issue_mode, share, money, lockup, buy_mode)

        # rs = re.search("(<img.*?>)", txt_html, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        # if rs != None:
        #     print(rs.group())
        # txt_html = re.sub("(<img.*?>)", "", txt_html, 0, re.IGNORECASE | re.DOTALL | re.MULTILINE)

        for ne in (target, issue_mode, share, money, lockup, target):
            # try:
                txt_html = re.sub(ne, '<label_target style="color:red">' + ne + '</label_target>',
                                  txt_html)
            # except Exception:
            #     print('Wrong NE label:', idx)
            #     return
    return txt_html

if __name__ == '__main__':
    start_ts = datetime.now()

    if os.path.isdir("C:\\projects\\fddc\\"):
        path_proj_folder = "C:\\projects\\fddc\\"
    else:
        path_proj_folder = "F:\\fddc\\"
    path_dir_pdf_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\pdf"
    path_dir_html_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\html"
    path_label_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\dingzeng.train"
    path_dir_output_html_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\html_train"

    print('++++++++++++++++++++++++++++++++++++++++ start running ++++++++++++++++++++++++++++++++++++++++++++++++++')
    s = open(path_label_dz_train, mode='r', encoding='utf-8-sig').readlines()

    my_cols = ["id_doc", "target", "issue_mode", "share", "money", "lockup", "buy_mode"]
    df_label_dz = pd.read_csv(path_label_dz_train,
                              names = my_cols,
                              sep='\t',
                              # encoding = 'utf-8-sig',
                              # header=None,
                              # engine='python',
                              # error_bad_lines=False
                              )
    df_label_dz = clean_dingzeng_labels(df_label_dz)
    # print(df_label_dz.head(5))
    print('Training label size', df_label_dz.shape)
    print('Time of loading labels:', datetime.now() - start_ts)
    start_ts = datetime.now()

    cnt_all_file = 0
    cnt_good_table = 0
    cnt_all_table = 0

    ###################################### attach training labels to text
    #### collect all files first
    id_and_files = []
    for root, dirs, files in os.walk(path_dir_html_dz_train):
        for file in files:
            doc_id = file.split('.')[0]
            path_html = os.path.join(root, file)
            id_and_files.append([doc_id, path_html])

    print('Training File:', len(id_and_files))

    for doc_id, path_html in id_and_files:
        cnt_all_file += 1
        if cnt_all_file % 100 == 0:
            print('processing file', cnt_all_file)
            print('good table vs all table vs all files:', cnt_good_table, cnt_all_table, cnt_all_file)
            print('  Time of processing this batch:', datetime.now() - start_ts)
            start_ts = datetime.now()
        flag_found_good_table = 0

        df_labels_cur = df_label_dz[df_label_dz['id_doc'] == int(doc_id)]
        if df_labels_cur.shape[0] == 0:
            print('doc %s does not have training label!!!!!')
            continue

        handle_file = codecs.open(path_html, mode='r', encoding='utf8')
        txt_html = handle_file.read()
        # txt = html2text.html2text(txt_html)

        txt_html = tc.clean_width(txt_html)  # 全角半角
        txt_html = tc.clean_locale_digit(txt_html)  # remove comma
        txt_html = tc.clean_currency_unit(txt_html)  # currency unit
        txt_html = tc.clean_date(txt_html)
        txt_html = tc.clean_image(txt_html)
        #match labels to txt
        txt_html = label_html_article(df_labels_cur, txt_html)
        ofile = codecs.open(os.path.join(path_dir_output_html_dz_train, doc_id+'.html'), "w", "utf-8")
        ofile.write(txt_html)
        ofile.close()
    print('good table vs all table vs all files:', cnt_good_table, cnt_all_table, cnt_all_file)
    print('Time of labeling all files:', datetime.now() - start_ts)
