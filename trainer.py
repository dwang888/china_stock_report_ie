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

#%%
CLUSTER_RADIU = 200

html_parser = HTMLParser.HTMLParser()

def is_nan(x):
    if x is None or x in ('', 'None', [], np.nan, 'NaN'):
        return True
    else:
        return False

def extend_amount_money(x):
    # x = 50000000
    if is_nan(x):
        return []
    extensions = []
    x = float(x)

    # form 0: original, 181000006.3
    extensions.append(str(x))
    # form 0: original, 181000006
    extensions.append('{:.0f}'.format(x))
    # form 1: original, 181000006.3
    extensions.append('{:.1f}'.format(x))
    # form 2: 181000006.30
    extensions.append('{:.2f}'.format(x))
    # form 3: 181,000,006
    extensions.append('{:,.0f}'.format(x))
    # form 4: 181,000,006.3
    extensions.append('{:,.1f}'.format(x))
    # form 3: 181,000,006.30
    extensions.append('{:,.2f}'.format(x))
    # form 9: 18100万
    extensions.append('{:.0f}'.format(x/10000))
    # form 5: original, 18100.1万
    extensions.append('{:.1f}'.format(x/10000))
    # form 6: 18100.10
    extensions.append('{:.2f}'.format(x/10000))
    # form 9: 18,100万
    extensions.append('{:,.0f}'.format(x / 10000))
    # form 8: 18,100.0万
    extensions.append('{:,.1f}'.format(x/10000))
    # form 7: 18,100.00万
    extensions.append('{:,.2f}'.format(x / 10000))
    # print('<->'.join(extensions))
    
    extensions = list(set(extensions))

    return extensions

def extend_amount_share(x):
    extensions = []
    x = float(x)

    # form 0: original, 181000006.3
    extensions.append(str(x))
    # form 0: original, 181000006
    extensions.append('{:.0f}'.format(x))
    # form 1: original, 181000006.3
    extensions.append('{:.1f}'.format(x))
    # form 2: 181000006.30
    extensions.append('{:.2f}'.format(x))
    # form 3: 181,000,006
    extensions.append('{:,.0f}'.format(x))
    # form 4: 181,000,006.3
    extensions.append('{:,.1f}'.format(x))
    # form 3: 181,000,006.30
    extensions.append('{:,.2f}'.format(x))
    # form 9: 18100万
    extensions.append('{:.0f}'.format(x / 10000))
    # form 5: original, 18100.1万
    extensions.append('{:.1f}'.format(x / 10000))
    # form 6: 18100.10
    extensions.append('{:.2f}'.format(x / 10000))
    # form 9: 18,100万
    extensions.append('{:,.0f}'.format(x / 10000))
    # form 8: 18,100.0万
    extensions.append('{:,.1f}'.format(x / 10000))
    # form 7: 18,100.00万
    extensions.append('{:,.2f}'.format(x / 10000))
    # print('<->'.join(extensions))

    return extensions

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

def locate_labels_idx(labels, text):
    # print(labels)
    kws_money = extend_amount_money(labels['money'])
    indices_money = get_keyword_idx(kws_money, text)
    kws_share = extend_amount_share(labels['share'])
    indices_share = get_keyword_idx(kws_share, text)

    indices_target = get_keyword_idx([labels['target']], text)
    indices_issue_mode = get_keyword_idx([labels['issue_mode']], text)
    indices_lockup = get_keyword_idx([labels['lockup']], text)
    indices_buy_mode = get_keyword_idx([labels['buy_mode']], text)

    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # print(labels['target'], indices_target)
    # print('\n||||||||||||||||||||||||||||||||||\n'.join([text[idx-100:idx+100] for idx in indices_target]))
    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # print(labels['issue_mode'], indices_issue_mode)
    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # print(labels['share'], indices_share)
    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # print(labels['money'], indices_money)
    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # print(labels['lockup'], indices_lockup)
    # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    # print(labels['buy_mode'], indices_buy_mode)

    dict_kw_idx = dict(zip(
        ['target','issue_mode','share','money','lockup','buy_mode'], 
        [indices_target, indices_issue_mode, indices_share, indices_money, indices_lockup, indices_buy_mode]))
    return dict_kw_idx

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


if __name__ == '__main__':
    if os.path.isdir("C:\\projects\\fddc\\"):
        path_proj_folder = "C:\\projects\\fddc\\"
    else:
        path_proj_folder = "F:\\fddc\\"
    path_dir_pdf_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\pdf"
    path_dir_html_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\html"
    path_label_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\dingzeng.train"
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


    cnt_all_file = 0
    cnt_good_table = 0
    cnt_all_table = 0
    # scan each trainin label
    for root, dirs, files in os.walk(path_dir_html_dz_train):
        for file in files:
            cnt_all_file += 1
            if cnt_all_file % 100 == 0:
                print('processing file', cnt_all_file)
                print('good table vs all table vs all files:', cnt_good_table, cnt_all_table, cnt_all_file)
            flag_found_good_table = 0
            #find corresponding labels
            doc_id = file.split('.')[0]
            df_labels_cur = df_label_dz[df_label_dz['id_doc'] == int(doc_id)]
            if df_labels_cur.shape[0] == 0:
                continue

            #load file text
            path_html = os.path.join(root, file)
            table_dicts = html_parser.parse_table(path_html)
            cnt_all_table += len(table_dicts)
            for table_dict in table_dicts:

                #traverse each row of table, check wheter column name is in some keywords
                for pair in table_dict.items():
                    elements = pair[1].values()
                    elements = ' '.join(elements)
                    # print(list(elements))
                    for kw in kws_table_cols:
                        if kw in elements:
                            cnt_good_table += 1
                            flag_found_good_table = 1
                            # display good table

                            break
                    if flag_found_good_table == 1:
                        break
                if flag_found_good_table == 1:
                    print('=====================================================')
                    head_df = sorted([pair for pair in table_dict[0].items()], key = lambda x:x[0])
                    head_df = [pair[1] for pair in head_df]
                    # print(table_dict.values())
                    data_df = [row.items() for row in list(table_dict.values())[1:] ]
                    new_data_df = []
                    for r in data_df:
                        new_data_df.append([pair[1] for pair in sorted(r, key=lambda x:x[0])])#sort by columns to make sure they are in right order
                    df_tmp = pd.DataFrame(new_data_df, columns = head_df)
                    try:
                        label_pred = extract_dingzeng(r)
                    except:
                        print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        print(path_html)
                        print()
                    # print(df_tmp)
                    # print(label_pred)

                    # for row_idx in table_dict.keys():
                    #     col_dict = table_dict[row_idx]
                    #     # print('\t'.join(col_dict.values()))
                    #     print(len(col_dict))

                    break

            continue
            handle_file = codecs.open(path_html, mode='r', encoding='utf8')
            txt_html = handle_file.read()
            txt = html2text.html2text(txt_html)

            #match labels to txt
            for idx, labels in df_labels_cur.iterrows():
                target = labels['target']
                issue_mode = labels['issue_mode']
                share = labels['share']
                money = labels['money']
                lockup = labels['lockup']
                buy_mode = labels['buy_mode']
                dict_label_idx = locate_labels_idx(labels, txt)
                cluster_labels_groups = get_label_clusters(dict_label_idx)
                for cluster_idxs in cluster_labels_groups:
                    txt_vis = vis_mark_label_found(txt, cluster_idxs)
                    print('++++++++++++++++++++++++++++++++++++++++ result ++++++++++++++++++++++++++++++++++++++++++++++++++')

                    # print(labels)
                    # print(txt_vis[cluster_idxs[0]-300 : cluster_idxs[0]+300])
                
            # break
        # break

    print('good table vs all table vs all files:', cnt_good_table, cnt_all_table, cnt_all_file)