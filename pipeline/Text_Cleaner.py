# -*- coding: utf-8 -*-
'''
Created on 6/12c/2018
label doc from Boson NLP
@author: WD
'''
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

import locale
# locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )
# locale.setlocale( locale.LC_ALL, 'en_US' )
locale.setlocale( locale.LC_ALL, 'English_United States.1252' )

CLUSTER_RADIU = 200

def is_nan(x):
    if x is None or x in ('', 'None', [], np.nan, 'NaN'):
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

def get_relative_editdistance(t1, t2):
    if t1 is None or t2 is None or t1 == '' or t2 == '':
        return 1
    else:
        return editdistance.eval(t1, t2) * 1.0 / max(len(t1), len(t2))

def clean_dingzeng_labels(df):
    df['lockup'] = df['lockup'].fillna(0).astype(int)
    return df

def FullToHalf(s):
    n = []
    # s = s.decode('utf-8')
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)

def clean_width(txt):
    lines = txt.splitlines()
    lines_new = []
    for line in lines:
        line_half = FullToHalf(line)
        lines_new.append(line_half)
    # print(lines_new)
    return '\n'.join(lines_new)

def clean_locale_digit(txt):
    lines = txt.splitlines()
    lines_new = []
    for line in lines:
        line_tmp = line
        rs = re.search(r'\d+,\d+', line_tmp)
        while rs != None:
            # print('----------------------------')
            # print(line_tmp)
            digit_clean = locale.atof(rs.group())
            line_tmp = line_tmp[:rs.start()] + str(int(digit_clean)) + line_tmp[rs.end():]
            rs = re.search(r'\d+,\d+', line_tmp)
            print(line_tmp)
        lines_new.append(line_tmp)
    # print(lines_new)
    return '\n'.join(lines_new)

# must run after removing comma
def clean_currency_unit(txt):
    lines = txt.splitlines()
    lines_new = []
    for line in lines:
        line_tmp = line
        rs = re.search(r'\d+\.*\d*\s*万', line_tmp)
        while rs != None:
            print(rs.start(), rs.end(), rs.group())
            print('----------------------------')
            print(line_tmp)
            if '万' in rs.group():
                digitstr = rs.group().replace('万','')
                digit = int(float(digitstr)*10000)
            elif '亿' in rs.group():
                digitstr = rs.group().replace('亿','')
                digit = int(float(digitstr)*100000000)
            line_tmp = line_tmp[:rs.start()] + str(digit) + line_tmp[rs.end():]
            rs = re.search(r'\d+\.*\d*\s*万', line_tmp)
            print(line_tmp)
        lines_new.append(line_tmp)
    # print(lines_new)
    return '\n'.join(lines_new)

def transfer_date(txt):
    lines = txt.splitlines()
    lines_new = []
    for line in lines:
        line_tmp = line
        rs = re.search(r'\d+年\d+月\d+日', line_tmp)
        while rs != None:
            print(rs.start(), rs.end(), rs.group())
            print('----------------------------')
            print(line_tmp)
            if '万' in rs.group():
                digitstr = rs.group().replace('万', '')
                digit = int(float(digitstr) * 10000)
            elif '亿' in rs.group():
                digitstr = rs.group().replace('亿', '')
                digit = int(float(digitstr) * 100000000)
            line_tmp = line_tmp[:rs.start()] + str(digit) + line_tmp[rs.end():]
            rs = re.search(r'\d+\.*\d*\s*万', line_tmp)
            print(line_tmp)
        lines_new.append(line_tmp)
    # print(lines_new)
    return '\n'.join(lines_new)


UTIL_CN_NUM = {
    u'零': 0,
    u'一': 1,
    u'二': 2,
    u'两': 2,
    u'三': 3,
    u'四': 4,
    u'五': 5,
    u'六': 6,
    u'七': 7,
    u'八': 8,
    u'九': 9,
}
UTIL_CN_UNIT = {
    u'十': 10,
    u'百': 100,
    u'千': 1000,
    u'万': 10000,
}


def cn2dig(src):
    if src == "":
        return None
    m = re.match("\d+", src)
    if m:
        return m.group(0)
    rsl = 0
    unit = 1
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            return None
    if rsl < unit:
        rsl += unit
    return str(rsl)


def parse_datetime(msg):
    if msg is None or len(msg) == 0:
        return None
    m = re.match(r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",
        msg)
    if m.group(0) is not None:
        res = {
            "year": m.group(1),
            "month": m.group(2),
            "day": m.group(3),
            "hour": m.group(5) if m.group(5) is not None else '00',
            "minute": m.group(6) if m.group(6) is not None else '00',
            "second": m.group(7) if m.group(7) is not None else '00',
            # "microsecond": '00',
        }
        params = {}
        for name in res:
            if res[name] is not None and len(res[name]) != 0:
                params[name] = int(cn2dig(res[name][:-1]))
        target_date = datetime.datetime.today().replace(**params)
        is_pm = m.group(4)
        if is_pm is not None:
            if is_pm == u'下午' or is_pm == u'晚上':
                hour = target_date.time().hour
                if hour < 12:
                    target_date = target_date.replace(hour=hour + 12)
        return target_date
    else:
        return None

def capture_date(txt):
    # rs = re.search(r'(\d+年)?(\d+月)?(\d+日|号)?', txt)
    rs = re.search(u'(\d+年)?(\d+月)?((\d+)(日|号))', txt)
    if rs is not None:
        rs_str = rs.group()
        return rs_str
    else:
        return None

if __name__ == '__main__':

    if os.path.isdir("C:\\projects\\fddc\\"):
        path_proj_folder = "C:\\projects\\fddc\\"
    else:
        path_proj_folder = "F:\\fddc\\"

    path_dir_pdf_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\pdf"
    path_dir_html_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\html"
    path_label_dz_train = path_proj_folder + "data_official\\round1_train_20180518\\dingzeng\\dingzeng.train"
    path_dir_output_dz = os.path.join(path_proj_folder + "data\\clean_file\\dz\\")
    print('++++++++++++++++++++++++++++++++++++++++ start running ++++++++++++++++++++++++++++++++++++++++++++++++++')
    s = open(path_label_dz_train, mode='r', encoding='utf-8-sig').readlines()

    my_cols = ["id_doc", "target", "issue_mode", "share", "money", "lockup", "buy_mode"]
    df_label_dz = pd.read_csv(path_label_dz_train,
                              names=my_cols,
                              sep='\t',
                              # encoding = 'utf-8-sig',
                              # header=None,
                              # engine='python',
                              # error_bad_lines=False
                              )
    df_label_dz = clean_dingzeng_labels(df_label_dz)
    # print(df_label_dz.head(5))
    print('Training label size', df_label_dz.shape)

    cnt_files_processed = 0
    cnt_all_table = 0
    # scan each trainin label
    report_files = []  # format[ [doc_id, path_file],[doc_id, path_file] ...]
    for root, dirs, files in os.walk(path_dir_html_dz_train):
        for file in files:
            doc_id = file.split('.')[0]
            path_html = os.path.join(root, file)
            report_files.append([doc_id, path_html])

    df_output = pd.DataFrame()

    txt = "注册资本：417,352 万元公司类型：国有独资"
    txt = "5000 万股股票,本次非公开发行股票的2018年11月03号发行对象前海基金以现金方式认购本次非公开发行的 5,000 万股股票。"
    txt_clean = clean_width(txt)  # 全角半角
    txt_clean = clean_locale_digit(txt_clean)  # remove comma
    txt_clean = clean_currency_unit(txt_clean)  # currency unit

    # txt_clean = transfer_date(txt_clean)

    dt_str = capture_date(txt_clean)
    print(dt_str)
    sys.exit()
    dt = parse_datetime(dt_str)
    print(dt)
    sys.exit()


    for doc_id, path_html in report_files:
        cnt_files_processed += 1
        if cnt_files_processed % 100 == 0:
            print('processed file:', cnt_files_processed)
        with codecs.open(path_html, encoding='utf-8', mode='r') as fp:
            txt = fp.read()
            # print(txt)
            txt_clean = clean_width(txt)  # 全角半角
            txt_clean = clean_locale_digit(txt_clean) # remove comma
            txt_clean = clean_currency_unit(txt_clean)#currency unit
            # file = codecs.open(os.path.join(path_dir_output_dz, doc_id+'_clean.html'), "w", "utf-8")
            # file.write(txt_clean)
            # file.close()