import csv

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolAlign, rdFMCS, QED, Crippen, Lipinski
from analysis.SA_Score.sascorer import calculateScore

import pandas as pd
import os
from datetime import datetime

# 返回每个文件下生成的文件个数
def read_sdf(path):
    num = 0
    for root, dirs, files in os.walk(path_new):
        # print(files)
        for i in files:
            if i.split('.')[1] == 'sdf':
                num = num + 1
    return num
# 返回总共使用时间
def read_txt(path):
    with open(path, "r", encoding="gbk") as f:
        r = f.readlines()
        time_start = r[0].split(',')[0].split('[')[1]
        time_start = datetime.strptime(time_start,"%Y-%m-%d %H:%M:%S")
        time_end = r[-1].split(',')[0].split('[')[1]
        time_end = datetime.strptime(time_end,"%Y-%m-%d %H:%M:%S")
        time = time_end - time_start
        return t2s(time)
def t2s(t):
    t = str(t)
    h,m,s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)
path_new = r'/home/linux/DiffFBDD/baseline/3DSBDD/outputs'
file_new = []
time = 0
for root, dirs, files in os.walk(path_new):
    for dir in dirs:
        if dir.split('-')[0]== 'sample':
            path = path_new +'/'+ dir
            for r,d,f in os.walk(path):
                for i in f:
                    if i == 'log.txt':
                        path_time = path+'/'+i
                        time += read_txt(path_time)
                for j in d:
                    if j == 'SDF':
                        path_sum = path + '/' + 'SDF'
                        print(path_sum)
                        sum = read_sdf(path_sum)
print(time)
print(time/sum)

# sum = 0
# length = 0
# for i in file_new:
#     with open(i) as f:
#         content = f.read()
#         sum += float(content.split(' ')[1])
#         length = length + 1
# print(sum/length)
