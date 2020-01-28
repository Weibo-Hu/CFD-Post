#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for merge multiple .plt into a single one

@author: weibo
"""
# %% Load necessary module
import os
from timer import timer
import tecplot as tp
import numpy as np
from glob import glob
import sys

# %%
path = '/media/weibo/VID2/BFS_M1.7TS1_HA/'
# splist = ['X_11', 'X_12', 'X_13', 'X_14']
# splist = ['Y_07', 'Y_08', 'Y_09', 'S_10']
# splist = ['Z_01', 'Z_02', 'Z_03', 'Z_04', 'Z_05']

# splist = ['X_011', 'X_012', 'X_013', 'X_014']
# splist = ['Y_007', 'Y_008', 'Y_009', 'S_010']
splist = ['Z_003', 'Z_004', 'Z_005', 'Z_006']  # 

InPath = path + 'snapshots/700/'
dirs = os.listdir(InPath)
num = np.size(dirs)

for j in range(np.size(splist)):
    sid = splist[j]
    snap = 'TP_2D_' + sid
    if not os.path.exists(path + 'snapshots/' + sid):
        os.mkdir(path + 'snapshots/' + sid)
    OutPath = path + 'snapshots/' + sid + '/'
    print('For ' + sid)
    for i in range(0, num):
        SubPath = InPath + dirs[i] + "/"
        file = glob(SubPath + snap + '*plt')
        extension = os.path.splitext(file[0])[1]
        with timer("merge " + dirs[i] + " file"):
            if extension == '.plt':
                dataset = tp.data.load_tecplot(file, read_data_option=2)
            elif extension == '.szplt':
                dataset = tp.data.load_tecplot_szl(file, read_data_option=2)
            else:
                sys.exit("File type is not exist!!!")
            num_st = dataset.num_solution_times
            SolTime = dataset.solution_times[0]
            tm = '_' + "%08.2f" % SolTime
            tp.data.save_tecplot_plt(OutPath+snap+tm+'.plt', dataset=dataset)
            # tp.data.save_tecplot_szl(path + snap + '.szplt', dataset=dataset)
