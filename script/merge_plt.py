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

# %%
path = '/media/weibo/IM1/BFS_M1.7Tur/'
sid = 'Z_05'
snap = 'TP_2D_' + sid
FoldPath = path + 'snapshots/1100/'
OutPath = path + 'snapshots/' + sid + '/'
dirs = os.listdir(FoldPath)

num = np.size(dirs)
for i in range(0, num):
    path = FoldPath + dirs[i] + "/"
    file = glob(path + snap + '_*.plt')
    with timer("merge " + dirs[i] + " file"):
        dataset = tp.data.load_tecplot(file, read_data_option=2)
        SolTime = dataset.solution_times[0]
        tm = '_' + "%08.2f" % SolTime
        tp.data.save_tecplot_plt(OutPath+snap+tm+'.plt', dataset=dataset)
        # tp.data.save_tecplot_szl(path + snap + '.szplt', dataset=dataset)
