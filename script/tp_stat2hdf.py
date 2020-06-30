#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for convert tp_stat(.plt) to meanflow(.h5)

@author: weibo
"""


# %% Load necessary module
import plt2pandas as p2p
import variable_analysis as va
import numpy as np
import pandas as pd
# import modin.pandas as pd
from glob import glob

# %%
path = '/media/weibo/IM2/FFS_M1.7ZA2/'

p2p.create_folder(path)
pathTP = path + 'TP_stat/'
pathM = path + 'MeanFlow/'
pathT = path + 'TimeAve/'
dirs = glob(pathTP + '*plt')
equ = ['{|gradp|}=sqrt(ddx({<p>})**2+ddy({<p>})**2+ddz({<p>})**2)']
varlist, equ = va.mean_var(opt='gradient')

num = np.size(dirs)
a1 = int(num / 4)
a2 = a1 * 2
a3 = a1 * 3
ind = [[0, a1],
       [a1, a2],
       [a2, a3],
       [a3, num]]
for i in range(4):
    print('finish part ' + str(i))
    FileList = dirs[ind[i][0]:ind[i][1]]
    df = p2p.ReadAllINCAResults(pathTP,
                                pathM,
                                FileName=FileList,
                                Equ=equ,
                                OutFile='MeanFlow_' + str(i))

dir1 = glob(pathM + 'MeanFlow_*')
df0 = pd.read_hdf(dir1[0])
grouped = df0.groupby(['x', 'y'])
mean0 = grouped.mean().reset_index()
del df0

df1 = pd.read_hdf(dir1[1])
grouped = df1.groupby(['x', 'y'])
mean1 = grouped.mean().reset_index()
del df1

df2 = pd.read_hdf(dir1[2])
grouped = df2.groupby(['x', 'y'])
mean2 = grouped.mean().reset_index()
del df2

df3 = pd.read_hdf(dir1[3])
grouped = df3.groupby(['x', 'y'])
mean3 = grouped.mean().reset_index()
del df3

df = pd.concat([mean0, mean1, mean2, mean3], ignore_index=True)
# df = df.drop_duplicates(keep='last')
# save time- and spanwise averaged flow field
grouped = df.groupby(['x', 'y'])
mean = grouped.mean().reset_index()
mean.to_hdf(pathM + 'MeanFlow.h5', "w", format="fixed")
# save time-averaged flow field
# new = df.loc[(df['x']>=-40.0) & (df['x']<=20.0) & (df['y']<=2.0)]
# new.to_hdf(pathT + 'TimeAve.h5', "w", format="fixed")
