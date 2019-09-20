#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for convert tp_stat(.plt) to meanflow(.h5)

@author: weibo
"""


# %% Load necessary module
import os
from timer import timer
import plt2pandas as p2p
import numpy as np
import pandas as pd
from glob import glob

# %%
path = '/media/weibo/VID2/BFS_M1.7TS/'
pathTP = path + 'TP_stat/'
pathM = path + 'MeanFlow/'
pathT = path + 'TimeAve/'
dirs = glob(pathTP + '*.plt')
equ = ['{|gradp|}=sqrt(ddx({<p>})**2+ddy({<p>})**2+ddz({<p>})**2)']

num = np.size(dirs)
a1 = int(num / 4)
a2 = a1 * 2
a3 = a1 * 3
ind = [[0, a1],
       [a1 + 1, a2],
       [a2 + 1, a3],
       [a3 + 1, num - 1]]
for i in range(4):
    FileList = dirs[ind[i][0]:ind[i][1]]
    df = p2p.ReadAllINCAResults(pathTP,
                                pathT,
                                FileName=FileList,
                                Equ=equ,
                                OutFile='MeanFlow' + str(i))

df0 = pd.read_hdf(pathT + 'MeanFlow0.h5')
df1 = pd.read_hdf(pathT + 'MeanFlow1.h5')
df2 = pd.read_hdf(pathT + 'MeanFlow2.h5')
df3 = pd.read_hdf(pathT + 'MeanFlow3.h5')
df = pd.concat([df0, df1, df2, df3], ignore_index=True)
# df = df.drop_duplicates(keep='last')
# save time- and spanwise averaged flow field
grouped = df.groupby(['x', 'y'])
mean = grouped.mean().reset_index()
mean.to_hdf(pathM + 'MeanFlow.h5', "w", format="fixed")
# save time-averaged flow field
new = df.loc[(df['x']>=-5.0) & (df['x']<=20.0) & (df['y']<=2.0)]
new.to_hdf(pathT + 'TimeAve.h5', "w", format="fixed")
