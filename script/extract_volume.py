#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    Extract a uniform volume for 3D analysis

@author: weibo
"""
# %% Load necessary module
import plt2pandas as p2p
import pandas as pd
import numpy as np
import tecplot as tp
from glob import glob

# %% Extract 3D flow domain
path = "/media/weibo/IM1/BFS_M1.7Tur/"
pathin = path + "TP_data_02087427/"
pathout = path
VarList = [
    'x',
    'y',
    'z',
    'u',
    'v',
    'w',
    'p',
    'rho',
    'vorticity_1',
    'vorticity_2',
    'vorticity_3',
    'L2-criterion'
    'T',
]

volume = [(-10.0, 30.0), (-3.0, 2.0), (-8, 8)]
dxyz = [0.25, 0.125, 0.125]
df0, time = p2p.ReadINCAResults(pathin, VarList, SubZone=volume)
xval = np.arange(volume[0][0], volume[0][1] + dxyz[0], dxyz[0])
yval = np.arange(volume[1][0], volume[1][1] + dxyz[1], dxyz[1])
zval = np.arange(volume[2][0], volume[2][1] + dxyz[2], dxyz[2])

df1 = df0[df0.x.isin(xval)]
df2 = df1[df1.y.isin(yval)]
df3 = df2[df2.z.isin(zval)]

time = np.around(time, decimals=2)
filename = "TP_data_" + str(time)
df3.to_hdf(pathout + filename + ".h5", 'w', format='fixed')
# %% save to tecplot format
# in front of the step
df = df3.query("x<=0.0 & y>=0.0")
p2p.frame2tec3d(df, pathout, filename + 'A', zname=1, stime=time)
p2p.tec2plt(pathout, filename + 'A')
# behind the step
df = df3.query("x>=0.0")
p2p.frame2tec3d(df, pathout, filename + 'B', zname=2, stime=time)
p2p.tec2plt(pathout, filename + 'B')

