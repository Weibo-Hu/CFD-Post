#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for convert plt to hdf for vorticity transportation equation

@author: weibo
"""


# %% Load necessary module
import os
from timer import timer
import tecplot as tp
import plt2pandas as p2p
import numpy as np
import pandas as pd
import sys
from glob import glob


# %% Convert .plt to .h5
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
#    'Q-criterion',
    'L2-criterion',
    'T',
    '|gradp|',
    'ux',
    'uy',
    'uz',
    'vx',
    'vy',
    'vz',
    'wx',
    'wy',
    'wz',
    'rhox',
    'rhoy',
    'rhoz',
    'px',
    'py',
    'pz'
]
equ = [
    '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)',
    '{ux} = ddx({u})',
    '{uy} = ddy({u})',
    '{uz} = ddz({u})',
    '{vx} = ddx({v})',
    '{vy} = ddy({v})',
    '{vz} = ddz({v})',
    '{wx} = ddx({w})',
    '{wy} = ddy({w})',
    '{wz} = ddz({w})',
    '{rhox} = ddx({rho})',
    '{rhoy} = ddy({rho})',
    '{rhoz} = ddz({rho})',
    '{px} = ddx({p})',
    '{py} = ddy({p})',
    '{pz} = ddz({p})',
]
FoldPath = "/media/weibo/VID2/BFS_M1.7TS1/bubble/"  # TS/Slice/TP_2D_S_10/"
OutFolder = "/media/weibo/VID2/BFS_M1.7TS1/"  # TS/Slice/TP_2D_S_10/"
subzone = [(-40.0, 30.0), (-3.0, 2.0)]
# subzone = [(-10.0, 30.0), (-3.0, 1.0)]
dirs = os.listdir(FoldPath)
for i in range(np.size(dirs)):
    # files = pd.read_csv(OutFolder + 'ZoomZone1.dat', header=None)[0]
    # infiles = [os.path.join(filedir, name) for name in files]
    filedir = FoldPath + dirs[i] + '/'
    outfile = 'TP_data'
    with timer("Read " + dirs[i]):
        DataFrame, time = \
        p2p.ReadINCAResults(filedir, VarList, SubZone=subzone, Equ=equ,
                            SavePath=OutFolder, OutFile=outfile)

# p2p.ReadINCAResults(FoldPath, VarList, FileName=FoldPath + 'TP_912.plt',
#                     Equ=equ, SavePath=OutFolder, OutFile='TP_912')
