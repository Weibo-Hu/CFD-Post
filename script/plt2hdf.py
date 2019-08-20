#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for convert plt to hdf

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
    'Q-criterion',
    'L2-criterion',
    '|grad(rho)|',
    'T',
    '|gradp|'
]
equ = '{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)'
FoldPath = "/media/weibo/VID1/BFS_M1.7TS/Slice/TP_2D_S_10/"
OutFolder = "/media/weibo/VID2/BFS_M1.7TS/Slice/TP_2D_S_10/"
subzone = [(-40.0, 70.0), (-3.0, 10.0)]
dirs = os.scandir(FoldPath)
for folder in dirs:
    file = FoldPath + folder.name
    outfile = os.path.splitext(folder.name)[0]
    with timer("Read " + folder.name + " data"):
        DataFrame, time = \
        p2p.ReadINCAResults(FoldPath, VarList, FileName=file, SubZone=subzone,
                            SavePath=OutFolder, OutFile=outfile, Equ=equ)
