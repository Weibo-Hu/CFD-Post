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
    '|gradp|',
    #    'ux',
    #    'uy',
    #    'uz',
    #    'vx',
    #    'vy',
    #    'vz',
    #    'wx',
    #    'wy',
    #    'wz',
    #    'rhox',
    #    'rhoy',
    #    'rhoz',
    #    'px',
    #    'py',
    #    'pz'
]
sp = "Z_003"
equ = ['{|gradp|}=sqrt(ddx({p})**2+ddy({p})**2)']
# FoldPath = "/media/weibo/VID1/BFS_M1.7TS/Slice/" + sp + "/"
path = "/media/weibo/IM2/FFS_M1.7TB/Slice/"
InPath = path + 'TP_2D_' + sp + '/'
OutPath = "/media/weibo/IM2/FFS_M1.7TB/Slice/" + sp + "/"
# subzone = [(-10.0, 30.0), (-3.0, 30.0), (-8.0, 8.0)]
if not os.path.exists(OutPath):
    os.mkdir(OutPath)

subzone = [(-70.0, 40.0), (0.0, 12.0), (-8.0, 8.0)]  # for 2D snapshots
dirs = os.scandir(InPath)
for folder in dirs:
    flnm = 'TP_2D_' + sp
    file = InPath + folder.name + '/' + flnm + '.szplt'
    # file = FoldPath + folder.name + '/TP_2D_' + sp + '.szplt'
    # outfile = os.path.splitext(folder.name)[0]
    
    with timer("Read " + folder.name + " data"):
        df, st = \
            p2p.ReadINCAResults(InPath, VarList, SubZone=subzone,
                                FileName=file, Equ=equ, opt=2)
    stime = "%08.2f" % st
    OutFile = flnm + '_' + stime + '.h5'
    df.to_hdf(OutPath + OutFile, 'w', format='fixed')
# p2p.ReadINCAResults(FoldPath, VarList, FileName=FoldPath + 'TP_912.plt',
#                     Equ=equ, SavePath=OutFolder, OutFile='TP_912')
# p2p.ReadAllINCAResults(FoldPath, FileName=FoldPath + 'iso_z0.plt',
#                        SavePath=OutFolder, OutFile='iso_z0')


# FoldPath = "/media/weibo/VID1/BFS_M1.7L/bubble/"
# skip = 0
# cube = [(8.0, 12.5), (-2.8, 0.0), (-8, 8)]
# FileId = p2p.extract_zone(FoldPath + 'TP_data_01402108/', cube, skip=skip)
# FileId.to_csv(FoldPath+str(skip)+'VortexListStage4.dat', index=False, sep=' ')
# FileId['name'].to_csv(FoldPath + 'ZoomZoneStage4.dat', index=False)
