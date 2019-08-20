#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for obating fluctuations flow without base flow

@author: weibo
"""


# %% Load necessary module
import plt2pandas as p2p
import pandas as pd
import numpy as np
import os
import tecplot as tp
from glob import glob

# %% Extract spanwise-averaged flow, obtain fluctuations
path2 = "/media/weibo/VID2/BFS_M1.7TS/"
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
    'T',
]

time = 904.0
# path2 = path + "TP_data_baseflow/"
# base = p2p.ReadAllINCAResults(path2, path, SpanAve=False, OutFile='baseflow')
# df_zone = p2p.save_zone_info(path1, filename=path + "ZoneInfo.dat")
orig = pd.read_hdf(path2 + "TP_data_" + str(time) + ".h5")[VarList]
base = pd.read_hdf(path2 + "baseflow.h5")[VarList]
df_zone = pd.read_csv(path2 + "VortexList.dat", sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
print("Load data... Done!")
path3 = path2 + str(time) + '/'
if (os.path.isdir(path3) is False):
    os.mkdir(path3)

# %% convert .h5 to tecplot .dat
for i in range(np.shape(df_zone)[0]):
    file = df_zone.iloc[i]
    cube = orig.query(
        "x>={0} & x<={1} & y>={2} & y<={3}".format(
            file['x1'],
            file['x2'],
            file['y1'],
            file['y2']
        )
    )

    if (file['nz'] != len(np.unique(cube['z']))):
        # remove dismatch grid point on the z boundary of the block
        # since the boundary grid may be finer
        zlist = np.linspace(file['z1'], file['z2'], int(file['nz']))
        blk1 = cube[cube['z'].isin(zlist)].reset_index(drop=True)
    else:
        blk1 = cube

    blk0 = base.query(
        "x>={0} & x<={1} & y>={2} & y<={3}".format(
            file['x1'],
            file['x2'],
            file['y1'],
            file['y2']
        )
    )
    new = blk0.loc[blk0.index.repeat(int(file['nz']))]
    new = new.reset_index(drop=True)
    new = new.sort_values(by=['x', 'y', 'z'])
    flc = blk1.sort_values(by=['x', 'y', 'z']).reset_index(drop=True)
    flc[VarList[3:]] = flc[VarList[3:]] - new[VarList[3:]]
    # flc.update(flc[varname] - new[varname])
    if (file['nx'] != len(np.unique(flc['x']))):
        # remove dismatch grid point on the x boundary of the block
        xlist = np.unique(flc['x'])[0::2]
        flc = flc[flc['x'].isin(xlist)]
    if (file['ny'] != len(np.unique(flc['y']))):
        # remove dismatch grid point on the y boundary of the block
        ylist = np.unique(flc['y'])[0::2]
        flc = flc[flc['y'].isin(ylist)]
    p2p.frame2tec3d(flc, path3, 'fluc_base' + str(i), zname=i, stime=time)

# %% convert .dat to .plt
filelist = glob(path3 + '*.dat')
dataset = tp.data.load_tecplot(filelist, read_data_option=2)
tp.data.save_tecplot_plt(
    path2 + "fluc_base" + str(time) + '.plt',
    dataset=dataset
)
