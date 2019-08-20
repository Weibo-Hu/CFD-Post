#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:48:11 2019
    This code for obating fluctuations flow without spanwise-averaged value

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
path = "/media/weibo/VID1/BFS_M1.7TS/K-H/"
path1 = path + "TP_data_01405528/"
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

df, time = p2p.ReadINCAResults(path1, VarList)
df.to_hdf(path2 + "TP_data_" + str(time) + ".h5", 'w', format='fixed')
# df = pd.read_hdf(path2 + "TP_data_901.h5")
df2 = df.groupby(['x', 'y']).transform(lambda x: (x-x.mean()))
df1 = pd.concat([df[['x', 'y']], df2], axis=1)
df1.to_hdf(path2 + 'ZFluctuation_' + str(time) + '.h5', 'w', format='fixed')

cube = [(-10.0, 20.0), (-3.0, 30.0), (-8, 8)]
df_zone = p2p.extract_zone(path1, cube, filename=path + "ZoneInfo1.dat")
p2p.save_tec_index(df, df_zone, filename=path + "ReadList1.dat")
# %% Save fluctuations flow as tecplot .dat
varname = [
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
df_zone = pd.read_csv(path2 + "VortexList.dat", sep=' ', skiprows=0,
                      index_col=False, skipinitialspace=True)
zflc = pd.read_hdf(path2 + "ZFluctuation_" + str(time) + ".h5")[VarList]

path3 = path2 + str(time) + '/'
if (os.path.isdir(path3) is False):
    os.mkdir(path3)
for i in range(np.shape(df_zone)[0]):
    file = df_zone.iloc[i]
    cube = zflc.query(
        "x>={0} & x<={1} & y>={2} & y<={3}".format(
            file['x1'],
            file['x2'],
            file['y1'],
            file['y2']
        )
    )
    if (file['nz'] != len(np.unique(cube['z']))):
        # remove dismatch grid point on the z boundary of the block
        # since boundary grid may be finer
        zlist = np.linspace(file['z1'], file['z2'], int(file['nz']))
        blk1 = cube[cube['z'].isin(zlist)].reset_index(drop=True)
    else:
        blk1 = cube
    if (file['nx'] != len(np.unique(blk1['x']))):
        # remove dismatch grid point on the x boundary of the block
        xlist = np.unique(blk1['x'])[0::2]
        blk1 = blk1[blk1['x'].isin(xlist)]
    if (file['ny'] != len(np.unique(blk1['y']))):
        # remove dismatch grid point on the y boundary of the block
        ylist = np.unique(blk1['y'])[0::2]
        blk1 = blk1[blk1['y'].isin(ylist)]
    p2p.frame2tec3d(blk1, path3, 'Zfluc' + str(i), zname=i, stime=time)

# %% convert .dat to .plt
filelist = glob(path3 + '*.dat')
dataset = tp.data.load_tecplot(filelist, read_data_option=2)
tp.data.save_tecplot_plt(path2 + "zfluc" + str(time) + '.plt', dataset=dataset)
