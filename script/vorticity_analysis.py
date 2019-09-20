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
# import tecplot as tp
# import plt2pandas as p2p
import numpy as np
import pandas as pd
import sys
from glob import glob
import variable_analysis as fv
from triaxial_field import TriField as tf


path = "/media/weibo/VID2/BFS_M1.7TS/"
pathV = path + "Vortex/"

xloc = np.arange(0.125, 30.0 + 0.125, 0.125)
tp_time = np.arange(900, 1000 + 5.0, 5.0)
y = np.linspace(-3.0, 0.0, 151)
z = np.linspace(-8.0, 8.0, 161)
ens, ens1, ens2, ens3 = np.zeros((4, np.size(xloc)))
nms = ['x', 'enstrophy', 'enstrophy_x', 'enstrophy_y', 'enstrophy_z']
tilt1, tilt2, stret, dilate, torque = np.zeros((5, np.size(xloc), 2))
nms1 = ['x', 'tilt1_p', 'tilt1_n', 'tilt2_p', 'tilt2_n', 'stretch_p', \
        'stretch_n', 'dilate_p', 'dilate_n', 'torque_p', 'torque_n']
flow = tf()
dirs = glob(pathV + '*.h5')
for j in range(np.size(dirs)):
    flow.load_3data(pathV, FileList=dirs[j], NameList='h5')
    file = os.path.basename(dirs[j])
    file = os.path.splitext(file)[0]
    # flow.copy_meanval()
    for i in range(np.size(xloc)):
        df = flow.TriData
        xslc = df.loc[df['x']==xloc[i]]
        ens[i] = fv.enstrophy(xslc, type='x', mode=None, rg1=y, rg2=z, opt=2)
        ens1[i] = fv.enstrophy(xslc, type='x', mode='x', rg1=y, rg2=z, opt=2)
        ens2[i] = fv.enstrophy(xslc, type='x', mode='y', rg1=y, rg2=z, opt=2)
        ens3[i] = fv.enstrophy(xslc, type='x', mode='z', rg1=y, rg2=z, opt=2)
        # votex term
        df1 = fv.vortex_dyna(xslc, type='x', opt=2)
        tilt1[i, :] = fv.integral_db(df1['y'], df1['z'], df1['tilt1'],
                                     range1=y, range2=z, opt=3)
        tilt2[i, :] = fv.integral_db(df1['y'], df1['z'], df1['tilt2'],
                                     range1=y, range2=z, opt=3)
        stret[i, :] = fv.integral_db(df1['y'], df1['z'], df1['stretch'],
                                       range1=y, range2=z, opt=3)
        dilate[i, :] = fv.integral_db(df1['y'], df1['z'], df1['dilate'],
                                      range1=y, range2=z, opt=3)
        torque[i, :] = fv.integral_db(df1['y'], df1['z'], df1['bar_tor'],
                                      range1=y, range2=z, opt=3)
    print("finish " + dirs[j])
    res = np.vstack((xloc, ens, ens1, ens2, ens3))
    enstrophy = pd.DataFrame(data=res.T, columns=nms)
    enstrophy.to_csv(pathV + 'Enstrophy_' + file + '.dat',
                     index=False, float_format='%1.8e', sep=' ')
    # votex term
    re1 = np.hstack((xloc.reshape(-1, 1), tilt1, tilt2, stret, dilate, torque))
    ens_z = pd.DataFrame(data=re1, columns=nms1)
    ens_z.to_csv(pathV + 'vortex_x' + file + '.dat',
                 index=False, float_format='%1.8e', sep=' ')

#%% Load data and calculate vorticity term in every direction
"""
xloc = np.arange(0.125, 30.0 + 0.125, 0.125)
tp_time = np.arange(900, 1000 + 5.0, 5.0)
y = np.linspace(-3.0, 0.0, 151)
z = np.linspace(-8.0, 8.0, 161)
tilt1, tilt2, stret, dilate, torque = np.zeros((5, np.size(xloc), 2))
nms1 = ['x', 'tilt1_p', 'tilt1_n', 'tilt2_p', 'tilt2_n', 'stretch_p', \
        'stretch_n', 'dilate_p', 'dilate_n', 'torque_p', 'torque_n']
flow = tf()
dirs = glob(pathV + '*.h5')
for j in range(np.size(dirs)):
    flow.load_3data(pathV, FileList=dirs[j], NameList='h5')
    file = os.path.basename(dirs[j])
    file = os.path.splitext(file)[0]
    for i in range(np.size(xloc)):
        df = flow.TriData
        xslc = df.loc[df['x'] == xloc[i]]
        df1 = fv.vortex_dyna(xslc, type='y', opt=2)
        tilt1[i, :] = fv.integral_db(df1['y'], df1['z'], df1['tilt1'],
                                     range1=y, range2=z, opt=3)
        tilt2[i, :] = fv.integral_db(df1['y'], df1['z'], df1['tilt2'],
                                     range1=y, range2=z, opt=3)
        stret[i, :] = fv.integral_db(df1['y'], df1['z'], df1['stretch'],
                                       range1=y, range2=z, opt=3)
        dilate[i, :] = fv.integral_db(df1['y'], df1['z'], df1['dilate'],
                                      range1=y, range2=z, opt=3)
        torque[i, :] = fv.integral_db(df1['y'], df1['z'], df1['bar_tor'],
                                      range1=y, range2=z, opt=3)
    print("finish " + dirs[j])
    re1 = np.hstack((xloc.reshape(-1, 1), tilt1, tilt2, stret, dilate, torque))
    ens_z = pd.DataFrame(data=re1, columns=nms1)
    ens_z.to_csv(pathV + 'vortex_y' + file + '.dat',
             index=False, float_format='%1.8e', sep=' ')

"""
