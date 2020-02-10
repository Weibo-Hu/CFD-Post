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

flow = tf()
dirs = glob(pathV + '*.h5')

xloc = np.arange(0.125, 30.0 + 0.125, 0.125)
tp_time = np.arange(900, 1000 + 5.0, 5.0)
y = np.linspace(-3.0, 0.0, 151)
z = np.linspace(-8.0, 8.0, 161)
Xtilt1, Xtilt2, Xstret, Xdilate, Xtorque = np.zeros((5, np.size(xloc), 2))
Ytilt1, Ytilt2, Ystret, Ydilate, Ytorque = np.zeros((5, np.size(xloc), 2))
Ztilt1, Ztilt2, Zstret, Zdilate, Ztorque = np.zeros((5, np.size(xloc), 2))
nms1 = ['x', 'tilt1_p', 'tilt1_n', 'tilt2_p', 'tilt2_n', 'stretch_p',
        'stretch_n', 'dilate_p', 'dilate_n', 'torque_p', 'torque_n']

enstro = False
x_out = False
y_out = True
z_out = True
if enstro:
    ens, ens1, ens2, ens3 = np.zeros((4, np.size(xloc)))
    nms = ['x', 'enstrophy', 'enstrophy_x', 'enstrophy_y', 'enstrophy_z']

for j in range(np.size(dirs)):
    flow.load_3data(pathV, FileList=dirs[j], NameList='h5')
    file = os.path.basename(dirs[j])
    file = os.path.splitext(file)[0]
    outfile = file.replace("TP_data_", "_")
    # flow.copy_meanval()
    for i in range(np.size(xloc)):
        df = flow.TriData
        xslc = df.loc[df['x'] == xloc[i]]
        if enstro:
            ens[i]  = fv.enstrophy(xslc, type='x', mode=None,
                                   rg1=y, rg2=z, opt=2)
            ens1[i] = fv.enstrophy(xslc, type='x', mode='x',
                                   rg1=y, rg2=z, opt=2)
            ens2[i] = fv.enstrophy(xslc, type='x', mode='y',
                                   rg1=y, rg2=z, opt=2)
            ens3[i] = fv.enstrophy(xslc, type='x', mode='z',
                                   rg1=y, rg2=z, opt=2)
        # votex term
        if x_out:
            df1 = fv.vortex_dyna(xslc, type='x', opt=2)
            Xtilt1[i, :]  = fv.integral_db(df1['y'], df1['z'], df1['tilt1'],
                                          range1=y, range2=z, opt=3)
            Xtilt2[i, :]  = fv.integral_db(df1['y'], df1['z'], df1['tilt2'],
                                          range1=y, range2=z, opt=3)
            Xstret[i, :]  = fv.integral_db(df1['y'], df1['z'], df1['stretch'],
                                          range1=y, range2=z, opt=3)
            Xdilate[i, :] = fv.integral_db(df1['y'], df1['z'], df1['dilate'],
                                          range1=y, range2=z, opt=3)
            Xtorque[i, :] = fv.integral_db(df1['y'], df1['z'], df1['bar_tor'],
                                          range1=y, range2=z, opt=3)
        if y_out:
            df2 = fv.vortex_dyna(xslc, type='y', opt=2)
            Ytilt1[i, :]  = fv.integral_db(df2['y'], df2['z'], df2['tilt1'],
                                          range1=y, range2=z, opt=3)
            Ytilt2[i, :]  = fv.integral_db(df2['y'], df2['z'], df2['tilt2'],
                                          range1=y, range2=z, opt=3)
            Ystret[i, :]  = fv.integral_db(df2['y'], df2['z'], df2['stretch'],
                                          range1=y, range2=z, opt=3)
            Ydilate[i, :] = fv.integral_db(df2['y'], df2['z'], df2['dilate'],
                                          range1=y, range2=z, opt=3)
            Ytorque[i, :] = fv.integral_db(df2['y'], df2['z'], df2['bar_tor'],
                                          range1=y, range2=z, opt=3)
        if z_out:
            df3 = fv.vortex_dyna(xslc, type='z', opt=2)
            Ztilt1[i, :]  = fv.integral_db(df3['y'], df3['z'], df3['tilt1'],
                                           range1=y, range2=z, opt=3)
            Ztilt2[i, :]  = fv.integral_db(df3['y'], df3['z'], df3['tilt2'],
                                           range1=y, range2=z, opt=3)
            Zstret[i, :]  = fv.integral_db(df3['y'], df3['z'], df3['stretch'],
                                           range1=y, range2=z, opt=3)
            Zdilate[i, :] = fv.integral_db(df3['y'], df3['z'], df3['dilate'],
                                           range1=y, range2=z, opt=3)
            Ztorque[i, :] = fv.integral_db(df3['y'], df3['z'], df3['bar_tor'],
                                           range1=y, range2=z, opt=3)
    print("finish " + dirs[j])
    if enstro:
        res = np.vstack((xloc, ens, ens1, ens2, ens3))
        enstrophy = pd.DataFrame(data=res.T, columns=nms)
        enstrophy.to_csv(pathV + 'Enstrophy_' + outfile + '.dat',
                         index=False, float_format='%1.8e', sep=' ')
    # votex term
    if x_out:
        re1 = np.hstack((xloc.reshape(-1, 1), Xtilt1, Xtilt2,
                        Xstret, Xdilate, Xtorque))
        ens_x = pd.DataFrame(data=re1, columns=nms1)
        ens_x.to_csv(pathV + 'vortex_x' + outfile + '.dat',
                     index=False, float_format='%1.8e', sep=' ')

    if y_out:
        re1 = np.hstack((xloc.reshape(-1, 1), Ytilt1, Ytilt2,
                        Ystret, Ydilate, Ytorque))
        ens_y = pd.DataFrame(data=re1, columns=nms1)
        ens_y.to_csv(pathV + 'vortex_y' + outfile + '.dat',
                     index=False, float_format='%1.8e', sep=' ')

    if z_out:
        re1 = np.hstack((xloc.reshape(-1, 1), Ztilt1, Ztilt2,
                        Zstret, Zdilate, Ztorque))
        ens_z = pd.DataFrame(data=re1, columns=nms1)
        ens_z.to_csv(pathV + 'vortex_z' + outfile + '.dat',
                     index=False, float_format='%1.8e', sep=' ')

