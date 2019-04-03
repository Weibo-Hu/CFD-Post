#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:22:31 2018
    This code for plotting line/curve figures

@author: weibo
"""
#%% Load necessary module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d, splev, splrep
import scipy.optimize as opt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
import copy
from DataPost import DataPost
import FlowVar as fv
from timer import timer
import os
import sys

plt.close("All")
plt.rc('text', usetex=True)
font = {
    'family': 'Times New Roman', #'color' : 'k',
    'weight': 'normal',
    'size': 'large'
}
font1 = {
    'family': 'Times New Roman', #'color' : 'k',
    'weight': 'normal',
    'size': 'medium'
}
path = "/media/weibo/Data1/BFS_M1.7L_0505/TimeAve/"
path1 = "/media/weibo/Data1/BFS_M1.7L_0505/probes/"
path2 = "/media/weibo/Data1/BFS_M1.7L_0505/temp/"
path3 = "/media/weibo/Data1/BFS_M1.7L_0505/SpanAve/"
path4 = "/media/weibo/Data1/BFS_M1.7L_0505/MeanFlow/"

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
textsize = 18
numsize = 15
matplotlib.rc('font', size=textsize)
#%% Load Data
VarName = [
    'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'T', 'uu', 'uv', 'uw', 'vv', 'vw',
    'ww', 'Q-criterion', 'L2-criterion', 'gradp'
]
MeanFlow = DataPost()
MeanFlow.UserData(VarName, path4+'MeanFlow.dat', 1, Sep='\t')
MeanFlow.AddWallDist(3.0)


#%% Temporal shock position
InFolder = "/media/weibo/Data1/BFS_M1.7L_0505/Snapshots/Snapshots2/"
OutFolder = "/media/weibo/Data1/BFS_M1.7L_0505/DataPost/Data/"
timezone = np.arange(600, 999.50 + 0.5, 0.5)
fv.ShockFoot(InFolder, OutFolder, timezone, -1.875, 0.7)





