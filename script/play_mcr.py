#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:39:40 2019
    This code uses pytecplot to plot 3D figures, like isosurface

@author: weibo
"""
# %% Load libraries
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
import os
import sys
import numpy as np
from glob import glob

# %% data path settings
path = "/media/weibo/VID1/BFS_M1.7L/"
pathP = path + "probes/"
pathF = path + "Figures/"
pathM = path + "MeanFlow/"
pathS = path + "SpanAve/"
pathT = path + "TimeAve/"
pathI = path + "Instant/"
pathV = path + 'video/'
pathD = path + 'DMD/'

# %% load data
# run this script with '-c' to connect to tecplot on port 7600
# if '-c' in sys.argv:
#     tp.session.connect() 
tp.session.connect()
FileId = pd.read_csv(path + "ReadList.dat", sep='\t')
filelist = FileId['name'].values
datafile = [os.path.join(path + 'TP_data_01405908/', name) for name in filelist]
  
dataset = tp.data.load_tecplot(datafile, read_data_option=2)
tp.macro.execute_file(path + 'test.mcr')

tp.export.save_jpeg(path + 'test.jpg', width=4096, supersample=3, quality=100) 
