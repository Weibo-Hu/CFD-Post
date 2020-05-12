#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11  22:39:40 2020
    This code uses pytecplot to make videos

@author: weibo
"""
# %% load libraries
import tecplot as tp
from tecplot.constant import *
from tecplot.exception import *
import pandas as pd
import os
import sys
import numpy as np
from glob import glob

path = "D:/ownCloud/5Presentation/May06-2020/0p023/"


# %% load data 
filelist = ['[0.023]DMD000A.plt', '[0.023]DMD000B.plt']
datafile = [os.path.join(path, name) for name in filelist]
dataset = tp.data.load_tecplot(datafile, read_data_option=2)
SolTime = dataset.solution_times[0]

# %% frame operation
frame = tp.active_frame()
frame.load_stylesheet(path + 'video.sty')
# turn off orange zone bounding box
tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')
# frame setting
frame.width = 13.5
frame.height = 8
frame.position = (-1.2, 0.25)
frame.plot().use_lighting_effect = False
plot = frame.plot(PlotType.Cartesian3D)

# 3d view settings
view = plot.view
view.magnification = 1.1
# plot.view.fit_to_nice()
view.rotation_origin = (10, 0.0, 0.0)
view.psi = 45
view.theta = 145
view.alpha = -140
view.position = (-46, 76, 94)
# view.distance = 300
view.width = 38
# export figs
tp.export.save_png(path + 'test' + str(SolTime) + '.png', width=4096)
tp.export.save_jpeg(path + 'test' + str(SolTime) + '.jpg', width=4096, quality=100) 
