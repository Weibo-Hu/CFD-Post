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
path = "/media/weibo/Data2/BFS_M1.7TS/"
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
filelist = FileId['name'].to_list()
datafile = [os.path.join(path + 'TP_data_00967937/', name) for name in filelist]
  
dataset = tp.data.load_tecplot(datafile)
frame = tp.active_frame()
frame.plot().use_lighting_effect=False
plot = frame.plot(PlotType.Cartesian3D)

tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')
#tp.macro.execute_command('''$!FrameControl ActivateByNumber Frame = 1''')
#tp.macro.execute_command('''$!Pick Shift
#                         X = -2.35294117647
#                         Y = 0
#                         PickSubposition = Left''')
#tp.macro.execute_command('''$!Pick Shift
#                         X = 2.32773109244
#                         Y = 0
#                         PickSubposition = Right''')


# %% Setup slice0 with contour0
slice0 = plot.slice(0)
slice0.orientation = SliceSurface.ZPlanes
slice0.origin = (slice0.origin[0], slice0.origin[1], -2.5)

contr0 = plot.contour(0)
contr0.variable = dataset.variable('|grad(rho)|')
contr0.levels.reset_levels([0.2])
slice0.contour.contour_type = ContourType.Lines
slice0.contour.line_color = Color.White
slice0.contour.line_thickness = 0.2
slice0.contour.show = True
contr0.legend.show = False


# %% Setup isosurface
# frame.plot.slice(PlotType.Cartesian3D).show_isosurfaces=True
isosf0 = plot.isosurface(0)
contr1 = plot.contour(1)
contr1.variable = dataset.variable('L2-criterion')
isosf0.definition_contour_group = plot.contour(1)
# isosf0.isosurface_selection = IsoSurfaceSelection.ThreeSpecificValues
isosf0.isosurface_values[0] = -0.0005

contr2 = plot.contour(2)
contr2.colormap_name = 'Small Rainbow'
contr2.variable = dataset.variable('u')
lev2 = np.arange(-0.4, 1.2 + 0.08, 0.08)
contr2.levels.reset_levels(lev2)
contr2.labels.step = 5
contr2.legend.show = True
contr2.legend.number_font.typeface = 'Times'
contr2.legend.vertical = False
contr2.legend.row_spacing = 2.0
contr2.legend.box.box_type = tp.constant.TextBox.None_
isosf0.contour.flood_contour_group = contr2

# %% Show plots
# Turn on Isosurfaces
plot.show_isosurfaces = True
isosf0.show = True
# Turn on slice
plot.show_slices = True
slice0.show = True


# %% Frame parameters setup
# Turn on Translucency
# isosf0.effects.use_translucency = True
# isosf0.effects.surface_translucency = 80

# %% view position and angle
view = plot.view
view.psi = 65
view.theta = 168
view.alpha = -150
view.position = (-50, 270, 120)
view.distance = 300
view.width = 30
# %% axis, tick, label
plot.axes.axis_mode = AxisMode.XYZDependent
x_axes = plot.axes.x_axis
x_axes.show = True
x_axes.min = 0
x_axes.max = 20
x_axes.tick_labels.font.typeface = 'Times'
x_axes.tick_labels.font.size = 3.5
x_axes.tick_labels.offset = 0.5
x_axes.ticks.spacing = 5
x_axes.ticks.length = 1.0
x_axes.ticks.line_thickness = 1.0
x_axes.ticks.minor_length = 0.8
x_axes.ticks.minor_line_thickness = 0.8
tp.macro.execute_command(""" $!AttachText
                         AnchorPos
                         {
                         X = 60
                         Y = 27
                         }
                         TextType=Latex
                         Text = '$x/\\delta_0$' """)

y_axes = plot.axes.y_axis
y_axes.show = True
y_axes.min = -3
y_axes.max = 2
y_axes.tick_labels.font.typeface = 'Times'
y_axes.tick_labels.font.size = 3.5
y_axes.tick_labels.offset = 0.5
y_axes.ticks.spacing = 2
y_axes.ticks.length = 1.0
y_axes.ticks.line_thickness = 1.0
y_axes.ticks.minor_length = 0.8
y_axes.ticks.minor_line_thickness = 0.8
tp.macro.execute_command(""" $!AttachText
                         AnchorPos
                         {
                         X = 20
                         Y = 27
                         }
                         TextType=Latex
                         Text = '$y/\\delta_0$' """)

z_axes = plot.axes.z_axis
z_axes.show = True
z_axes.min = -8.0
z_axes.max = 8.2
z_axes.ticks.spacing = 2
z_axes.tick_labels.font.typeface = 'Times'
z_axes.tick_labels.font.size = 3.5
z_axes.tick_labels.offset = 0.5
z_axes.ticks.spacing = 5
z_axes.ticks.length = 1.0
z_axes.ticks.line_thickness = 1.0
z_axes.ticks.minor_length = 0.8
z_axes.ticks.minor_line_thickness = 0.8
tp.macro.execute_command(""" $!AttachText
                         AnchorPos
                         {
                         X = 40
                         Y = 30
                         }
                         TextType=Latex
                         Text = '$z/\\delta_0$' """)

tp.export.save_png(pathF + 'test.png', width=4096, supersample=3) 
