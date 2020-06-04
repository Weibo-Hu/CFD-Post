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

path = "/media/weibo/IM1/BFS_M1.7Tur/3D_DMD_1200/"
pathin = path + "0p085/"
pathout = path + "0p085_ani/"
# %% load data 
# tp.session.connect()
file = '[0.085]DMD'
dirs = os.listdir(pathin)
num = int(np.size(dirs)/2)
val1 = -0.06  # for u
val2 = -0.004  # for p
# num = 1
for ii in range(num):
    ind = '{:03}'.format(ii)
    filelist = [file+ind+'A.plt', file+ind+'B.plt']
    print(filelist)
    datafile = [os.path.join(pathin, name) for name in filelist]
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

    # create contour for isosurfaces for velocity
    cont = plot.contour(0)
    cont.colormap_name = 'Diverging - Blue/Red'
    cont.variable = dataset.variable('u`')
    cont.levels.reset_levels(np.linspace(val1/2*3, -val1/2*3, 7))
    cont.legend.show = False
    # create isosurfaces
    plot.show_isosurfaces = True
    iso = plot.isosurface(0)
    iso.contour.flood_contour_group = cont
    iso.definition_contour_group.variable = dataset.variable('u`')
    iso.isosurface_selection = IsoSurfaceSelection.TwoSpecificValues
    iso.isosurface_values = (val1, -val1)
    iso.effects.lighting_effect = LightingEffect.Gouraud
    iso.contour.show = True
    iso.contour.use_lighting_effect = True
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont.colormap_filter.continuous_min = val1/2*3
    cont.colormap_filter.continuous_max = -val1/2*3
    iso.effects.use_translucency = True
    iso.effects.surface_translucency = 30

    # contour for slice
    cont1 = plot.contour(5)
    cont1.colormap_name = 'Diverging - Purple/Green'
    cont1.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont1.colormap_filter.continuous_min = val2/2*3
    cont1.colormap_filter.continuous_max = -val2/2*3
    cont1.levels.reset_levels(np.linspace(val2/2*3, -val2/2*3, 7))
    cont1.variable = dataset.variable('p`')
    cont1.legend.show = True
    cont1.legend.vertical = False
    cont1.legend.row_spacing=1.3
    cont1.legend.number_font.typeface='Times New Roman'
    cont1.legend.show_header=False
    cont1.legend.box.box_type=tp.constant.TextBox.None_
    cont1.labels.step=2
    cont1.legend.position=(28, 89.5)
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 78.5
          Y = 90
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 3.6
          }
        TextType = LaTeX
        Text = '$\\phi$='""")
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 81.5
          Y = 90.2
          }
        TextShape
          {
          FontFamily = 'Times New Roman'
          IsBold = No
          SizeUnits = Frame
          Height = 3.6
          }
        Text = '&(solutiontime)'""")
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 83.7
          Y = 89.4
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 3.6
          }
        TextType = LaTeX
        Text = '$\\pi/16$'""")
    # create slice
    plot.show_slices = True
    slices = plot.slice(0)
    slices.show = True
    slices.orientation = SliceSurface.ZPlanes
    slices.origin = (slices.origin[0], slices.origin[1], -8)
    slices.contour.flood_contour_group = cont1

    # export figs
    tp.export.save_png(pathout + file + str(SolTime) + '.png', width=2048)
    # tp.export.save_jpeg(path + file + str(SolTime) + '.jpg', width=2048, quality=100) 
    
# %% generate animation
# %% Convert plots to animation
import imageio
from glob import glob
from natsort import natsorted, ns
dirs = glob(pathout + '*.png')
dirs = natsorted(dirs, key=lambda y: y.lower())
flnm = path + file + 'Anima.mp4'
with imageio.get_writer(flnm, mode='I', fps=12, macro_block_size=None) as writer:   
    for ii in range(np.size(dirs)*6):
        ind = ii % 32  # mod, get reminder
        image = imageio.imread(dirs[ind])
        writer.append_data(image)

