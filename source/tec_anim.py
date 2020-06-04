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
pathin = path + "0p022/"
pathout = path + "0p022_ani/"

def limit_val(dataset, var):
    minv = dataset.variable(var).min()
    maxv = dataset.variable(var).max()
    print("Limit value of " + var, minv, maxv)

def plt_isosurf(iso, cont, var, val):
    # create contour for isosurfaces for velocity
    cont.colormap_name = 'Diverging - Blue/Red'
    cont.variable = dataset.variable(var)
    cont.levels.reset_levels(np.linspace(val/2*3, -val/2*3, 7))
    cont.legend.show = False
    # create isosurfaces
    plot.show_isosurfaces = True
    iso = plot.isosurface(0)
    iso.contour.flood_contour_group = cont
    iso.definition_contour_group.variable = dataset.variable('u`')
    iso.isosurface_selection = IsoSurfaceSelection.TwoSpecificValues
    iso.isosurface_values = (val, -val)
    iso.effects.lighting_effect = LightingEffect.Gouraud
    iso.contour.show = True
    iso.contour.use_lighting_effect = True
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont.colormap_filter.continuous_min = val/2*3
    cont.colormap_filter.continuous_max = -val/2*3
    iso.effects.use_translucency = True
    iso.effects.surface_translucency = 30

def plt_slice(slc, cont, var, val, label=True):
    # contour for slice
    plot.show_slices = True
    cont = plot.contour(5)
    cont.colormap_name = 'Diverging - Purple/Green'
    cont.colormap_filter.distribution = ColorMapDistribution.Continuous
    cont.colormap_filter.continuous_min = val/2*3
    cont.colormap_filter.continuous_max = -val/2*3
    cont.levels.reset_levels(np.linspace(val/2*3, -val/2*3, 7))
    cont.variable = dataset.variable(var)
    cont.legend.show = True
    cont.legend.vertical = False
    cont.legend.row_spacing=1.3
    cont.legend.number_font.typeface='Times New Roman'
    cont.legend.show_header=False
    cont.legend.box.box_type=tp.constant.TextBox.None_
    cont.labels.step=2
    cont.legend.position=(28, 89.5)

    slc.show = True
    slc.orientation = SliceSurface.ZPlanes
    slc.origin = (slc.origin[0], slc.origin[1], -8)
    slc.contour.flood_contour_group = cont
    
    if label == True:
        tp.macro.execute_command("""$!AttachText 
            AnchorPos
              {
              X = 27
              Y = 85.2
              }
            TextShape
              {
              SizeUnits = Frame
              Height = 4.2
              }
            TextType = LaTeX
            Text = '$p^{\\prime}$'""")


def axis_set(axes_obj):
    axes_obj.grid_area.filled=False
    axes_obj.x_axis.show=True
    axes_obj.x_axis.min=-5
    axes_obj.x_axis.max=20
    axes_obj.x_axis.tick_labels.font.size=3.0
    axes_obj.x_axis.tick_labels.font.typeface='Times'
    axes_obj.x_axis.title.show=False

    axes_obj.y_axis.show=True
    axes_obj.y_axis.min=-3
    axes_obj.y_axis.max=2
    axes_obj.y_axis.tick_labels.font.size=3.0
    axes_obj.y_axis.tick_labels.font.typeface='Times'
    axes_obj.y_axis.title.show=False
    axes_obj.y_axis.ticks.show_on_opposite_edge=True
    axes_obj.y_axis.tick_labels.show_on_opposite_edge=True
    axes_obj.y_axis.ticks.show=False
    axes_obj.y_axis.tick_labels.show=False
    
    axes_obj.z_axis.show=True
    axes_obj.z_axis.min=-8
    axes_obj.z_axis.max=8
    axes_obj.z_axis.tick_labels.font.size=3.0
    axes_obj.z_axis.tick_labels.font.typeface='Times'
    axes_obj.z_axis.title.show=False
    #axes_obj.x_axis.title.show_on_opposite_edge=True
    #axes_obj.x_axis.title.show_on_opposite_edge=False

def axis_lab():
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 60
          Y = 12
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 4.2
          }
        TextType = LaTeX
        Text = '$x/\\delta_0$'""")
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 4.0
          Y = 74
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 4.2
          }
        TextType = LaTeX
        Text = '$y/\\delta_0$'""")
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 12.5
          Y = 18
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 4.2
          }
        TextType = LaTeX
        Text = '$z/\\delta_0$'""")

def figure_ind():
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 5.0
          Y = 92
          }
        TextShape
          {
          FontFamily = 'Times New Roman'
          IsBold = No
          SizeUnits = Frame
          Height = 3.6
          }
        Text = '(a)'""")

def show_time():
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 80.5
          Y = 90
          }
        TextShape
          {
          SizeUnits = Frame
          Height = 3.6
          }
        TextType = LaTeX
        Text = '$\\theta=\\frac{\\pi}{16}\\times$'""")
    tp.macro.execute_command("""$!AttachText 
        AnchorPos
          {
          X = 88.0
          Y = 91.1
          }
        TextShape
          {
          FontFamily = 'Times New Roman'
          IsBold = No
          SizeUnits = Frame
          Height = 3.0
          }
        Text = '&(solutiontime)'""")

def show_wall(plot):
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = 0
        Y1 = -3
        Z1 = -8
        X2 = 20
        Y2 = -3
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = 0
        Y1 = -3
        Z1 = -8
        X2 = 0
        Y2 = 0
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    tp.macro.execute_command('''$!CreateRectangularZone 
        IMax = 10
        JMax = 10
        KMax = 10
        X1 = -5
        Y1 = 0
        Z1 = -8
        X2 = 0
        Y2 = 0
        Z2 = 8
        XVar = 1
        YVar = 2
        ZVar = 3''')
    plot.use_lighting_effect=True
    plot.show_shade=True
    plot.fieldmap(1).shade.show=True
    plot.fieldmap(2).shade.show=True
    plot.fieldmap(3).shade.show=True
    plot.fieldmap(1).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces
    plot.fieldmap(2).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces
    plot.fieldmap(3).surfaces.surfaces_to_plot = \
        SurfacesToPlot.BoundaryFaces

# %% load data 
# tp.session.connect()
file = '[0.022]DMD'
dirs = os.listdir(pathin)
num = int(np.size(dirs)/2)
val1 = -0.42  # for u
val2 = -0.02  # for p
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
    # frame.load_stylesheet(path + 'video.sty')
    # turn off orange zone bounding box
    tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')
    # frame setting
    frame.width = 12
    frame.height = 7.2
    frame.position = (-1.0, 0.8)
    plot = frame.plot(PlotType.Cartesian3D)
    plot.axes.orientation_axis.show=False
    axes = plot.axes
    axis_set(axes)
    axis_lab()

    # 3d view settings
    view = plot.view
    view.magnification = 1.1
    # view.fit_to_nice()
    view.rotation_origin = (10, 0.0, 0.0)
    view.psi = 45
    view.theta = 145
    view.alpha = -140
    view.position = (-46.5, 76, 94)
    # view.distance = 300
    view.width = 36
    
    limit_val(dataset, 'u`')
    # create contour for isosurfaces for velocity
    cont = plot.contour(0)
    # create isosurfaces
    iso = plot.isosurface(0)
    plt_isosurf(iso, cont, 'u`', val1)

    # contour for slice
    limit_val(dataset, 'p`')
    cont1 = plot.contour(5)
    slices = plot.slice(0)
    # plt_slice(slices, cont1, 'p`', val2)

    # show_time()
    figure_ind()
    show_wall(plot)

    # export figs
    tp.export.save_png(pathout + file + str(SolTime) + '.png', width=2048)
    # tp.export.save_jpeg(pathout + file + str(SolTime) + '.jpg', width=2048, quality=100) 
    
# %% generate animation
# %% Convert plots to animation
# import imageio
# from glob import glob
# from natsort import natsorted, ns
# dirs = glob(pathout + '*.png')
# dirs = natsorted(dirs, key=lambda y: y.lower())
# flnm = path + file + 'Anima.mp4'
# with imageio.get_writer(flnm, mode='I', fps=12, macro_block_size=None) as writer:   
#     for ii in range(np.size(dirs)*6):
#         ind = ii % 32  # mod, get reminder
#         image = imageio.imread(dirs[ind])
#         writer.append_data(image)

